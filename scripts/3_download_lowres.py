#!/usr/bin/env python3
"""Step 3: Download low-res IA variants for relevant records only.

Downloads up to 5 files in parallel with rich progress bars.
Periodically re-reads the classified JSONL to pick up new records
added by an upstream process running concurrently.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

_DATETIME_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_")

import aiohttp
import aiofiles
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.theme import Theme

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from astro_ia_harvest.config import (  # noqa: E402
    CLASSIFIED_JSONL,
    DOWNLOAD_DIR,
    DOWNLOAD_FAILURES_JSONL,
    DOWNLOAD_LOG_CSV,
    EXISTING_DOWNLOAD_DIR,
    ensure_directories,
)
from astro_ia_harvest.download_utils import (  # noqa: E402
    build_candidate_urls,
    build_output_name,
    canonical_video_key,
)
from astro_ia_harvest.jsonl_utils import load_jsonl  # noqa: E402


def _build_content_signature_map(rows: list[dict]) -> dict[str, str]:
    """Build a map from content-signature to canonical_video_key.

    NASA sometimes uploads the same video under multiple IA identifiers
    with different filenames.  By grouping classified candidates on
    ``(video_size, video_length)`` we can detect these content duplicates
    *before* downloading.  When two records share the same signature, the
    first one seen wins; subsequent ones are skipped.

    Returns ``{signature_str: canonical_video_key}``.
    """
    sig_map: dict[str, str] = {}
    for r in rows:
        if not r.get("likely_relevant"):
            continue
        size = r.get("video_size")
        length = r.get("video_length")
        if not size or not length:
            continue
        sig = f"{size}|{length}"
        ck = canonical_video_key(str(r.get("filename", "")))
        if sig not in sig_map:
            sig_map[sig] = ck
    return sig_map

MAX_CONCURRENT_DOWNLOADS = 5
MAX_RETRIES = 3
RETRY_BACKOFF = [15, 45, 120]  # seconds to wait before retry 1, 2, 3
SOCK_READ_TIMEOUT = 120       # abort if no data received for 120 s
PER_RECORD_TIMEOUT = 45 * 60  # 45-minute safety-net per record
THROTTLE_WINDOW = 60          # seconds — if all recent failures fall in this window…
THROTTLE_THRESHOLD = 5        # …and there are this many, trigger a cooldown
THROTTLE_COOLDOWN = 120       # seconds to pause all downloads when throttled
RETRYABLE_HTTP = {429, 503, 502, 504}  # server-side retryable status codes

# Shared mutable state for throttle detection across workers
_recent_failures: list[float] = []   # timestamps of recent timeout/network failures
_throttle_lock = None  # initialised to asyncio.Lock() inside async_main

# Persistent skip-list: records that returned HTTP 404 on all URLs are never retried.
# Keys are derived from download_failures.jsonl (records whose error contains "http_404").
_404_skip_keys: set[str] = set()


def load_404_skip_keys() -> None:
    """Populate _404_skip_keys from download_failures.jsonl (error contains 'http_404')."""
    if DOWNLOAD_FAILURES_JSONL.exists():
        for line in DOWNLOAD_FAILURES_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "http_404" in str(entry.get("error", "")):
                ident = entry.get("identifier", "")
                fname = entry.get("filename", "")
                if ident or fname:
                    _404_skip_keys.add(f"{ident}|{fname}")


console = Console(
    theme=Theme(
        {
            "success": "green",
            "error": "bold red",
            "info": "cyan",
            "warning": "yellow",
        }
    )
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def load_existing_video_keys() -> tuple[set[str], dict[str, Path]]:
    """Return (all_keys, existing_dir_map) where existing_dir_map maps
    canonical_key -> source Path for files in EXISTING_DOWNLOAD_DIR that match
    a likely_relevant record in the classified JSONL (i.e. ones we actually want)."""
    keys: set[str] = set()
    existing_dir_map: dict[str, Path] = {}

    for p in DOWNLOAD_DIR.rglob("*.mp4"):
        keys.add(canonical_video_key(p.name))
        # For files we created with build_output_name (identifier__stem_lowres.mp4),
        # also key on just the IA filename portion so it matches the source filename.
        if "__" in p.stem:
            _, _, ia_part = p.name.partition("__")
            keys.add(canonical_video_key(ia_part))

    if EXISTING_DOWNLOAD_DIR.exists():
        # Only copy files that are actually relevant — build the allowed key set first.
        relevant_keys: set[str] = set()
        for r in load_jsonl(CLASSIFIED_JSONL):
            if r.get("likely_relevant"):
                fname = str(r.get("filename", ""))
                if fname:
                    relevant_keys.add(canonical_video_key(fname))

        for p in EXISTING_DOWNLOAD_DIR.rglob("*.mp4"):
            ck = canonical_video_key(p.name)
            keys.add(ck)
            if ck in relevant_keys:
                existing_dir_map[ck] = p

    return keys, existing_dir_map


def _short_id(identifier: str, filename: str) -> str:
    candidate = identifier or filename or "unknown"
    mx = 24
    if len(candidate) <= mx:
        tag = candidate.ljust(mx)
    else:
        tag = f"{candidate[:11]}...{candidate[-10:]}"
    # Escape square brackets so rich doesn't interpret them as markup
    return f"\\[{tag}]"


def append_failure(record: dict, message: str) -> None:
    payload = {
        "identifier": record.get("identifier"),
        "filename": record.get("filename"),
        "error": message,
        "ts": int(time.time()),
    }
    with open(DOWNLOAD_FAILURES_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_csv_row(filename: str, identifier: str, status: str, detail: str) -> None:
    exists = DOWNLOAD_LOG_CSV.exists()
    with open(DOWNLOAD_LOG_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "identifier", "status", "detail"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "filename": filename,
                "identifier": identifier,
                "status": status,
                "detail": detail,
            }
        )


def _record_key(rec: dict) -> str:
    """Unique key for a JSONL record so we can detect duplicates across refreshes."""
    return f"{rec.get('identifier', '')}|{rec.get('filename', '')}"


# ------------------------------------------------------------------
# Async download
# ------------------------------------------------------------------


async def _check_throttle(tag: str) -> None:
    """If recent failures suggest server-side throttling, pause with a cooldown."""
    global _recent_failures
    should_cool = False
    async with _throttle_lock:
        now = time.time()
        # Prune old entries
        _recent_failures = [t for t in _recent_failures if now - t < THROTTLE_WINDOW]
        if len(_recent_failures) >= THROTTLE_THRESHOLD:
            console.print(
                f"[warning]{tag} Throttle detected "
                f"({len(_recent_failures)} failures in {THROTTLE_WINDOW}s) — "
                f"cooling down {THROTTLE_COOLDOWN}s…[/warning]"
            )
            _recent_failures.clear()  # reset so only one cooldown fires
            should_cool = True
    # Sleep outside the lock so other coroutines can proceed
    if should_cool:
        await asyncio.sleep(THROTTLE_COOLDOWN)
        console.print(f"[info]{tag} Resuming after cooldown[/info]")


async def _record_failure_ts() -> None:
    async with _throttle_lock:
        _recent_failures.append(time.time())


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    out_path: Path,
    ident_tag: str,
    label: str,
    progress: Progress,
    active_downloads: set[Path],
) -> tuple[bool, str]:
    """Download a single URL to out_path with a rich progress bar task."""
    tmp_path = Path(str(out_path) + ".tmp")
    try:
        active_downloads.add(out_path)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        # HEAD to check availability and get content-length (non-fatal on timeout)
        file_size = 0
        try:
            head_timeout = aiohttp.ClientTimeout(total=60, connect=30)
            async with session.head(url, timeout=head_timeout, headers=headers, allow_redirects=True) as hr:
                if hr.status == 404:
                    return False, f"{label}:http_404"
                if hr.status in RETRYABLE_HTTP:
                    await _record_failure_ts()
                    return False, f"{label}:http_{hr.status}"
                if hr.status not in (200, 302):
                    return False, f"{label}:http_{hr.status}"
                file_size = int(hr.headers.get("content-length", 0))
        except (asyncio.TimeoutError, aiohttp.ClientError):
            # HEAD timed out — not fatal, proceed to GET without known size
            pass

        # GET and stream to file
        dl_timeout = aiohttp.ClientTimeout(
            connect=30, total=0, sock_read=SOCK_READ_TIMEOUT
        )
        async with session.get(url, timeout=dl_timeout, headers=headers, allow_redirects=True) as resp:
            if resp.status in RETRYABLE_HTTP:
                await _record_failure_ts()
                return False, f"{label}:http_{resp.status}"
            if resp.status != 200:
                return False, f"{label}:http_{resp.status}"

            # If HEAD didn't give us a size, try from GET response
            if not file_size:
                file_size = int(resp.headers.get("content-length", 0))

            out_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = 0
            start = time.time()

            disp = out_path.name
            if len(disp) > 40:
                disp = disp[:37] + "..."
            task_id = progress.add_task(disp, ident=ident_tag, total=file_size or None)

            try:
                async with aiofiles.open(tmp_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(256 * 1024):
                        await f.write(chunk)
                        downloaded += len(chunk)
                        progress.update(task_id, completed=downloaded)
            finally:
                progress.remove_task(task_id)

            # Rename tmp -> final (retry for Windows file handle release delay)
            for attempt in range(5):
                try:
                    tmp_path.rename(out_path)
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    await asyncio.sleep(0.2 * (attempt + 1))

            mb = downloaded / (1024 * 1024)
            elapsed = time.time() - start
            speed = mb / elapsed if elapsed > 0 else 0
            console.print(
                f"{ident_tag} [success]Done:[/success] {out_path.name} "
                f"[info]({mb:.1f} MB in {elapsed:.1f}s, {speed:.1f} MB/s)[/info]"
            )
            return True, "ok"

    except (asyncio.TimeoutError, asyncio.CancelledError):
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        await _record_failure_ts()
        return False, f"{label}:timeout"
    except aiohttp.ClientError as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        await _record_failure_ts()
        return False, f"{label}:network_error:{exc}"
    except Exception as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False, f"{label}:network_error:{exc}"
    finally:
        active_downloads.discard(out_path)


async def process_record(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    rec: dict,
    existing_keys: set[str],
    progress: Progress,
    active_downloads: set[Path],
    dry_run: bool = False,
) -> str:
    """Process one record. Returns 'downloaded', 'skip_existing', 'dry_run', or 'failed'."""
    async with semaphore:
        ident = str(rec.get("identifier", ""))
        filename = str(rec.get("filename", ""))
        if not ident or not filename:
            return "failed"

        source_key = canonical_video_key(filename)
        output_name = build_output_name(ident, filename)
        output_path = DOWNLOAD_DIR / output_name

        if output_path.exists():
            append_csv_row(output_name, ident, "skip_existing", "already_in_download_dir")
            return "skip_existing"

        if source_key in existing_keys:
            append_csv_row(filename, ident, "skip_existing", "matched_existing_key")
            return "skip_existing"

        urls = build_candidate_urls(ident, filename)
        tag = _short_id(ident, filename)

        if dry_run:
            append_csv_row(output_name, ident, "dry_run", urls[0][1] if urls else "no_candidate_url")
            return "dry_run"

        console.print(f"{tag} [info]Starting download...[/info]")

        last_detail = "no_candidate_url"
        for url, label in urls:
            for attempt in range(1, MAX_RETRIES + 2):  # 1 try + MAX_RETRIES retries
                # Before each attempt, check if we should back off globally
                await _check_throttle(tag)

                ok, detail = await download_file(
                    session, url, output_path, tag, label, progress, active_downloads
                )
                if ok:
                    existing_keys.add(source_key)
                    append_csv_row(output_name, ident, "success", label)
                    return "downloaded"
                last_detail = detail
                is_retryable = any(tok in detail for tok in (
                    "timeout", "network_error",
                    "http_429", "http_503", "http_502", "http_504",
                ))
                if is_retryable and attempt <= MAX_RETRIES:
                    wait = RETRY_BACKOFF[attempt - 1]
                    console.print(
                        f"{tag} [warning]Retry {attempt}/{MAX_RETRIES} "
                        f"in {wait}s ({detail})[/warning]"
                    )
                    await asyncio.sleep(wait)
                else:
                    break  # non-retryable or retries exhausted

        # All attempts failed
        append_csv_row(output_name, ident, "failure", last_detail)
        append_failure(rec, last_detail)
        try:
            Path(str(output_path) + ".tmp").unlink(missing_ok=True)
        except OSError:
            pass
        console.print(f"{tag} [error]All attempts failed:[/error] {last_detail}")
        # Persist 404s so they are never retried in future runs
        if "http_404" in last_detail:
            rk = _record_key(rec)
            if rk not in _404_skip_keys:
                _404_skip_keys.add(rk)
            console.print(f"{tag} [warning]Saved to 404 skip-list (via failures JSONL)[/warning]")
        return "failed"


# ------------------------------------------------------------------
# Main loop with JSONL refresh
# ------------------------------------------------------------------


def copy_from_existing_dir(
    existing_dir_map: dict[str, Path],
    existing_keys: set[str],
    dry_run: bool,
) -> int:
    """Copy relevant files from EXISTING_DOWNLOAD_DIR to DOWNLOAD_DIR with the
    datetime prefix stripped.  Updates existing_keys in-place so the main loop
    treats copied files as already present.  Returns the number of files copied."""
    copied = 0
    for ck, src in sorted(existing_dir_map.items()):
        clean_name = _DATETIME_PREFIX_RE.sub("", src.name)
        dest = DOWNLOAD_DIR / clean_name
        if dest.exists():
            # Already copied on a previous run
            existing_keys.add(canonical_video_key(clean_name))
            continue
        if dry_run:
            console.print(f"[info]Would copy:[/info] {clean_name}")
            copied += 1
            continue
        try:
            shutil.copy2(src, dest)
            existing_keys.add(canonical_video_key(clean_name))
            console.print(f"[info]Copied from legacy dir:[/info] {clean_name}")
            copied += 1
        except Exception as exc:
            console.print(f"[error]Copy failed ({src.name}):[/error] {exc}")
    return copied


async def async_main(dry_run: bool, limit: int) -> None:
    ensure_directories()
    load_404_skip_keys()
    existing_keys, existing_dir_map = load_existing_video_keys()

    actual_file_count = sum(1 for _ in DOWNLOAD_DIR.rglob("*.mp4")) if DOWNLOAD_DIR.exists() else 0

    console.print("=" * 70)
    console.print("[info]STEP 3: Download Low-Res Videos[/info]")
    console.print("=" * 70)
    console.print(f"Parallel downloads: {MAX_CONCURRENT_DOWNLOADS}")
    console.print(f"Videos in download dir: {actual_file_count}")
    console.print(f"Existing dedup keys (covers filename variants): {len(existing_keys)}")
    console.print(f"  (of which in legacy dir to copy: {len(existing_dir_map)})")
    console.print(f"404 skip-list entries: {len(_404_skip_keys)}")
    console.print()

    if existing_dir_map:
        console.print(
            f"[info]Copying {len(existing_dir_map)} file(s) from legacy dir to "
            f"{DOWNLOAD_DIR}…[/info]"
        )
        n_copied = copy_from_existing_dir(existing_dir_map, existing_keys, dry_run)
        console.print(f"[info]Done — {n_copied} file(s) copied.[/info]")
        console.print()

    global _throttle_lock
    _throttle_lock = asyncio.Lock()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
    session = aiohttp.ClientSession(
        connector=connector,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
    )
    active_downloads: set[Path] = set()

    progress = Progress(
        TextColumn("{task.fields[ident]}", justify="left", style="cyan"),
        TextColumn("{task.description}", justify="left", style="white"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        refresh_per_second=5,
    )

    downloaded = 0
    skipped_existing = 0
    skipped_prequeue = 0  # relevant records whose file already existed, never queued
    failed = 0
    total_processed = 0
    seen_keys: set[str] = set()  # tracks records already queued/processed
    queued_video_keys: set[str] = set()  # canonical keys for in-flight dedup

    try:
        with progress:
            while True:
                # (Re-)read the JSONL to discover new records
                rows = load_jsonl(CLASSIFIED_JSONL)
                targets = [r for r in rows if bool(r.get("likely_relevant"))]

                # Find only records we haven't queued yet, deduplicating by
                # canonical video key so .mxf/.ia.mp4/.mp4 entries for the same
                # video don't race to download the same output file.
                # Also skip content-duplicates: different IA identifiers whose
                # (video_size, video_length) match an already-queued record.
                content_sigs = _build_content_signature_map(targets)
                new_targets: list[dict] = []
                for r in targets:
                    rk = _record_key(r)
                    if rk in seen_keys or rk in _404_skip_keys:
                        continue
                    seen_keys.add(rk)
                    ck = canonical_video_key(str(r.get("filename", "")))
                    if ck in existing_keys or ck in queued_video_keys:
                        skipped_prequeue += 1
                        continue
                    # Content-signature dedup: skip if a different file with
                    # identical (size, length) has already been queued/downloaded.
                    size = r.get("video_size")
                    length = r.get("video_length")
                    if size and length:
                        sig = f"{size}|{length}"
                        first_ck = content_sigs.get(sig)
                        if first_ck and first_ck != ck and (
                            first_ck in existing_keys or first_ck in queued_video_keys
                        ):
                            skipped_prequeue += 1
                            continue
                    queued_video_keys.add(ck)
                    new_targets.append(r)

                if limit > 0:
                    remaining = limit - total_processed
                    if remaining <= 0:
                        break
                    new_targets = new_targets[:remaining]

                if not new_targets:
                    break

                console.print(
                    f"[info]Queued {len(new_targets)} new record(s) "
                    f"({skipped_prequeue} already have a file; "
                    f"{len(seen_keys)} relevant JSONL records total)[/info]"
                )

                # Launch downloads for this batch — semaphore limits concurrency
                tasks = [
                    asyncio.create_task(
                        asyncio.wait_for(
                            process_record(
                                session, semaphore, rec, existing_keys,
                                progress, active_downloads, dry_run,
                            ),
                            timeout=PER_RECORD_TIMEOUT,
                        )
                    )
                    for rec in new_targets
                ]

                # As each task completes, tally and re-check JSONL for new records
                pending = set(tasks)
                while pending:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in done:
                        try:
                            result = t.result()
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            result = "failed"
                            console.print(
                                "[error]Record exceeded per-record "
                                f"timeout ({PER_RECORD_TIMEOUT}s), aborted[/error]"
                            )
                        except Exception as exc:
                            result = "failed"
                            console.print(
                                f"[error]Unexpected error: {exc}[/error]"
                            )
                        total_processed += 1
                        if result == "downloaded":
                            downloaded += 1
                        elif result in ("skip_existing", "dry_run"):
                            skipped_existing += 1
                        else:
                            failed += 1

                    # After each completion batch, check for new JSONL records
                    rows = load_jsonl(CLASSIFIED_JSONL)
                    fresh = [r for r in rows if bool(r.get("likely_relevant"))]
                    new_batch: list[dict] = []
                    for r in fresh:
                        rk = _record_key(r)
                        if rk in seen_keys or rk in _404_skip_keys:
                            continue
                        seen_keys.add(rk)
                        ck = canonical_video_key(str(r.get("filename", "")))
                        if ck in existing_keys or ck in queued_video_keys:
                            skipped_prequeue += 1
                            continue
                        queued_video_keys.add(ck)
                        new_batch.append(r)
                    if limit > 0:
                        remaining = limit - total_processed - len(pending)
                        new_batch = new_batch[:max(0, remaining)]
                    if new_batch:
                        console.print(
                            f"[info]JSONL refresh: {len(new_batch)} new record(s) queued[/info]"
                        )
                        for rec in new_batch:
                            task = asyncio.create_task(
                                asyncio.wait_for(
                                    process_record(
                                        session, semaphore, rec, existing_keys,
                                        progress, active_downloads, dry_run,
                                    ),
                                    timeout=PER_RECORD_TIMEOUT,
                                )
                            )
                            pending.add(task)
    finally:
        # Clean up any partial downloads
        for p in list(active_downloads):
            if p.exists():
                try:
                    console.print(f"[warning]Cleaning partial:[/warning] {p.name}")
                    p.unlink()
                except Exception as e:
                    console.print(f"[error]Cleanup error {p.name}: {e}[/error]")
        await session.close()

    console.print()
    console.print("[info]Done[/info]")
    console.print(f"Relevant JSONL records:  {len(seen_keys)}")
    console.print(f"  Already had a file:    {skipped_prequeue}  (skipped before queuing — file existed)")
    console.print(f"  Queued this run:       {total_processed}")
    console.print(f"    Downloaded:          {downloaded}")
    console.print(f"    Skipped (task saw existing file): {skipped_existing}")
    console.print(f"    Failed:              {failed}")
    console.print(f"Output dir: {DOWNLOAD_DIR}  ({actual_file_count} files before this run)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download low-res IA video files")
    parser.add_argument("--limit", type=int, default=0, help="Max relevant records to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    asyncio.run(async_main(args.dry_run, args.limit))


if __name__ == "__main__":
    main()
