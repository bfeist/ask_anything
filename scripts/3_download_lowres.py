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
import sys
import time
from pathlib import Path

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

MAX_CONCURRENT_DOWNLOADS = 5

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


def load_existing_video_keys() -> set[str]:
    keys: set[str] = set()
    for p in DOWNLOAD_DIR.rglob("*.mp4"):
        keys.add(canonical_video_key(p.name))
        # For files we created with build_output_name (identifier__stem_lowres.mp4),
        # also key on just the IA filename portion so it matches the source filename.
        if "__" in p.stem:
            _, _, ia_part = p.name.partition("__")
            keys.add(canonical_video_key(ia_part))
    if EXISTING_DOWNLOAD_DIR.exists():
        for p in EXISTING_DOWNLOAD_DIR.rglob("*.mp4"):
            keys.add(canonical_video_key(p.name))
    return keys


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

        # HEAD to check availability and get content-length
        head_timeout = aiohttp.ClientTimeout(total=30, connect=10)
        async with session.head(url, timeout=head_timeout, headers=headers, allow_redirects=True) as hr:
            if hr.status == 404:
                return False, f"{label}:http_404"
            if hr.status not in (200, 302):
                return False, f"{label}:http_{hr.status}"
            file_size = int(hr.headers.get("content-length", 0))

        # GET and stream to file
        dl_timeout = aiohttp.ClientTimeout(connect=30, total=0)
        async with session.get(url, timeout=dl_timeout, headers=headers, allow_redirects=True) as resp:
            if resp.status != 200:
                return False, f"{label}:http_{resp.status}"

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

    except asyncio.TimeoutError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False, f"{label}:timeout"
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
        if source_key in existing_keys:
            append_csv_row(filename, ident, "skip_existing", "matched_existing_key")
            return "skip_existing"

        output_name = build_output_name(ident, filename)
        output_path = DOWNLOAD_DIR / output_name
        if output_path.exists():
            append_csv_row(output_name, ident, "skip_existing", "already_in_download_dir")
            return "skip_existing"

        urls = build_candidate_urls(ident, filename)
        tag = _short_id(ident, filename)

        if dry_run:
            append_csv_row(output_name, ident, "dry_run", urls[0][1] if urls else "no_candidate_url")
            return "dry_run"

        console.print(f"{tag} [info]Starting download...[/info]")

        last_detail = "no_candidate_url"
        for url, label in urls:
            ok, detail = await download_file(
                session, url, output_path, tag, label, progress, active_downloads
            )
            if ok:
                existing_keys.add(source_key)
                append_csv_row(output_name, ident, "success", label)
                return "downloaded"
            last_detail = detail

        # All attempts failed
        append_csv_row(output_name, ident, "failure", last_detail)
        append_failure(rec, last_detail)
        try:
            Path(str(output_path) + ".tmp").unlink(missing_ok=True)
        except OSError:
            pass
        console.print(f"{tag} [error]All attempts failed:[/error] {last_detail}")
        return "failed"


# ------------------------------------------------------------------
# Main loop with JSONL refresh
# ------------------------------------------------------------------


async def async_main(dry_run: bool, limit: int) -> None:
    ensure_directories()
    existing_keys = load_existing_video_keys()

    console.print("=" * 70)
    console.print("[info]STEP 3: Download Low-Res Videos[/info]")
    console.print("=" * 70)
    console.print(f"Parallel downloads: {MAX_CONCURRENT_DOWNLOADS}")
    console.print(f"Existing normalized video keys: {len(existing_keys)}")
    console.print()

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
                new_targets: list[dict] = []
                for r in targets:
                    rk = _record_key(r)
                    if rk in seen_keys:
                        continue
                    seen_keys.add(rk)
                    ck = canonical_video_key(str(r.get("filename", "")))
                    if ck in existing_keys or ck in queued_video_keys:
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
                    f"(total seen: {len(seen_keys)})[/info]"
                )

                # Launch downloads for this batch — semaphore limits concurrency
                tasks = [
                    asyncio.create_task(
                        process_record(
                            session, semaphore, rec, existing_keys,
                            progress, active_downloads, dry_run,
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
                        result = t.result()
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
                        if rk in seen_keys:
                            continue
                        seen_keys.add(rk)
                        ck = canonical_video_key(str(r.get("filename", "")))
                        if ck in existing_keys or ck in queued_video_keys:
                            continue
                        queued_video_keys.add(ck)
                        new_batch.append(r)
                    if limit > 0:
                        remaining = limit - total_processed - len(pending)
                        new_batch = new_batch[:max(0, remaining)]
                    if new_batch:
                        console.print(
                            f"[info]JSONL refresh: {len(new_batch)} new record(s) found[/info]"
                        )
                        for rec in new_batch:
                            task = asyncio.create_task(
                                process_record(
                                    session, semaphore, rec, existing_keys,
                                    progress, active_downloads, dry_run,
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
    console.print(f"Downloaded: {downloaded}")
    console.print(f"Skipped existing: {skipped_existing}")
    console.print(f"Failed: {failed}")
    console.print(f"Output dir: {DOWNLOAD_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download low-res IA video files")
    parser.add_argument("--limit", type=int, default=0, help="Max relevant records to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    asyncio.run(async_main(args.dry_run, args.limit))


if __name__ == "__main__":
    main()
