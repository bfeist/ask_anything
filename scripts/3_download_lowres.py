#!/usr/bin/env python3
"""Step 3: Download low-res IA variants for relevant records only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

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


def load_existing_video_keys() -> set[str]:
    keys: set[str] = set()

    for p in DOWNLOAD_DIR.rglob("*.mp4"):
        keys.add(canonical_video_key(p.name))

    if EXISTING_DOWNLOAD_DIR.exists():
        for p in EXISTING_DOWNLOAD_DIR.rglob("*.mp4"):
            keys.add(canonical_video_key(p.name))

    return keys


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


def try_download(url: str, out_path: Path) -> tuple[bool, str]:
    try:
        with requests.get(url, stream=True, timeout=90) as resp:
            if resp.status_code != 200:
                return False, f"http_{resp.status_code}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        return True, "ok"
    except requests.RequestException as exc:
        return False, f"network_error:{exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download low-res IA video files")
    parser.add_argument("--limit", type=int, default=0, help="Max relevant records to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    ensure_directories()

    rows = load_jsonl(CLASSIFIED_JSONL)
    targets = [r for r in rows if bool(r.get("likely_relevant"))]
    if args.limit > 0:
        targets = targets[: args.limit]

    existing_keys = load_existing_video_keys()

    print("=" * 70)
    print("STEP 3: Download Low-Res Videos")
    print("=" * 70)
    print(f"Relevant candidates: {len(targets)}")
    print(f"Existing normalized video keys: {len(existing_keys)}")

    downloaded = 0
    skipped_existing = 0
    failed = 0

    for idx, rec in enumerate(targets, start=1):
        ident = str(rec.get("identifier", ""))
        filename = str(rec.get("filename", ""))
        if not ident or not filename:
            continue

        source_key = canonical_video_key(filename)
        if source_key in existing_keys:
            skipped_existing += 1
            append_csv_row(filename, ident, "skip_existing", "matched_existing_key")
            continue

        output_name = build_output_name(ident, filename)
        output_path = DOWNLOAD_DIR / output_name
        if output_path.exists():
            skipped_existing += 1
            append_csv_row(output_name, ident, "skip_existing", "already_in_download_dir")
            continue

        urls = build_candidate_urls(ident, filename)
        if args.dry_run:
            skipped_existing += 0
            append_csv_row(output_name, ident, "dry_run", urls[0][1] if urls else "no_candidate_url")
            if idx == 1 or idx % 25 == 0:
                print(
                    f"  [{idx}/{len(targets)}] downloaded={downloaded} "
                    f"skip_existing={skipped_existing} failed={failed}"
                )
            continue

        ok = False
        last_detail = "no_candidate_url"
        for url, label in urls:
            ok, detail = try_download(url, output_path)
            if ok:
                downloaded += 1
                existing_keys.add(source_key)
                append_csv_row(output_name, ident, "success", label)
                break
            last_detail = f"{label}:{detail}"

        if not ok:
            failed += 1
            append_csv_row(output_name, ident, "failure", last_detail)
            append_failure(rec, last_detail)
            if output_path.exists():
                output_path.unlink(missing_ok=True)

        if idx == 1 or idx % 25 == 0:
            print(
                f"  [{idx}/{len(targets)}] downloaded={downloaded} "
                f"skip_existing={skipped_existing} failed={failed}"
            )

    print("\nDone")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Failed: {failed}")
    print(f"Output dir: {DOWNLOAD_DIR}")


if __name__ == "__main__":
    main()
