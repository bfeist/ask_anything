#!/usr/bin/env python3
"""Step 1: Scan Internet Archive and collect candidate video file metadata."""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from astro_ia_harvest.config import (  # noqa: E402
    IA_METADATA_JSONL,
    IA_PROGRESS_FILE,
    IA_ROWS_PER_REQUEST,
    IA_UPLOADERS,
    ensure_directories,
)
from astro_ia_harvest.ia_api import build_records, fetch_item_metadata, search_identifiers, search_updated_identifiers  # noqa: E402
from astro_ia_harvest.jsonl_utils import append_jsonl, remove_identifiers_from_jsonl  # noqa: E402


def load_seen_identifiers() -> set[str]:
    if not IA_PROGRESS_FILE.exists():
        return set()
    return {
        line.strip()
        for line in IA_PROGRESS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def append_seen(identifier: str) -> None:
    with open(IA_PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{identifier}\n")


def rescan_stale_identifiers(
    stale: set[str],
    seen: set[str],
) -> tuple[int, int]:
    """Directly re-fetch metadata for identifiers known to have changed.

    Returns (identifiers_rescanned, records_written).
    """
    total_rescanned = 0
    total_written = 0
    for ident in sorted(stale):
        print(f"  Re-scanning {ident} ...", end=" ", flush=True)
        metadata = fetch_item_metadata(ident)
        if not metadata:
            print("no metadata, skipping")
            append_seen(ident)
            seen.add(ident)
            continue
        records = build_records(ident, metadata)
        for rec in records:
            append_jsonl(IA_METADATA_JSONL, rec)
            total_written += 1
        print(f"{len(records)} record(s)")
        append_seen(ident)
        seen.add(ident)
        total_rescanned += 1
        time.sleep(0.3)
    return total_rescanned, total_written


def run_query(
    query: str,
    seen: set[str],
    max_pages: int | None,
    max_items: int | None,
) -> tuple[int, int]:
    ids, total = search_identifiers(query, page=1)
    pages = max(1, math.ceil(total / IA_ROWS_PER_REQUEST))
    if max_pages is not None:
        pages = min(pages, max_pages)
    print(f"  Found {total} total item(s), {pages} page(s) to scan")

    new_identifiers = 0
    written_records = 0

    for page in range(1, pages + 1):
        if page == 1:
            page_ids = ids
        else:
            print(f"  Fetching page {page}/{pages} ...")
            page_ids, _ = search_identifiers(query, page=page)

        already_seen_on_page = 0
        for identifier in page_ids:
            if max_items is not None and new_identifiers >= max_items:
                return new_identifiers, written_records

            if identifier in seen:
                already_seen_on_page += 1
                continue

            print(f"  [{new_identifiers + 1}] {identifier} ...", end=" ", flush=True)
            metadata = fetch_item_metadata(identifier)
            if not metadata:
                print("no metadata, skipping")
                append_seen(identifier)
                seen.add(identifier)
                continue

            records = build_records(identifier, metadata)
            for rec in records:
                append_jsonl(IA_METADATA_JSONL, rec)
                written_records += 1

            print(f"{len(records)} record(s) written")
            append_seen(identifier)
            seen.add(identifier)
            new_identifiers += 1
            time.sleep(0.3)

        if page_ids and already_seen_on_page == len(page_ids):
            break

    return new_identifiers, written_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan IA metadata for candidate videos")
    parser.add_argument("--max-pages", type=int, default=0, help="Max pages per query (0 = all)")
    parser.add_argument("--max-queries", type=int, default=0, help="Max queries to run (0 = all)")
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Max new identifiers per query for short tests (0 = all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore seen-identifiers cache and reprocess all items",
    )
    parser.add_argument(
        "--rescan",
        nargs="+",
        metavar="IDENTIFIER",
        help="Force rescan of specific IA identifier(s), replacing their existing records",
    )
    args = parser.parse_args()

    ensure_directories()
    seen = load_seen_identifiers()

    if args.rescan:
        print("=" * 70)
        print("STEP 1: Rescan specific identifiers")
        print("=" * 70)
        targets = set(args.rescan)
        removed = remove_identifiers_from_jsonl(IA_METADATA_JSONL, targets)
        print(f"Removed {removed} existing record(s) from metadata")
        lines = IA_PROGRESS_FILE.read_text(encoding="utf-8").splitlines() if IA_PROGRESS_FILE.exists() else []
        IA_PROGRESS_FILE.write_text(
            "\n".join(l for l in lines if l.strip() not in targets) + "\n",
            encoding="utf-8",
        )
        for ident in targets:
            seen.discard(ident)
        n_rescanned, n_written = rescan_stale_identifiers(targets, seen)
        print(f"\nRescanned {n_rescanned} identifier(s), wrote {n_written} record(s)")
        return

    if args.force:
        seen = set()
        if IA_METADATA_JSONL.exists():
            IA_METADATA_JSONL.unlink()
        if IA_PROGRESS_FILE.exists():
            IA_PROGRESS_FILE.unlink()

    queries = [f'uploader:"{u}"' for u in IA_UPLOADERS]
    if args.max_queries > 0:
        queries = queries[: args.max_queries]

    print("=" * 70)
    print("STEP 1: Scan IA Metadata")
    print("=" * 70)
    print(f"Seen identifiers: {len(seen)}")

    total_new_ids = 0
    total_written = 0

    # Automatically rescan identifiers that were updated on IA since the last run.
    if not args.force and seen and IA_PROGRESS_FILE.exists():
        last_run_ts = IA_PROGRESS_FILE.stat().st_mtime
        since_date = datetime.fromtimestamp(last_run_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"\nLooking for items updated since last run ({since_date}) ...")
        stale: set[str] = set()
        for query in queries:
            updated = search_updated_identifiers(query, since_date)
            newly_stale = updated & seen
            stale |= newly_stale
            if newly_stale:
                print(f"  {query}: {len(newly_stale)} stale identifier(s): {', '.join(sorted(newly_stale))}")
        if stale:
            removed = remove_identifiers_from_jsonl(IA_METADATA_JSONL, stale)
            for ident in stale:
                seen.discard(ident)
            lines = IA_PROGRESS_FILE.read_text(encoding="utf-8").splitlines()
            IA_PROGRESS_FILE.write_text(
                "\n".join(l for l in lines if l.strip() not in stale) + "\n",
                encoding="utf-8",
            )
            print(f"  Removed {removed} stale record(s) from metadata; re-scanning {len(stale)} identifier(s) directly.")
            n_rescanned, n_written = rescan_stale_identifiers(stale, seen)
            total_new_ids += n_rescanned
            total_written += n_written
        else:
            print("  No updated items found since last run.")

    for query in queries:
        print(f"\nQuery: {query}")
        try:
            n_ids, n_written = run_query(
                query,
                seen,
                max_pages=(args.max_pages if args.max_pages > 0 else None),
                max_items=(args.max_items if args.max_items > 0 else None),
            )
        except Exception as exc:
            print(f"  Error in query: {exc}")
            continue

        total_new_ids += n_ids
        total_written += n_written
        print(f"  New identifiers: {n_ids}")
        print(f"  Video records written: {n_written}")

    print("\nDone")
    print(f"Total new identifiers: {total_new_ids}")
    print(f"Total records written: {total_written}")
    print(f"Metadata file: {IA_METADATA_JSONL}")


if __name__ == "__main__":
    main()
