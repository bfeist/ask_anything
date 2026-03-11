#!/usr/bin/env python3
"""Utility: Fetch a per-file date for every video asset that has been QA-extracted.

For each QA file in data/qa/, resolves the real IA identifier, fetches the IA
item metadata, then matches the specific video file within that item to extract
its individual mtime (the date the file was actually uploaded/added to IA).

Many IA items (e.g. "Expedition-71-Content") accumulate videos over months, so
the item-level publicdate is useless — we need the per-file mtime.

Date resolution priority (per file):
  1. mtime of the matching video file in the IA item → YYYY-MM-DD
  2. publicdate / addeddate of the IA item → YYYY-MM-DD  (fallback)
  3. date field of the IA item → as-is                  (last resort)

Output: data/video_dates.json
  {
    "<qa_filename>": {
      "date": "YYYY-MM-DD",   # best available; empty string if nothing found
      "source": "ia_file_mtime" | "ia_publicdate" | "ia_addeddate" | "ia_date" | "none",
      "identifier": "<ia_identifier>"
    },
    ...
  }

Re-running is safe: already-resolved entries are skipped unless missing a date.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from astro_ia_harvest.config import CLASSIFIED_JSONL, QA_DIR  # noqa: E402
from astro_ia_harvest.download_utils import canonical_video_key  # noqa: E402
from astro_ia_harvest.ia_api import fetch_item_metadata  # noqa: E402

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = re.compile(
    r"(?i)(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"[ _-]?(\d{1,2})[ _,_-](\d{4})"
)
_YYMMDD_RE = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)")
_YYYYMM_RE = re.compile(r"(?<!\d)(20\d{2})[ _-](\d{2})(?!\d)")


def extract_date_from_name(name: str) -> str:
    """Best-effort date extraction from a QA or video filename.

    Tries month-name patterns first (e.g. 'June18_2018'), then 6-digit
    YYMMDD groups, then YYYY_MM year-month patterns.  Returns '' if nothing
    plausible is found.
    """
    # 1. Month-name style: June18_2018, July_19_2018
    m = _MONTH_RE.search(name)
    if m:
        mon_key = m.group(1).lower()[:3]
        mon = _MONTH_MAP.get(mon_key, 0)
        day = int(m.group(2))
        year = int(m.group(3))
        if mon and 1 <= day <= 31 and 2000 <= year <= 2040:
            return f"{year:04d}-{mon:02d}-{day:02d}"

    # 2. 6-digit YYMMDD (e.g. 231006, 250717)
    for m in _YYMMDD_RE.finditer(name):
        yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            year = 2000 + yy if yy <= 35 else 1900 + yy
            return f"{year:04d}-{mm:02d}-{dd:02d}"

    # 3. YYYY_MM style (e.g. 2018_07 → 2018-07-01)
    m = _YYYYMM_RE.search(name)
    if m:
        year, mon = int(m.group(1)), int(m.group(2))
        if 1 <= mon <= 12:
            return f"{year:04d}-{mon:02d}-01"

    return ""

OUTPUT_FILE = ROOT / "data" / "video_dates.json"

VIDEO_EXTENSIONS = (".mp4", ".mxf", ".mov", ".m4v", ".avi", ".mpg", ".mpeg", ".webm")


# ---------------------------------------------------------------------------
# Identifier resolution (same logic as before)
# ---------------------------------------------------------------------------

def build_identifier_lookups() -> tuple[set[str], dict[str, str]]:
    """Return (real_identifiers, canonical_key_to_identifier) from classified_candidates.jsonl."""
    real_idents: set[str] = set()
    ck_to_ident: dict[str, str] = {}

    if not CLASSIFIED_JSONL.exists():
        return real_idents, ck_to_ident

    with open(CLASSIFIED_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ident = str(rec.get("identifier", "")).strip()
            filename = str(rec.get("filename", "")).strip()
            if not ident:
                continue
            real_idents.add(ident)
            if filename:
                ck_to_ident[canonical_video_key(filename)] = ident

    return real_idents, ck_to_ident


def resolve_identifier(qa_prefix: str, real_idents: set[str], ck_to_ident: dict[str, str]) -> str:
    if qa_prefix in real_idents:
        return qa_prefix
    ck = canonical_video_key(qa_prefix)
    return ck_to_ident.get(ck, qa_prefix)


# ---------------------------------------------------------------------------
# QA file → (identifier, video_stem_ck) mapping
# ---------------------------------------------------------------------------

def qa_stem_ck(qa_name: str) -> str:
    """Return the canonical key for the video stem embedded in a QA filename.

    QA filenames end in ``.qa.json`` (optionally preceded by ``_lowres``).
    We strip those suffixes then apply canonical_video_key so the result
    matches whatever IA stores as the actual file name.
    """
    stem = qa_name
    if stem.endswith(".qa.json"):
        stem = stem[: -len(".qa.json")]
    return canonical_video_key(stem)


def collect_qa_entries(
    real_idents: set[str], ck_to_ident: dict[str, str]
) -> list[tuple[str, str, str]]:
    """Return list of (qa_filename, identifier, video_ck).

    video_ck is the canonical key to match against IA file names.
    """
    entries: list[tuple[str, str, str]] = []
    for p in sorted(QA_DIR.glob("*.qa.json")):
        name = p.name
        if "__" in name:
            prefix, rest = name.split("__", 1)
            ident = resolve_identifier(prefix, real_idents, ck_to_ident)
            ck = qa_stem_ck(rest)
        else:
            stem = name[: -len(".qa.json")]
            ck_stem = canonical_video_key(stem)
            ident = ck_to_ident.get(ck_stem, stem)
            ck = ck_stem
        entries.append((name, ident, ck))
    return entries


# ---------------------------------------------------------------------------
# Per-file date resolution
# ---------------------------------------------------------------------------

def build_file_ck_map(ia_files: list[dict]) -> dict[str, dict]:
    """Return {canonical_video_key(filename): file_entry} for all video files in an IA item."""
    result: dict[str, dict] = {}
    for f in ia_files:
        fn = str(f.get("name", ""))
        if not fn:
            continue
        low = fn.lower()
        if not any(low.endswith(ext) or f".{ext.lstrip('.')}." in low or low.endswith(".ia.mp4")
                   for ext in VIDEO_EXTENSIONS):
            # Quick check: must be a video-like file
            has_video_ext = any(
                low.endswith(ext) for ext in VIDEO_EXTENSIONS
            ) or low.endswith(".ia.mp4")
            if not has_video_ext:
                continue
        ck = canonical_video_key(fn)
        if ck:
            result[ck] = f
    return result


def mtime_to_date(mtime: str | int | None) -> str:
    if not mtime:
        return ""
    try:
        dt = datetime.fromtimestamp(int(mtime), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return ""


def resolve_date_for_file(
    video_ck: str, ia_files: list[dict], ia_metadata: dict
) -> tuple[str, str]:
    """Return (date_str, source) for a single QA file.

    Tries to match the specific file in the IA item first (mtime), then falls
    back to item-level metadata fields.
    """
    # 1. Match the specific video file by canonical key
    file_map = build_file_ck_map(ia_files)
    matched = file_map.get(video_ck)
    if matched:
        date = mtime_to_date(matched.get("mtime"))
        if date:
            return date, "ia_file_mtime"

    # 2. Item-level fallbacks
    m = ia_metadata.get("metadata", {})
    for field, source in (("publicdate", "ia_publicdate"), ("addeddate", "ia_addeddate")):
        val = str(m.get(field, "")).strip()
        if val and len(val) >= 10:
            return val[:10], source

    # 3. mtime of the largest video file in the item
    video_files = [
        f for f in ia_files
        if any(str(f.get("name", "")).lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    ]
    if video_files:
        largest = max(video_files, key=lambda f: int(f.get("size") or 0))
        date = mtime_to_date(largest.get("mtime"))
        if date:
            return date, "ia_file_mtime_largest"

    # 4. date field
    val = str(m.get("date", "")).strip()
    if val:
        return val, "ia_date"

    return "", "none"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("fetch_video_dates.py — Build per-file data/video_dates.json")
    print("=" * 70)

    real_idents, ck_to_ident = build_identifier_lookups()
    print(f"Real IA identifiers (from classified): {len(real_idents)}")

    entries = collect_qa_entries(real_idents, ck_to_ident)
    print(f"QA files found: {len(entries)}")
    print()

    # Load existing output
    output: dict[str, dict] = {}
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, encoding="utf-8") as f:
                output = json.load(f)
        except (json.JSONDecodeError, OSError):
            output = {}

    # Determine which QA files still need dates
    needs_work = [
        (qa_name, ident, ck)
        for qa_name, ident, ck in entries
        if qa_name not in output or not output[qa_name].get("date")
    ]
    print(f"Already have dates (skipping): {len(entries) - len(needs_work)}")
    print(f"Need to resolve: {len(needs_work)}")
    print()

    if not needs_work:
        print("All dates already populated.")
        return

    # Group by identifier so we fetch each IA item's metadata only once
    by_ident: dict[str, list[tuple[str, str]]] = {}
    for qa_name, ident, ck in needs_work:
        by_ident.setdefault(ident, []).append((qa_name, ck))

    total_idents = len(by_ident)
    for idx, (ident, file_list) in enumerate(sorted(by_ident.items()), start=1):
        print(f"[{idx}/{total_idents}] {ident} ({len(file_list)} file(s)) ...", flush=True)
        metadata = fetch_item_metadata(ident)
        if metadata is None or (not metadata.get("files") and not metadata.get("metadata")):
            # IA returned nothing useful — fall back to filename date parsing
            print(f"  WARNING: no IA data for {ident!r}, trying filename fallback")
            for qa_name, ck in file_list:
                date = extract_date_from_name(qa_name)
                source = "filename_date" if date else "none"
                output[qa_name] = {"date": date, "source": source, "identifier": ident}
                print(f"  {qa_name}")
                print(f"    -> {date or '(no date)'} [{source}]")
        else:
            ia_files = metadata.get("files", [])
            for qa_name, ck in file_list:
                date, source = resolve_date_for_file(ck, ia_files, metadata)
                if not date:
                    # Last resort: extract date from the QA filename itself
                    date = extract_date_from_name(qa_name)
                    source = "filename_date" if date else "none"
                output[qa_name] = {"date": date, "source": source, "identifier": ident}
                print(f"  {qa_name}")
                print(f"    -> {date or '(no date)'} [{source}]")

        # Save after each identifier so partial runs preserve progress
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        time.sleep(0.3)

    has_date = sum(1 for v in output.values() if v.get("date"))
    missing = sum(1 for v in output.values() if not v.get("date"))
    print()
    print(f"Written: {OUTPUT_FILE}")
    print(f"Files with a date : {has_date} / {len(output)}")
    print(f"Files with no date: {missing}")
    if missing:
        print("\nFiles still missing dates:")
        for qa_name, rec in sorted(output.items()):
            if not rec.get("date"):
                print(f"  {qa_name}  [{rec.get('identifier')}]")


if __name__ == "__main__":
    main()
