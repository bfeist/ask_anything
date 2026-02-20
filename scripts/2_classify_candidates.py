#!/usr/bin/env python3
"""Step 2: Classify IA video filenames for likely interview/Q&A relevance."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from astro_ia_harvest.config import (  # noqa: E402
    CLASSIFIED_JSONL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    IA_METADATA_JSONL,
    env_or_default,
    ensure_directories,
)
from astro_ia_harvest.jsonl_utils import append_jsonl, load_jsonl  # noqa: E402
from astro_ia_harvest.ollama_client import classify_filename_with_ollama  # noqa: E402


def existing_keys() -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for rec in load_jsonl(CLASSIFIED_JSONL):
        ident = str(rec.get("identifier", "")).strip()
        name = str(rec.get("filename", "")).strip()
        if ident and name:
            keys.add((ident, name))
    return keys


def classify_record(
    record: dict,
    ollama_url: str,
    ollama_model: str,
) -> dict:
    ident = str(record.get("identifier", ""))
    filename = str(record.get("filename", ""))
    title = str(record.get("title", ""))
    description = str(record.get("description", ""))
    subject = str(record.get("subject", ""))

    try:
        decision, confidence, reason = classify_filename_with_ollama(
            ollama_url=ollama_url,
            model=ollama_model,
            filename=filename,
            title=title,
            description=description,
            subject=subject,
        )
        method = "ollama"
    except Exception as exc:
        decision = "reject"
        confidence = 0.0
        reason = f"ollama_failed:{exc}"
        method = "ollama_fallback"

    likely_relevant = decision == "keep"
    return {
        "identifier": ident,
        "filename": filename,
        "title": title,
        "likely_relevant": likely_relevant,
        "decision": decision,
        "method": method,
        "model": ollama_model,
        "confidence": confidence,
        "reason": reason,
        "classified_at": int(time.time()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify IA candidate video files")
    parser.add_argument("--ollama-url", default=env_or_default("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument("--model", default=env_or_default("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    ensure_directories()
    records = load_jsonl(IA_METADATA_JSONL)
    done = existing_keys()

    print("=" * 70)
    print("STEP 2: Classify Interview/Q&A Relevance")
    print("=" * 70)
    print(f"Input records: {len(records)}")
    print(f"Already classified: {len(done)}")

    remaining = [r for r in records if (r.get("identifier"), r.get("filename")) not in done]
    if args.limit > 0:
        remaining = remaining[: args.limit]

    print(f"To classify now: {len(remaining)}")
    print(f"Model: {args.model}")

    kept = 0
    rejected = 0
    for idx, rec in enumerate(remaining, start=1):
        out = classify_record(
            rec,
            ollama_url=args.ollama_url,
            ollama_model=args.model,
        )
        append_jsonl(CLASSIFIED_JSONL, out)
        if out["likely_relevant"]:
            kept += 1
        else:
            rejected += 1

        if idx == 1 or idx % 100 == 0:
            print(f"  [{idx}/{len(remaining)}] kept={kept} rejected={rejected}")

    print("\nDone")
    print(f"Kept: {kept}")
    print(f"Rejected: {rejected}")
    print(f"Output file: {CLASSIFIED_JSONL}")


if __name__ == "__main__":
    main()
