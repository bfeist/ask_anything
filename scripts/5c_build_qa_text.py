#!/usr/bin/env python3
"""Stage 5c: Build full Q&A text from time-boundary outputs (5b) + transcripts.

Reads each .qa.json produced by 5b, loads the matching transcript, and
reconstructs the verbatim question and answer text by slicing transcript
segments at the stored timestamps.  Speaker attribution is added by
finding the dominant speaker (by talk-time) in each time window.

This script is a PURE RECONSTRUCTION tool — it contains no filtering or
quality-gate logic.  All content quality decisions are made in Stage 5b.
The output is intended for spot-checking and human review only; Stage 6
does not depend on this script.

Usage:
  uv run python scripts/5c_build_qa_text.py
  uv run python scripts/5c_build_qa_text.py --qa-file data/qa/<file>.qa.json
  uv run python scripts/5c_build_qa_text.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from astro_ia_harvest.config import (  # noqa: E402
    QA_DIR,
    QA_TEXT_DIR,
    TRANSCRIPTS_DIR,
    ensure_directories,
)
from astro_ia_harvest.transcript_utils import load_transcript, slice_segments  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dominant_speaker(segments: list[dict], start: float, end: float) -> str | None:
    """Return the speaker ID with the most talk-time in [start, end], or None."""
    durations: dict[str, float] = defaultdict(float)
    for seg in segments:
        sp = seg.get("speaker")
        if not sp:
            continue
        # Compute overlap between this segment and the window
        overlap_start = max(seg["start"], start)
        overlap_end = min(seg["end"], end)
        if overlap_end > overlap_start:
            durations[sp] += overlap_end - overlap_start
    if not durations:
        return None
    return max(durations, key=lambda k: durations[k])


def merge_contiguous_answers(answer_bounds: list[dict], max_gap: float = 5.0) -> list[dict]:
    """Merge consecutive answer boundary entries when the gap is small.

    Operates on raw boundary dicts (answer_start, answer_end) BEFORE text is
    assembled.  Speaker is resolved later from the merged range.
    """
    if len(answer_bounds) <= 1:
        return answer_bounds

    merged: list[dict] = [dict(answer_bounds[0])]
    for ans in answer_bounds[1:]:
        prev = merged[-1]
        gap = ans["answer_start"] - prev["answer_end"]
        if gap <= max_gap:
            prev["answer_end"] = max(prev["answer_end"], ans["answer_end"])
        else:
            merged.append(dict(ans))
    return merged


def build_text_pair(
    pair: dict,
    segments: list[dict],
) -> dict:
    """Reconstruct a single Q&A pair with text and speaker attribution.

    Answer boundaries are merged BEFORE text assembly so that each word
    appears at most once (preventing duplication at contiguous boundaries).
    """
    q_start = pair["question_start"]
    q_end = pair["question_end"]

    question = {
        "start": q_start,
        "end": q_end,
        "speaker": dominant_speaker(segments, q_start, q_end),
        "text": slice_segments(segments, q_start, q_end),
    }

    # Merge contiguous answer boundaries first, then assemble text once
    raw_answers = pair.get("answers", [])
    merged_bounds = merge_contiguous_answers(raw_answers)

    answers = []
    for ans in merged_bounds:
        a_start = ans["answer_start"]
        a_end = ans["answer_end"]
        answers.append({
            "start": a_start,
            "end": a_end,
            "speaker": dominant_speaker(segments, a_start, a_end),
            "text": slice_segments(segments, a_start, a_end),
        })

    # Drop answer chunks spoken by the questioner — they are likely the
    # questioner continuing to talk, not an actual answer.
    # Skip this filter when there is no audible question (zero-width marker).
    q_speaker = question["speaker"]
    if q_speaker and len(answers) > 1:
        filtered = [a for a in answers if a["speaker"] != q_speaker]
        # Only apply the filter if it doesn't remove ALL answers
        if filtered:
            answers = filtered

    return {"question": question, "answers": answers}


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_qa_file(qa_path: Path, *, force: bool = False) -> bool:
    """Build the qa_text JSON for a single .qa.json file.

    Returns True on success, False on failure.
    """
    out_path = QA_TEXT_DIR / (qa_path.stem.replace(".qa", "") + ".qa_text.json")
    if out_path.exists() and not force:
        print(f"  Skipping (already exists): {out_path.name}")
        return True

    # Load qa boundaries
    try:
        qa_data = json.loads(qa_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  ERROR reading {qa_path.name}: {exc}")
        return False

    # Locate matching transcript
    transcript_file = qa_data.get("transcript_file", "")
    transcript_path = TRANSCRIPTS_DIR / transcript_file
    if not transcript_path.exists():
        # Fall back: derive stem from qa filename
        stem = qa_path.stem.replace(".qa", "")
        transcript_path = TRANSCRIPTS_DIR / f"{stem}.json"

    if not transcript_path.exists():
        print(f"  ERROR: Transcript not found for {qa_path.name}")
        print(f"         Looked for: {transcript_path}")
        return False

    try:
        transcript = load_transcript(transcript_path)
    except Exception as exc:
        print(f"  ERROR loading transcript {transcript_path.name}: {exc}")
        return False

    segments = transcript["segments"]
    qa_pairs_raw = qa_data.get("qa_pairs", [])

    # Build text pairs — pure reconstruction, no filtering
    built_pairs = []
    for i, pair in enumerate(qa_pairs_raw, 1):
        try:
            text_pair = build_text_pair(pair, segments)
            text_pair["index"] = i
            # Reorder keys for readability
            built_pairs.append({
                "index": text_pair["index"],
                "question": text_pair["question"],
                "answers": text_pair["answers"],
            })
        except Exception as exc:
            print(f"  WARNING: Pair {i} failed: {exc}")

    # Re-index
    for i, p in enumerate(built_pairs, 1):
        p["index"] = i

    # Assemble output
    output = {
        "transcript_file": qa_data.get("transcript_file", transcript_path.name),
        "source_qa_file": qa_path.name,
        "event_type": qa_data.get("event_type", "unknown"),
        "event_type_confidence": qa_data.get("event_type_confidence", "unknown"),
        "extracted_at": qa_data.get("extracted_at"),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "num_pairs": len(built_pairs),
        "qa_pairs": built_pairs,
    }

    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved {len(built_pairs)} pairs -> {out_path}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5c: Reconstruct Q&A text from time-boundary files"
    )
    parser.add_argument(
        "--qa-file", type=Path, default=None,
        help="Process a single .qa.json file instead of all files in data/qa/.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing .qa_text.json files.",
    )
    args = parser.parse_args()

    ensure_directories()

    print("=" * 70)
    print("Stage 5c: Build Q&A Text")
    print("=" * 70)
    print(f"  QA source dir:  {QA_DIR}")
    print(f"  Output dir:     {QA_TEXT_DIR}")
    print(f"  Transcripts:    {TRANSCRIPTS_DIR}")

    if args.qa_file:
        if not args.qa_file.exists():
            print(f"ERROR: File not found: {args.qa_file}")
            sys.exit(1)
        candidates = [args.qa_file]
    else:
        candidates = sorted(QA_DIR.glob("*.qa.json"))

    print(f"  .qa.json files: {len(candidates)}")

    if not candidates:
        print("\nNo .qa.json files found.")
        return

    successes = 0
    failures = 0
    for i, qa_path in enumerate(candidates, 1):
        print(f"\n[{i}/{len(candidates)}] {qa_path.name}")
        ok = process_qa_file(qa_path, force=args.force)
        if ok:
            successes += 1
        else:
            failures += 1

    print(f"\n{'=' * 70}")
    print(f"Done: {successes} ok, {failures} failed")
    print(f"Output: {QA_TEXT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
