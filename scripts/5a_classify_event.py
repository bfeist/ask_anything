#!/usr/bin/env python3
"""Stage 5a: Classify the event type of a transcript using an LLM.

Feeds the first ~3 minutes of diarized transcript plus speaker statistics
to the LLM to determine the event type.

Event types:
  - student_qa:       School downlink / ARISS contact
  - press_conference: News conference with reporters
  - panel:            Panel discussion / roundtable
  - other:            Ceremony, lecture, etc.

Output:  <transcript_stem>.classify.json
  {
    "transcript_file": "<stem>.json",
    "model": "gemma3:12b",
    "event_type": "press_conference",
    "confidence": "high",
    "classified_at": "2026-02-20T..."
  }

Usage:
  uv run python scripts/5a_classify_event.py [--transcript PATH]
      [--model MODEL] [--ollama-url URL] [--temperature FLOAT]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from astro_ia_harvest.config import (  # noqa: E402
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    TRANSCRIPTS_DIR,
    env_or_default,
)
from astro_ia_harvest.ollama_client import call_ollama, extract_json  # noqa: E402
from astro_ia_harvest.transcript_utils import (  # noqa: E402
    format_diarized_transcript,
    format_plain_transcript,
    load_transcript,
    speaker_stats,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVENT_TYPES = ("student_qa", "press_conference", "panel", "other")

CLASSIFY_SYSTEM_PROMPT = """\
You are a NASA video transcript classifier. Given speaker statistics and the \
opening portion of a transcript, determine the event type.

Classify as exactly ONE of:
- "student_qa": A school downlink / ARISS contact where students ask an astronaut \
questions from Earth. Characterized by: one dominant speaker (astronaut) with many \
brief speakers (students), satellite delay gaps, students saying "Hi my name is...".
- "press_conference": A news conference where journalists/reporters ask questions to \
astronauts, officials, or a panel. Characterized by: a moderator introducing \
reporters by name/outlet, multiple crew members giving opening statements, formal \
Q&A with named reporters.
- "panel": A panel discussion or roundtable with roughly equal speaking time among \
participants. Less structured Q&A.
- "other": Anything that doesn't fit the above (ceremony, lecture, documentary, etc).

Respond with a JSON object containing exactly two keys:
  {"event_type": "<one of the four types>", "confidence": "<high/medium/low>"}

No markdown fencing, no commentary — just the JSON object.\
"""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_event(
    segments: list[dict],
    ollama_url: str,
    model: str,
    temperature: float,
) -> tuple[str, str]:
    """Classify the event type. Returns (event_type, confidence)."""
    stats = speaker_stats(segments)

    # Use the first ~3 minutes of diarized transcript for context
    cutoff = 300.0  # seconds
    early_segs = [s for s in segments if s["start"] < cutoff]
    has_diarization = any(s.get("speaker") for s in segments)

    if has_diarization:
        opening_text = format_diarized_transcript(early_segs)
    else:
        opening_text = format_plain_transcript(early_segs)

    user_prompt = (
        f"SPEAKER STATISTICS:\n{stats}\n\n"
        f"OPENING TRANSCRIPT (first {cutoff:.0f}s):\n\n{opening_text}"
    )

    print(f"  Classifying event type ({len(user_prompt)} chars)...")
    t0 = time.time()
    raw = call_ollama(
        ollama_url, model, user_prompt,
        system=CLASSIFY_SYSTEM_PROMPT,
        temperature=temperature, num_predict=256,
    )
    elapsed = time.time() - t0
    print(f"  Classification response in {elapsed:.1f}s")

    result = extract_json(raw)
    if isinstance(result, dict):
        event_type = result.get("event_type", "other")
        confidence = result.get("confidence", "low")
        if event_type not in EVENT_TYPES:
            print(f"  WARNING: Unknown event type '{event_type}', falling back to 'other'")
            event_type = "other"
        return event_type, confidence

    print(f"  WARNING: Could not parse classification response: {raw[:200]}")
    return "other", "low"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5a: Classify event type of a transcript"
    )
    parser.add_argument(
        "--transcript", type=Path,
        help="Path to a single transcript JSON to classify.",
    )
    parser.add_argument("--model", default=env_or_default("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    parser.add_argument("--ollama-url", default=env_or_default("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--force", action="store_true", help="Re-classify even if .classify.json exists")
    parser.add_argument("--limit", type=int, default=0, help="Max transcripts to classify (0 = all)")
    args = parser.parse_args()

    print("=" * 70)
    print("Stage 5a: Classify Event Type")
    print("=" * 70)
    print(f"Model: {args.model}")

    # Build list of transcripts to process
    if args.transcript:
        candidates = [args.transcript]
    else:
        all_jsons = sorted(TRANSCRIPTS_DIR.glob("*.json"))
        candidates = [j for j in all_jsons if ".qa" not in j.name and ".classify" not in j.name]

    if not candidates:
        print("No transcript files found in", TRANSCRIPTS_DIR)
        sys.exit(1)

    # Filter already-classified unless --force
    if not args.force:
        candidates = [p for p in candidates if not p.with_suffix(".classify.json").exists()]

    if args.limit > 0:
        candidates = candidates[: args.limit]

    print(f"Transcripts to classify: {len(candidates)}")

    successes = 0
    failures = 0
    for i, transcript_path in enumerate(candidates, 1):
        print(f"\n[{i}/{len(candidates)}] {transcript_path.name}")
        try:
            data = load_transcript(transcript_path)
            segments = data["segments"]

            event_type, confidence = classify_event(
                segments, args.ollama_url, args.model, args.temperature,
            )
            print(f"  Event type: {event_type} (confidence: {confidence})")

            output = {
                "transcript_file": transcript_path.name,
                "model": args.model,
                "event_type": event_type,
                "confidence": confidence,
                "classified_at": datetime.now(timezone.utc).isoformat(),
            }
            out_path = transcript_path.with_suffix(".classify.json")
            out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  Saved to: {out_path}")
            successes += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failures += 1

    print(f"\n{'=' * 70}")
    print(f"Done: {successes} classified, {failures} failed")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
