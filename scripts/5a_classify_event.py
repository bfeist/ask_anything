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
    CLASSIFY_DIR,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    TRANSCRIPTS_DIR,
    ensure_directories,
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

EVENT_TYPES = (
    "student_qa",
    "press_conference",
    "media_interview",
    "panel",
    "produced_content",
    "produced_interview",
    "other",
)

CLASSIFY_SYSTEM_PROMPT = """\
You are a NASA video transcript classifier. Given a video filename, speaker \
statistics, and the opening portion of a transcript, determine the event type.

IMPORTANT: The VIDEO FILENAME is your strongest signal. Keywords in the filename \
should be treated as near-definitive evidence of the event type:
- "News_Conference", "News Conference", "Press_Conference", "Postflight", \
"Post-Flight", "Post_Flight", "Mission_Overview_Briefing", \
"Flight_Readiness" → press_conference
- "Education", "Inflight_with", "EDU", "ARISS", "School", "Student", \
"Answers_Questions_From_Students" → student_qa
- "Inflight_KTTV", "Inflight_KDVR", "Inflight_[TV station]", \
"Discusses_Life_In_Space_With", "Talks_with", "Answers_Media_Questions", \
"Interviews_with" → media_interview
- "Interview_Reel", "CCP_Interview", "Quick_Questions", \
"discusses_his_background", "discusses_launch_day", "discusses_the", \
"discusses_training" → produced_interview
- "Explainer", "What_Human_Health", "Connecting_Classrooms_to_Space" (without \
live Q&A), montage, overview with no Q&A portion, PSA → produced_content

Classify as exactly ONE of:
- "student_qa": A school downlink / ARISS contact where students ask an astronaut \
questions from Earth. Characterized by: one dominant speaker (astronaut) with many \
brief speakers (students), satellite delay gaps, students saying "Hi my name is...". \
Note: some student Q&A events are between an astronaut and a museum or educational \
organization — still classify as student_qa.
- "press_conference": A news conference where journalists/reporters ask questions to \
astronauts, officials, or a panel. Characterized by: a moderator introducing \
reporters by name/outlet, multiple crew members giving opening statements, formal \
Q&A with named reporters. Also includes post-flight news conferences, mission \
overview briefings with Q&A, and pre-launch press events.
- "media_interview": A TV station, radio, or newspaper interview with an astronaut. \
Typically 1-2 interviewers from a single media outlet talking with 1-2 astronauts. \
Characterized by: informal conversational tone, TV anchor/host asking questions, \
astronaut answering. Often called "inflight" events (e.g. "Inflight KTTV-TV" or \
"Discusses Life in Space with [outlet]"). Usually 2-4 speakers total.
- "panel": A panel discussion or roundtable with roughly equal speaking time among \
participants. Less structured Q&A with multiple expert speakers.
- "produced_interview": A pre-produced interview where the astronaut answers \
off-camera questions but the INTERVIEWER'S AUDIO HAS BEEN EDITED OUT. Only the \
astronaut is heard. Characterized by: exactly 1 speaker for the entire video, \
clear topic transitions (the astronaut shifts subjects as if responding to a new \
question), often 5-20 minutes long. Common filename patterns: "Interview_Reel", \
"CCP_Interview", "discusses_his_background", "Quick_Questions". The astronaut \
is clearly answering questions but you cannot hear the interviewer.
- "produced_content": A pre-recorded, edited, or narrated video with NO Q&A at \
all — not even implied questions. Examples: mission overview montages, explainer \
videos, PSAs, narrated documentaries, ham radio knowledge episodes. Characterized \
by: scripted monologue, no topic transitions that suggest questions, promotional \
or educational narration style.
- "other": Anything that doesn't fit the above (ceremony, lecture, etc).

IMPORTANT RULES:
1. If a video has ONLY 1 speaker and the filename contains "Interview_Reel", \
"CCP_Interview", "Quick_Questions", or "discusses", classify as "produced_interview". \
These are interviews with the questioner's audio edited out.
2. If a video has ONLY 1-2 speakers and one speaker dominates >90% of talk time \
but the transcript shows clear topic transitions (the speaker shifts subjects as if \
responding to new questions), classify as "produced_interview" — NOT "produced_content".
3. Only classify as "produced_content" if there is NO evidence of Q&A pattern at \
all — the content is purely scripted narration, a montage, PSA, or explainer.
4. If the transcript shows a conversational back-and-forth between a TV host and \
an astronaut, classify as "media_interview" even if speaker count is >=3 (audio \
bleed from broadcast can create extra speaker labels).
5. A video named "News Conference" or "Post-flight" is very likely "press_conference".
6. When uncertain between "other" and a more specific type, prefer the specific type \
if there is any Q&A pattern visible.
7. Very short videos (under 3 minutes) with only sound bites and no Q&A exchange \
pattern are almost always "produced_content" — they are promotional montages or \
highlight reels.

Respond with a JSON object containing exactly two keys:
  {"event_type": "<one of the six types>", "confidence": "<high/medium/low>"}

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
    transcript_filename: str = "",
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

    # Build prompt with filename hint and duration info when available
    total_duration = segments[-1]["end"] if segments else 0
    parts = []
    if transcript_filename:
        # Clean up the filename for display — remove extensions, underscores, etc.
        clean_name = transcript_filename.replace("_lowres.json", "").replace("_", " ")
        parts.append(f"VIDEO FILENAME: {clean_name}")
    parts.append(f"VIDEO DURATION: {total_duration:.0f} seconds ({total_duration/60:.1f} minutes)")
    parts.append(f"SPEAKER STATISTICS:\n{stats}")
    parts.append(f"OPENING TRANSCRIPT (first {cutoff:.0f}s):\n\n{opening_text}")
    user_prompt = "\n\n".join(parts)

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

    ensure_directories()

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
        candidates = [p for p in candidates if not (CLASSIFY_DIR / (p.stem + ".classify.json")).exists()]

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
                transcript_filename=transcript_path.name,
            )
            print(f"  Event type: {event_type} (confidence: {confidence})")

            output = {
                "transcript_file": transcript_path.name,
                "model": args.model,
                "event_type": event_type,
                "confidence": confidence,
                "classified_at": datetime.now(timezone.utc).isoformat(),
            }
            out_path = CLASSIFY_DIR / (transcript_path.stem + ".classify.json")
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
