#!/usr/bin/env python3
"""Stage 5b: Extract Q&A time boundaries from a classified transcript.

Reads the classification from <stem>.classify.json (produced by 5a) or
classifies inline if not found.  Applies an event-type-specific LLM prompt
to extract Q&A time boundaries.

The LLM returns only time boundaries — no names, no text.  Verbatim text is
reconstructed later by slicing transcript segments.

Output format (.qa.json):
  {
    "transcript_file": "<stem>.json",
    "model": "gemma3:12b",
    "event_type": "press_conference",
    "event_type_confidence": "high",
    "extracted_at": "2026-02-20T...",
    "qa_pairs": [
      {
        "question_start": 899.7,
        "question_end": 927.7,
        "answers": [
          {"answer_start": 928.8, "answer_end": 982.4}
        ]
      }, ...
    ]
  }

Usage:
  uv run python scripts/5b_extract_qa.py [--transcript PATH] [--model MODEL]
      [--ollama-url URL] [--temperature FLOAT] [--preview] [--event-type TYPE]
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
    slice_segments,
)

# ---------------------------------------------------------------------------
# Event types and prompt routing
# ---------------------------------------------------------------------------

EVENT_TYPES = ("student_qa", "press_conference", "panel", "other")

# For student Q&A, plain transcript works better (proven).
# For press conferences, diarized labels help.
USE_DIARIZATION_FOR = {"press_conference", "panel"}

# ---------------------------------------------------------------------------
# Extraction prompts  (timestamps-only — no names / affiliations)
# ---------------------------------------------------------------------------

EXTRACT_PROMPT_STUDENT_QA = """\
You are a precise transcript analyst. Your job is to identify the time boundaries of \
ALL question-and-answer pairs from this school downlink / ARISS event where students \
ask an astronaut questions from Earth.

The transcript has timestamps in seconds. The pattern repeats: a student introduces \
themselves and asks a question, then after a satellite delay (~6 seconds) the \
astronaut gives a long answer. There are typically 15-20 questions — process the \
ENTIRE transcript to the very end and do not stop early.

Your task:
1. Identify EVERY student QUESTION and the astronaut's ANSWER.
2. Skip preamble, voice checks, Mission Control chatter, and closing remarks.
3. Output items in CHRONOLOGICAL ORDER by question_start.
4. For each pair output ONLY:
   - question_start: timestamp (seconds) where student starts speaking
   - question_end: timestamp where student finishes their question
   - answers: An array with ONE object for the astronaut's answer:
     [{"answer_start": <float>, "answer_end": <float>}]

Do NOT include any text, names, or affiliations — ONLY timestamps.
IMPORTANT: Process the ENTIRE transcript. Do NOT stop early.

Respond ONLY with a JSON array of objects. No markdown fencing, no commentary.\
"""

EXTRACT_PROMPT_PRESS_CONFERENCE = """\
You are a precise transcript analyst. Your job is to identify the time boundaries of \
ALL question-and-answer exchanges in this NASA press conference transcript.

The transcript has timestamps and speaker labels (SPEAKER_XX). The typical pattern: \
a moderator introduces a reporter, the reporter asks one or more questions, then one \
or more panelists/crew answer. Multi-part questions from the same reporter count as \
ONE question block.

CRITICAL RULES:
- Do NOT include opening statements, prepared remarks, biographies, or moderator \
logistics as Q&A pairs. Only include actual QUESTIONS followed by answers.
- Opening statements typically happen before the first reporter question. Skip them.
- The Q&A portion usually begins when the moderator says something like "we'll now \
take questions" or "first question goes to...".
- Output items in CHRONOLOGICAL ORDER by question_start timestamp.

Your task:
1. Identify EVERY question and ALL subsequent answer(s).
2. Use speaker labels to help determine answer boundaries.
3. If MULTIPLE panelists answer the SAME question in sequence, include EACH as a \
separate entry in the "answers" array — do NOT create duplicate question entries.
4. For each Q&A pair output ONLY:
   - question_start: timestamp where the question starts
   - question_end: timestamp where the question ends
   - answers: An array of answer objects, one per respondent:
     [{"answer_start": <float>, "answer_end": <float>}, ...]
     The last answer's answer_end should be just before the next transition.

Do NOT include any text, names, or affiliations — ONLY timestamps.
IMPORTANT: Process the ENTIRE transcript from start to finish. Do NOT stop early.
A typical NASA press conference has 10-15 questions. If you have found fewer than 8, \
re-scan the transcript for questions you may have missed.

Respond ONLY with a JSON array of objects. No markdown fencing, no commentary.\
"""

EXTRACT_PROMPT_GENERIC = """\
You are a precise transcript analyst. Your job is to identify the time boundaries of \
ALL question-and-answer exchanges in a NASA event transcript.

The transcript has timestamps in seconds and may include speaker labels. Identify \
every instance where someone asks a question and receives a substantive answer.

Your task:
1. Identify EVERY Q&A exchange from beginning to end.
2. Skip opening remarks, prepared statements, and closing remarks.
3. Output items in CHRONOLOGICAL ORDER by question_start.
4. For each Q&A pair output ONLY:
   - question_start: timestamp where the question begins
   - question_end: timestamp where the question ends
   - answers: An array of answer objects, one per respondent:
     [{"answer_start": <float>, "answer_end": <float>}, ...]

Do NOT include any text, names, or affiliations — ONLY timestamps.
IMPORTANT: Process the ENTIRE transcript. Do NOT stop early.

Respond ONLY with a JSON array of objects. No markdown fencing, no commentary.\
"""

EXTRACT_PROMPTS = {
    "student_qa": EXTRACT_PROMPT_STUDENT_QA,
    "press_conference": EXTRACT_PROMPT_PRESS_CONFERENCE,
    "panel": EXTRACT_PROMPT_GENERIC,
    "other": EXTRACT_PROMPT_GENERIC,
}


# ---------------------------------------------------------------------------
# Extraction engine
# ---------------------------------------------------------------------------

def run_extraction(
    segments: list[dict],
    event_type: str,
    ollama_url: str,
    model: str,
    temperature: float,
) -> tuple[list[dict] | None, float, str]:
    """Run Q&A extraction, chunking long transcripts for better coverage.

    Returns (qa_pairs, elapsed_seconds, raw_response_summary).
    """
    system_prompt = EXTRACT_PROMPTS[event_type]

    has_diarization = any(s.get("speaker") for s in segments)
    use_diarized = event_type in USE_DIARIZATION_FOR and has_diarization
    label = "diarized" if use_diarized else "plain"
    format_fn = format_diarized_transcript if use_diarized else format_plain_transcript

    # Chunk long transcripts for better coverage
    CHUNK_DURATION = 600.0   # 10 minutes per chunk
    OVERLAP = 60.0           # 1 minute overlap
    total_duration = segments[-1]["end"] if segments else 0

    if total_duration <= CHUNK_DURATION * 2.5:
        # Under ~25 minutes — process in one shot
        chunks = [segments]
    else:
        chunks = []
        chunk_start = 0.0
        while chunk_start < total_duration:
            chunk_end = chunk_start + CHUNK_DURATION
            chunk_segs = [s for s in segments
                          if s["end"] >= chunk_start and s["start"] < chunk_end]
            if chunk_segs:
                chunks.append(chunk_segs)
            chunk_start += CHUNK_DURATION - OVERLAP

    all_pairs: list[dict] = []
    total_elapsed = 0.0
    raw_parts: list[str] = []

    print(f"  Processing {len(chunks)} chunk(s) via {model} ({label}, {event_type})...")

    for ci, chunk in enumerate(chunks, 1):
        formatted = format_fn(chunk)
        t_start = chunk[0]["start"]
        t_end = chunk[-1]["end"]
        user_prompt = (
            f"Here is the transcript ({label}), "
            f"covering {t_start:.0f}s to {t_end:.0f}s:\n\n{formatted}"
        )

        print(f"    Chunk {ci}/{len(chunks)}: {t_start:.0f}s-{t_end:.0f}s "
              f"({len(formatted)} chars, {len(chunk)} segs)...")
        t0 = time.time()
        raw = call_ollama(
            ollama_url, model, user_prompt,
            system=system_prompt, temperature=temperature,
        )
        elapsed = time.time() - t0
        total_elapsed += elapsed
        raw_parts.append(raw)

        result = extract_json(raw)
        if isinstance(result, list):
            print(f"    -> {len(result)} pairs in {elapsed:.1f}s")
            all_pairs.extend(result)
        else:
            print(f"    -> FAILED to parse chunk {ci} ({elapsed:.1f}s): {raw[:200]}")

    print(f"  Total extraction time: {total_elapsed:.1f}s")

    if not all_pairs:
        return None, total_elapsed, "\n---\n".join(raw_parts)

    qa_pairs = normalize_qa_pairs(all_pairs)
    return qa_pairs, total_elapsed, "\n---\n".join(raw_parts)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _strip_name_fields(pair: dict) -> None:
    """Remove any name/affiliation keys the LLM may have included."""
    for key in ("questioner_name", "questioner_affiliation", "answerer_name"):
        pair.pop(key, None)
    for ans in pair.get("answers", []):
        ans.pop("answerer_name", None)


def normalize_qa_pairs(pairs: list[dict]) -> list[dict]:
    """Normalize, merge, and sort Q&A pairs.

    1. Convert legacy flat format (answer_start, answer_end at top level)
       to the ``answers`` array format.
    2. Strip any name / affiliation fields.
    3. Merge entries that share overlapping question timestamps (within 15 s).
    4. Sort by question_start for chronological order.
    """
    # Step 1: ensure every entry has an "answers" list
    for p in pairs:
        if "answers" not in p:
            ans_entry: dict = {}
            if "answer_start" in p:
                ans_entry["answer_start"] = p.pop("answer_start")
            if "answer_end" in p:
                ans_entry["answer_end"] = p.pop("answer_end")
            p.pop("answerer_name", None)
            p["answers"] = [ans_entry] if ans_entry.get("answer_start") is not None else []

    # Step 2: strip name fields
    for p in pairs:
        _strip_name_fields(p)

    # Step 3: merge entries with overlapping/nearby question timestamps
    MERGE_THRESHOLD = 15.0  # seconds
    merged: list[dict] = []
    for p in sorted(pairs, key=lambda x: x.get("question_start", 0)):
        qs = p.get("question_start", 0)
        qe = p.get("question_end", 0)
        matched = False
        for m in merged:
            mqs = m.get("question_start", 0)
            mqe = m.get("question_end", 0)
            if abs(qs - mqs) <= MERGE_THRESHOLD or (mqs <= qs <= mqe):
                # Merge: extend question window and add answers
                m["question_start"] = min(m["question_start"], p.get("question_start", 0))
                m["question_end"] = max(m["question_end"], p.get("question_end", 0))
                # Append answers, avoiding duplicates (within 5 s)
                existing_starts = {a.get("answer_start") for a in m["answers"]}
                for a in p.get("answers", []):
                    a_start = a.get("answer_start", 0)
                    if any(abs(a_start - es) < 5.0 for es in existing_starts if es is not None):
                        continue
                    m["answers"].append(a)
                    existing_starts.add(a_start)
                matched = True
                break
        if not matched:
            merged.append(p)

    # Step 4: sort chronologically and sort each answers list by answer_start
    for m in merged:
        m["answers"] = sorted(m["answers"], key=lambda a: a.get("answer_start", 0))

    sorted_pairs = sorted(merged, key=lambda x: x.get("question_start", 0))

    # Step 5: absorb chunk-overlap artifacts — entries whose question_start
    # falls within a prior entry's answer range (and have NO identifying info).
    cleaned: list[dict] = []
    for p in sorted_pairs:
        if cleaned:
            prev = cleaned[-1]
            prev_answer_end = max(
                (a.get("answer_end", 0) for a in prev.get("answers", [])), default=0
            )
            qs = p.get("question_start", 0)
            if qs <= prev_answer_end:
                existing_starts = {a.get("answer_start") for a in prev["answers"]}
                for a in p.get("answers", []):
                    a_start = a.get("answer_start", 0)
                    if not any(abs(a_start - es) < 5.0 for es in existing_starts if es is not None):
                        prev["answers"].append(a)
                        existing_starts.add(a_start)
                prev["answers"] = sorted(
                    prev["answers"], key=lambda a: a.get("answer_start", 0)
                )
                continue
        cleaned.append(p)

    return cleaned


def compute_coverage(qa_pairs: list[dict], segments: list[dict]) -> dict:
    """Analyze how well the Q&A pairs cover the transcript timeline.

    Returns dict with qa_start, qa_end, covered_pct, and gaps > 2 min.
    """
    if not qa_pairs:
        return {"qa_start": 0, "qa_end": 0, "covered_pct": 0, "gaps": []}

    first_q = qa_pairs[0].get("question_start", 0)
    last_answers = qa_pairs[-1].get("answers", [])
    last_end = max((a.get("answer_end", 0) for a in last_answers), default=0)
    total_span = last_end - first_q if last_end > first_q else 1

    # Build covered intervals
    covered = 0.0
    prev_end = first_q
    gaps: list[dict] = []
    for p in qa_pairs:
        qs = p.get("question_start", 0)
        answers = p.get("answers", [])
        ae = max((a.get("answer_end", 0) for a in answers), default=qs)

        gap_dur = qs - prev_end
        if gap_dur > 120:  # > 2 minutes
            gaps.append({"start": prev_end, "end": qs, "duration": gap_dur})

        covered += ae - qs
        prev_end = max(prev_end, ae)

    covered_pct = covered / total_span * 100 if total_span else 0

    return {
        "qa_start": first_q,
        "qa_end": last_end,
        "covered_pct": min(covered_pct, 100),
        "gaps": gaps,
    }


# ---------------------------------------------------------------------------
# Preview helper
# ---------------------------------------------------------------------------

def print_qa_preview(qa_pairs: list[dict], segments: list[dict]) -> None:
    """Print a human-readable preview with text sliced from the transcript."""
    print(f"\n{'='*70}")
    print(f"  {len(qa_pairs)} Q&A pairs extracted")
    print(f"{'='*70}")
    for i, qa in enumerate(qa_pairs, 1):
        qs = qa.get("question_start", 0)
        qe = qa.get("question_end", 0)
        answers = qa.get("answers", [])
        q_text = slice_segments(segments, qs, qe)[:150]

        print(f"\n  Q{i}. [{qs:.1f}s-{qe:.1f}s]:")
        print(f"       {q_text}...")
        for j, ans in enumerate(answers, 1):
            a_s = ans.get("answer_start", 0)
            ae = ans.get("answer_end", 0)
            a_text = slice_segments(segments, a_s, ae)[:150]
            prefix = f"  A{j}" if len(answers) > 1 else "   A"
            print(f"    {prefix}: [{a_s:.1f}s-{ae:.1f}s]:")
            print(f"       {a_text}...")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_classification(transcript_path: Path) -> tuple[str, str] | None:
    """Try to load a pre-computed .classify.json for this transcript."""
    cls_path = transcript_path.with_suffix(".classify.json")
    if cls_path.exists():
        try:
            cls = json.loads(cls_path.read_text(encoding="utf-8"))
            return cls["event_type"], cls.get("confidence", "low")
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5b: Extract Q&A time boundaries from transcript"
    )
    parser.add_argument(
        "--transcript", type=Path,
        help="Path to transcript JSON. Defaults to first file in transcripts dir.",
    )
    parser.add_argument("--model", default=env_or_default("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    parser.add_argument("--ollama-url", default=env_or_default("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--event-type", choices=EVENT_TYPES, default=None,
        help="Skip classification and use this event type directly.",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Also print a human-readable preview with text sliced from transcript.",
    )
    args = parser.parse_args()

    # Find transcript
    if args.transcript:
        transcript_path = args.transcript
    else:
        jsons = sorted(TRANSCRIPTS_DIR.glob("*.json"))
        jsons = [j for j in jsons if ".qa" not in j.name and ".classify" not in j.name]
        if not jsons:
            print("No transcript files found in", TRANSCRIPTS_DIR)
            sys.exit(1)
        transcript_path = jsons[0]

    print(f"Transcript: {transcript_path.name}")
    print(f"Model: {args.model}")

    data = load_transcript(transcript_path)
    segments = data["segments"]

    # Resolve event type: CLI flag > .classify.json > inline classification
    if args.event_type:
        event_type = args.event_type
        confidence = "override"
        print(f"  Event type: {event_type} (user override)")
    else:
        cached = _load_classification(transcript_path)
        if cached:
            event_type, confidence = cached
            print(f"  Event type: {event_type} (from .classify.json, {confidence})")
        else:
            # Inline classification — import here to stay lightweight
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "classify_mod",
                Path(__file__).parent / "5a_classify_event.py",
            )
            classify_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(classify_mod)  # type: ignore[union-attr]
            classify_event = classify_mod.classify_event

            print("  No .classify.json found — classifying inline...")
            event_type, confidence = classify_event(
                segments, args.ollama_url, args.model, args.temperature,
            )
            print(f"  Event type: {event_type} (confidence: {confidence})")

    # Extraction
    qa_pairs, elapsed, raw = run_extraction(
        segments, event_type,
        args.ollama_url, args.model,
        temperature=args.temperature,
    )

    if qa_pairs is None:
        print("\nFAILED to parse JSON from response.")
        print("Raw response (first 1000 chars):")
        print(raw[:1000])
        sys.exit(1)

    # Summary
    print(f"\n  {len(qa_pairs)} Q&A pairs found in {elapsed:.1f}s")
    for i, qa in enumerate(qa_pairs, 1):
        answers = qa.get("answers", [])
        qs = qa.get("question_start", "?")
        qe = qa.get("question_end", "?")
        n_ans = len(answers)
        last_end = answers[-1].get("answer_end", "?") if answers else "?"
        print(f"  {i:2d}. [{qs:>7}s-{qe:>7}s]  {n_ans} answer(s), ends {last_end}s")

    # Optional preview
    if args.preview:
        print_qa_preview(qa_pairs, segments)

    # Coverage analysis
    coverage = compute_coverage(qa_pairs, segments)
    if coverage["gaps"]:
        print(f"\n  Coverage: {coverage['covered_pct']:.0f}% of Q&A region "
              f"({coverage['qa_start']:.0f}s - {coverage['qa_end']:.0f}s)")
        print(f"  WARNING: {len(coverage['gaps'])} gap(s) > 2 min may contain missed questions:")
        for gap in coverage["gaps"]:
            print(f"    {gap['start']:.0f}s - {gap['end']:.0f}s "
                  f"({gap['duration']:.0f}s / {gap['duration']/60:.1f} min)")

    # Save — timestamps-only output
    output = {
        "transcript_file": transcript_path.name,
        "model": args.model,
        "event_type": event_type,
        "event_type_confidence": confidence,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "qa_pairs": qa_pairs,
    }
    if coverage["gaps"]:
        output["coverage_gaps"] = coverage["gaps"]
    out_path = transcript_path.with_suffix(".qa.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
