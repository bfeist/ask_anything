#!/usr/bin/env python3
"""Stage 5: Extract Q&A pair time boundaries from a transcript using an LLM.

Two-phase approach:
  1. CLASSIFY — Feed the first ~3 minutes of diarized transcript plus speaker
     statistics to the LLM to determine the event type (student_qa,
     press_conference, panel, other).
  2. EXTRACT — Use an event-type-specific prompt to identify Q&A boundaries.
     - student_qa:       plain transcript (diarization hurts here)
     - press_conference: diarized transcript (speaker labels help ID answerers)

The LLM returns only time boundaries and names. Verbatim text is reconstructed
later by slicing transcript segments — no text hallucination possible.

Output format (.qa.json):
  {
    "transcript_file": "<stem>.json",
    "model": "gemma3:12b",
    "event_type": "press_conference",
    "extracted_at": "2026-02-20T...",
    "qa_pairs": [
      {
        "questioner_name": "Marcia Dunn",
        "questioner_affiliation": "Associated Press",
        "question_start": 899.7,
        "question_end": 927.7,
        "answers": [
          {
            "answerer_name": "Jessica Meir",
            "answer_start": 928.8,
            "answer_end": 982.4
          }
        ]
      }, ...
    ]
  }

Usage:
  uv run python scripts/5_extract_qa_prototype.py [--transcript PATH] [--model MODEL]
      [--ollama-url URL] [--temperature FLOAT] [--preview] [--event-type TYPE]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from astro_ia_harvest.config import (  # noqa: E402
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    TRANSCRIPTS_DIR,
    env_or_default,
)

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

EVENT_TYPES = ("student_qa", "press_conference", "panel", "other")

# ---------------------------------------------------------------------------
# Transcript formatting helpers
# ---------------------------------------------------------------------------

def load_transcript(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_plain_transcript(segments: list[dict]) -> str:
    """Timestamped text only — no speaker labels."""
    lines = []
    for seg in segments:
        ts = f"[{seg['start']:.1f}s]"
        lines.append(f"{ts} {seg['text'].strip()}")
    return "\n".join(lines)


def format_diarized_transcript(segments: list[dict]) -> str:
    """Timestamped text with speaker labels."""
    lines = []
    for seg in segments:
        ts = f"[{seg['start']:.1f}s]"
        sp = seg.get("speaker", "?")
        lines.append(f"{ts} {sp}: {seg['text'].strip()}")
    return "\n".join(lines)


def speaker_stats(segments: list[dict]) -> str:
    """Compute speaker statistics for classification context."""
    counts: Counter[str] = Counter()
    durations: dict[str, float] = {}
    for seg in segments:
        sp = seg.get("speaker")
        if not sp:
            continue
        counts[sp] += 1
        durations[sp] = durations.get(sp, 0.0) + (seg["end"] - seg["start"])

    if not counts:
        return "No diarization data available."

    total_dur = sum(durations.values())
    lines = [f"Total speakers: {len(counts)}"]
    for sp, cnt in counts.most_common():
        dur = durations[sp]
        pct = dur / total_dur * 100 if total_dur else 0
        lines.append(f"  {sp}: {cnt} segments, {dur:.0f}s ({pct:.0f}% of talk time)")
    return "\n".join(lines)


def slice_segments(segments: list[dict], start: float, end: float) -> str:
    """Return concatenated text from all segments overlapping [start, end]."""
    parts = []
    for seg in segments:
        if seg["end"] >= start and seg["start"] <= end:
            parts.append(seg["text"].strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Ollama interaction
# ---------------------------------------------------------------------------

def call_ollama(
    ollama_url: str,
    model: str,
    system: str,
    user_prompt: str,
    temperature: float = 0.1,
    num_ctx: int = 32768,
    num_predict: int = 8192,
    timeout: int = 600,
) -> str:
    """Call Ollama /api/generate and return the response text."""
    resp = requests.post(
        ollama_url,
        json={
            "model": model,
            "system": system,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "temperature": temperature,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def extract_json(text: str) -> dict | list | None:
    """Robustly extract a JSON object or array from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding a JSON array
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try finding a JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Phase 1: Event classification
# ---------------------------------------------------------------------------

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
        ollama_url, model, CLASSIFY_SYSTEM_PROMPT, user_prompt,
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
# Phase 2: Q&A extraction — event-type-specific prompts
# ---------------------------------------------------------------------------

EXTRACT_PROMPT_STUDENT_QA = """\
You are a precise transcript analyst. Your job is to identify the time boundaries of \
ALL question-and-answer pairs from this school downlink / ARISS event where students \
ask an astronaut questions from Earth.

The transcript has timestamps in seconds. The pattern repeats: a student says \
"Hi, my name is ___" and asks a question, then after a satellite delay (~6 seconds) \
the astronaut gives a long answer. There are typically 15-20 questions — process the \
ENTIRE transcript to the very end and do not stop early.

Your task:
1. Identify EVERY student QUESTION and the astronaut's ANSWER.
2. Students usually introduce themselves: "Hi, my name is X. My question is..."
3. Skip preamble, voice checks, Mission Control chatter, and closing remarks.
4. Output items in CHRONOLOGICAL ORDER by question_start.
5. For each pair output ONLY:
   - questioner_name: Student's name if stated, else null
   - questioner_affiliation: null (not applicable for students)
   - question_start: timestamp (seconds) where student starts speaking
   - question_end: timestamp where student finishes their question
   - answers: An array with ONE object for the astronaut's answer:
     [{"answerer_name": "...", "answer_start": <float>, "answer_end": <float>}]

Do NOT include any question text or answer text — only timestamps and names.
IMPORTANT: Process the ENTIRE transcript. Do NOT stop early.

Respond ONLY with a JSON array of objects. No markdown fencing, no commentary.\
"""

EXTRACT_PROMPT_PRESS_CONFERENCE = """\
You are a precise transcript analyst. Your job is to identify the time boundaries of \
ALL question-and-answer exchanges in this NASA press conference transcript.

The transcript has timestamps and speaker labels (SPEAKER_XX). The typical pattern: \
a moderator introduces a reporter by name and outlet, the reporter asks one or more \
questions, then one or more panelists/crew answer. Multi-part questions from the same \
reporter count as ONE question block.

CRITICAL RULES:
- Do NOT include opening statements, prepared remarks, biographies, or moderator \
logistics as Q&A pairs. Only include actual reporter/audience QUESTIONS followed by \
answers.
- Opening statements typically happen before the first reporter question. Skip them.
- The Q&A portion usually begins when the moderator says something like "we'll now \
take questions" or "first question goes to...".
- Output items in CHRONOLOGICAL ORDER by question_start timestamp.

Your task:
1. Identify EVERY reporter question and ALL subsequent answer(s).
2. When the moderator reads a social media question, the moderator is the questioner \
and their affiliation is "social media".
3. Use speaker labels to help determine who is answering.
4. If MULTIPLE panelists answer the SAME question in sequence, include EACH answerer \
as a separate entry in the "answers" array — do NOT create duplicate question entries.
5. For each Q&A pair output ONLY:
   - questioner_name: Reporter's name or null
   - questioner_affiliation: News outlet, "social media", or null
   - question_start: timestamp where reporter/questioner starts speaking
   - question_end: timestamp where they finish asking
   - answers: An array of answer objects, one per respondent:
     [{"answerer_name": "<real name, not speaker label>", "answer_start": <float>, \
"answer_end": <float>}, ...]
     The last answer's answer_end should be the point just before the next moderator \
transition or next question.

Do NOT include any question text or answer text — only timestamps and names.
IMPORTANT: Process the ENTIRE transcript from start to finish. Do NOT stop early.
A typical NASA press conference has 10-15 questions. If you have found fewer than 8, \
re-scan the transcript for questions you may have missed. Pay special attention to \
social media questions read by the moderator, which often appear between reporter \
questions.

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
   - questioner_name: The questioner's name if stated, else null
   - questioner_affiliation: Their organization if mentioned, else null
   - question_start: timestamp where the question begins
   - question_end: timestamp where the question ends
   - answers: An array of answer objects, one per respondent:
     [{"answerer_name": "...", "answer_start": <float>, "answer_end": <float>}, ...]

Do NOT include any question text or answer text — only timestamps and names.
IMPORTANT: Process the ENTIRE transcript. Do NOT stop early.

Respond ONLY with a JSON array of objects. No markdown fencing, no commentary.\
"""

EXTRACT_PROMPTS = {
    "student_qa": EXTRACT_PROMPT_STUDENT_QA,
    "press_conference": EXTRACT_PROMPT_PRESS_CONFERENCE,
    "panel": EXTRACT_PROMPT_GENERIC,
    "other": EXTRACT_PROMPT_GENERIC,
}

# For student Q&A, plain transcript works better (proven).
# For press conferences, diarized labels help identify answerers.
USE_DIARIZATION_FOR = {"press_conference", "panel"}


# ---------------------------------------------------------------------------
# Main pipeline
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
            ollama_url, model, system_prompt, user_prompt, temperature=temperature
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


def normalize_qa_pairs(pairs: list[dict]) -> list[dict]:
    """Normalize, merge, and sort Q&A pairs.

    1. Convert legacy flat format (answerer_name, answer_start, answer_end)
       to the ``answers`` array format.
    2. Merge entries that share overlapping question timestamps (within 15s).
    3. Sort by question_start for chronological order.
    """
    # Step 1: ensure every entry has an "answers" list
    for p in pairs:
        if "answers" not in p:
            ans_entry: dict = {}
            if p.get("answerer_name") is not None:
                ans_entry["answerer_name"] = p.pop("answerer_name")
            else:
                p.pop("answerer_name", None)
                ans_entry["answerer_name"] = None
            if "answer_start" in p:
                ans_entry["answer_start"] = p.pop("answer_start")
            if "answer_end" in p:
                ans_entry["answer_end"] = p.pop("answer_end")
            p["answers"] = [ans_entry] if ans_entry.get("answer_start") is not None else []

    # Step 2: merge entries with overlapping/nearby question timestamps
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
                # Prefer non-null questioner name
                if not m.get("questioner_name") and p.get("questioner_name"):
                    m["questioner_name"] = p["questioner_name"]
                if not m.get("questioner_affiliation") and p.get("questioner_affiliation"):
                    m["questioner_affiliation"] = p["questioner_affiliation"]
                # Append answers, avoiding duplicates (exact start or within 5s)
                existing_starts = {a.get("answer_start") for a in m["answers"]}
                for a in p.get("answers", []):
                    a_start = a.get("answer_start", 0)
                    # Skip if an answer with a very similar start time already exists
                    if any(abs(a_start - es) < 5.0 for es in existing_starts if es is not None):
                        continue
                    m["answers"].append(a)
                    existing_starts.add(a_start)
                matched = True
                break
        if not matched:
            merged.append(p)

    # Step 3: sort chronologically and sort each answers list by answer_start
    for m in merged:
        m["answers"] = sorted(m["answers"], key=lambda a: a.get("answer_start", 0))

    sorted_pairs = sorted(merged, key=lambda x: x.get("question_start", 0))

    # Step 4: absorb chunk-overlap artifacts — entries with no questioner name
    # whose question_start falls within a prior entry's answer range.
    cleaned: list[dict] = []
    for p in sorted_pairs:
        if (cleaned
                and not p.get("questioner_name")
                and not p.get("questioner_affiliation")):
            prev = cleaned[-1]
            prev_answer_end = max(
                (a.get("answer_end", 0) for a in prev.get("answers", [])), default=0
            )
            qs = p.get("question_start", 0)
            if qs <= prev_answer_end:
                # Absorb answers into the previous entry
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
    gaps = []
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


def print_qa_preview(qa_pairs: list[dict], segments: list[dict]) -> None:
    """Print a human-readable preview with text sliced from the transcript."""
    print(f"\n{'='*70}")
    print(f"  {len(qa_pairs)} Q&A pairs extracted")
    print(f"{'='*70}")
    for i, qa in enumerate(qa_pairs, 1):
        name = qa.get("questioner_name") or "Unknown"
        affil = qa.get("questioner_affiliation") or ""
        qs = qa.get("question_start", 0)
        qe = qa.get("question_end", 0)
        answers = qa.get("answers", [])

        q_text = slice_segments(segments, qs, qe)[:150]
        who = f"{name} ({affil})" if affil else name

        print(f"\n  Q{i}. [{qs}s-{qe}s] {who}:")
        print(f"       {q_text}...")
        for j, ans in enumerate(answers, 1):
            a_name = ans.get("answerer_name") or "Unknown"
            a_s = ans.get("answer_start", 0)
            ae = ans.get("answer_end", 0)
            a_text = slice_segments(segments, a_s, ae)[:150]
            prefix = f"  A{j}" if len(answers) > 1 else "   A"
            print(f"    {prefix}: [{a_s}s-{ae}s] {a_name}:")
            print(f"       {a_text}...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: Classify event type, then extract Q&A time boundaries"
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
        jsons = [j for j in jsons if ".qa" not in j.name]
        if not jsons:
            print("No transcript files found in", TRANSCRIPTS_DIR)
            sys.exit(1)
        transcript_path = jsons[0]

    print(f"Transcript: {transcript_path.name}")
    print(f"Model: {args.model}")
    print(f"Ollama URL: {args.ollama_url}")

    data = load_transcript(transcript_path)
    segments = data["segments"]

    # Phase 1: Classification
    if args.event_type:
        event_type = args.event_type
        confidence = "override"
        print(f"\n  Event type: {event_type} (user override)")
    else:
        event_type, confidence = classify_event(
            segments, args.ollama_url, args.model, args.temperature,
        )
        print(f"  Event type: {event_type} (confidence: {confidence})")

    # Phase 2: Extraction
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
        name = qa.get("questioner_name") or "Unknown"
        affil = qa.get("questioner_affiliation") or ""
        answers = qa.get("answers", [])
        qs = qa.get("question_start", "?")
        qe = qa.get("question_end", "?")
        tag = f" ({affil})" if affil else ""
        answerers = ", ".join(a.get("answerer_name") or "?" for a in answers) or "?"
        last_end = answers[-1].get("answer_end", "?") if answers else "?"
        print(f"  {i:2d}. [{qs:>7}s-{qe:>7}s] {name}{tag} -> {answerers}  (answer ends {last_end}s)")

    # Optional preview with actual text
    if args.preview:
        print_qa_preview(qa_pairs, segments)

    # Coverage analysis — warn about large uncovered gaps
    coverage = compute_coverage(qa_pairs, segments)
    if coverage["gaps"]:
        print(f"\n  Coverage: {coverage['covered_pct']:.0f}% of Q&A region "
              f"({coverage['qa_start']:.0f}s - {coverage['qa_end']:.0f}s)")
        print(f"  WARNING: {len(coverage['gaps'])} gap(s) > 2 min may contain missed questions:")
        for gap in coverage["gaps"]:
            print(f"    {gap['start']:.0f}s - {gap['end']:.0f}s ({gap['duration']:.0f}s / {gap['duration']/60:.1f} min)")

    # Save — timestamps-only output, no extracted text
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
