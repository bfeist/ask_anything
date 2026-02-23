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
    CLASSIFY_DIR,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    QA_DIR,
    QA_EMPTY_JSONL,
    TRANSCRIPTS_DIR,
    ensure_directories,
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
# Empty-result tracking
# ---------------------------------------------------------------------------

def _record_empty(
    transcript_path: Path,
    *,
    event_type: str,
    confidence: str,
    model: str,
    reason: str,
) -> None:
    """Append one line to qa_empty.jsonl for a transcript that yielded no Q&A."""
    record = {
        "transcript_file": transcript_path.name,
        "event_type": event_type,
        "event_type_confidence": confidence,
        "model": model,
        "reason": reason,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with QA_EMPTY_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Event types and prompt routing
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

# Event types that should skip Q&A extraction entirely
SKIP_EXTRACTION_TYPES = {"produced_content"}

# For student Q&A, plain transcript works better (proven).
# For press conferences and media interviews, diarized labels help.
# For produced interviews, plain works (single speaker).
USE_DIARIZATION_FOR = {"press_conference", "media_interview", "panel"}

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
a moderator introduces a reporter/host, the reporter/host asks one or more questions, \
then one or more panelists/crew members answer. Multi-part questions from the same \
speaker count as ONE question block.

WHAT COUNTS AS A QUESTION:
- A genuine interrogative or request for information directed at panelists/crew.
- Must contain a question mark or be a clear request like "Tell us about..." or \
"Describe...".
- Short handoffs like "go ahead" or phrases like "I'll echo that" are NOT questions.
- Voice checks, greetings, and protocol exchanges are NOT questions.

WHAT DOES NOT COUNT:
- Opening speeches, prepared remarks, or introductory bios — no matter how many \
speaker changes occur during them.
- A panelist setting context or expanding on someone else's statement.
- The moderator handing over to the next speaker.
- Closing remarks or thank-you exchanges.

CRITICAL RULES:
1. A Q&A pair MUST involve at least two different speakers — someone who asks and \
someone DIFFERENT who answers. If the same speaker asks AND answers, it is NOT valid.
2. question_start to question_end is the ENTIRE span where the questioner speaks, \
including setup/context/multi-part questions.
3. answer_start to answer_end covers ONLY the respondent (a different speaker). \
Do NOT include the questioner's own statements in the answers array.
4. If the questioner speaks again after the respondent, that starts a NEW exchange, \
not part of the previous answer.
5. Each answer segment must be at least 10 seconds long — brief acknowledgments \
("thank you", "okay", "go ahead") are NOT answers.
6. Output items in CHRONOLOGICAL ORDER by question_start timestamp.

Your task:
1. Identify EVERY real question and ALL subsequent answer(s) from different speakers.
2. Use speaker labels to determine boundaries. When the speaker changes from the \
respondent back to a questioner, the answer ends.
3. If MULTIPLE panelists answer the SAME question (different speakers), include EACH \
as a separate entry in the "answers" array.
4. For each Q&A pair output ONLY:
   - question_start: timestamp where the questioner starts speaking
   - question_end: timestamp where the questioner finishes (before a DIFFERENT \
speaker begins answering)
   - answers: Array of answer objects from DIFFERENT speaker(s) than the questioner:
     [{"answer_start": <float>, "answer_end": <float>}, ...]

Do NOT include any text, names, or affiliations — ONLY timestamps.
IMPORTANT: Process the ENTIRE transcript from start to finish. Do NOT stop early.
A typical NASA press conference has 10-15 questions. If you have found fewer than 8, \
re-scan the transcript.

Respond ONLY with a JSON array of objects. No markdown fencing, no commentary.\
"""

EXTRACT_PROMPT_MEDIA_INTERVIEW = """\
You are a precise transcript analyst. Your job is to identify the time boundaries of \
ALL question-and-answer exchanges in this NASA media interview transcript.

This is a TV station, radio, or newspaper interview with an astronaut on the \
International Space Station. The typical pattern: a host/anchor asks questions, and \
the astronaut answers. Sometimes there are 2 astronauts answering.

The transcript has timestamps and speaker labels (SPEAKER_XX).

CRITICAL RULES:
1. A Q&A pair MUST involve at least two different speakers — the host who asks and \
the astronaut(s) who answer. If the same speaker asks AND "answers," it is NOT a \
valid Q&A pair.
2. question_start to question_end covers the ENTIRE span where the host speaks \
their question, including setup and context.
3. answer_start to answer_end covers ONLY the astronaut's response (a DIFFERENT \
speaker). Do NOT include the host's own statements in the answers array.
4. If the host speaks again after the astronaut, that is either a new question or \
a transition — NOT part of the previous answer.
5. Do NOT include the host's introduction, greetings, sign-off, or Mission Control \
voice checks as Q&A pairs.
6. Only include actual QUESTIONS (interrogative statements or clear prompts) \
followed by substantive answers from a different speaker.
7. If both astronauts answer the same question sequentially, include EACH answer \
as a separate entry in the "answers" array.
8. Output items in CHRONOLOGICAL ORDER by question_start timestamp.

Your task:
1. Identify EVERY question from the interviewer and the astronaut's answer(s).
2. Use speaker labels to determine boundaries — when the speaker changes from the \
astronaut back to the host, the answer ends.
3. For each Q&A pair output ONLY:
   - question_start: timestamp where the host starts asking
   - question_end: timestamp where the host finishes (before a DIFFERENT speaker \
begins answering)
   - answers: Array of answer objects from DIFFERENT speaker(s) than the host:
     [{{"answer_start": <float>, "answer_end": <float>}}, ...]

Do NOT include any text, names, or affiliations — ONLY timestamps.
IMPORTANT: Process the ENTIRE transcript from start to finish. Do NOT stop early.
A typical media interview has 5-15 questions.

Respond ONLY with a JSON array of objects. No markdown fencing, no commentary.\
"""

EXTRACT_PROMPT_PRODUCED_INTERVIEW = """\
You are a precise transcript analyst. This is a pre-produced interview where an \
astronaut answers questions from an off-camera interviewer whose audio has been \
EDITED OUT. Only the astronaut's responses are audible — there is only ONE speaker.

Your job is to identify the TOPIC SEGMENTS: distinct answer blocks where the \
astronaut shifts to responding to a new (unheard) question.

The transcript has timestamps in seconds. The PRIMARY and MOST RELIABLE signal \
for topic boundaries is GAPS in the timestamps — pauses of 5+ seconds where the \
off-camera interviewer asked their next question. Focus on these gaps above all else.

HOW TO IDENTIFY SEGMENTS:
1. Look at the timestamps. Find every gap of 5+ seconds between consecutive \
transcript segments. Each gap corresponds to a new question being asked.
2. Everything between two gaps is ONE answer/segment.
3. The astronaut's first words mark the start of the first segment.
4. The astronaut's last words mark the end of the last segment.

For each topic segment, output:
- answer_start: timestamp (seconds) where the astronaut begins this answer
- answer_end: timestamp (seconds) where the astronaut finishes this answer

CRITICAL RULES:
1. ONLY split at timestamp gaps >= 5 seconds. Do NOT split within continuous speech.
2. If the astronaut talks for 2 minutes without a 5-second gap, that is ONE segment.
3. The number of segments should roughly equal the number of large gaps + 1.
4. Output items in CHRONOLOGICAL ORDER by answer_start.

Respond ONLY with a JSON array of objects. No markdown fencing, no commentary.

Example output format:
[
  {{"answer_start": 3.0, "answer_end": 45.2}},
  {{"answer_start": 49.8, "answer_end": 97.5}},
  ...
]\
"""

EXTRACT_PROMPT_GENERIC = """\
You are a precise transcript analyst. Your job is to identify the time boundaries of \
ALL question-and-answer exchanges in a NASA event transcript.

The transcript has timestamps in seconds and may include speaker labels. Identify \
every instance where someone asks a question and receives a substantive answer \
from a DIFFERENT speaker.

CRITICAL RULES:
1. A Q&A pair MUST involve at least two different speakers — one who asks and one \
who answers. If only one speaker is talking, it is NOT a Q&A pair.
2. Only include actual QUESTIONS followed by substantive answers.
3. Do NOT include opening remarks, prepared statements, speeches, or closing remarks.
4. A "question" must be an interrogative statement or a clear prompt for information \
from one speaker, followed by a response from a DIFFERENT speaker.
5. The answer MUST be from a different speaker than the questioner.
6. Output items in CHRONOLOGICAL ORDER by question_start.

Your task:
1. Identify EVERY Q&A exchange from beginning to end.
2. Skip opening remarks, prepared statements, and closing remarks.
3. For each Q&A pair output ONLY:
   - question_start: timestamp where the question begins
   - question_end: timestamp where the question ends
   - answers: Array of answer objects from DIFFERENT speaker(s) than the questioner:
     [{{"answer_start": <float>, "answer_end": <float>}}, ...]

Do NOT include any text, names, or affiliations — ONLY timestamps.
IMPORTANT: Process the ENTIRE transcript. Do NOT stop early.

Respond ONLY with a JSON array of objects. No markdown fencing, no commentary.\
"""

EXTRACT_PROMPTS = {
    "student_qa": EXTRACT_PROMPT_STUDENT_QA,
    "press_conference": EXTRACT_PROMPT_PRESS_CONFERENCE,
    "media_interview": EXTRACT_PROMPT_MEDIA_INTERVIEW,
    "produced_interview": EXTRACT_PROMPT_PRODUCED_INTERVIEW,
    "panel": EXTRACT_PROMPT_GENERIC,
    "other": EXTRACT_PROMPT_GENERIC,
}

# Appended to the retry prompt when the first response contains null timestamps.
_SCHEMA_REMINDER = (
    "\n\nYOUR PREVIOUS RESPONSE CONTAINED null VALUES FOR TIMESTAMP FIELDS. "
    "THIS IS NOT ALLOWED.\n"
    "Every timestamp field MUST be a number (float). "
    "If you are uncertain about a boundary, OMIT the entire pair rather than using null.\n"
    "Required schema — no other fields, no null values:\n"
    '  {"question_start": <float>, "question_end": <float>, '
    '"answers": [{"answer_start": <float>, "answer_end": <float>}]}\n'
    "Respond ONLY with a corrected JSON array. No markdown fencing, no commentary."
)


def _validate_qa_pairs(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split pairs into (valid, malformed).

    A pair is malformed if any timestamp field is not a real number (e.g. None).
    """
    valid: list[dict] = []
    malformed: list[dict] = []
    for p in pairs:
        qs = p.get("question_start")
        qe = p.get("question_end")
        if not isinstance(qs, (int, float)) or not isinstance(qe, (int, float)):
            malformed.append(p)
            continue
        bad_answer = any(
            not isinstance(a.get("answer_start"), (int, float))
            or not isinstance(a.get("answer_end"), (int, float))
            for a in p.get("answers", [])
        )
        (malformed if bad_answer else valid).append(p)
    return valid, malformed


def _validate_produced_interview_pairs(
    pairs: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Validate produced_interview format: {answer_start, answer_end}.

    Returns (valid, malformed).
    """
    valid: list[dict] = []
    malformed: list[dict] = []
    for p in pairs:
        a_start = p.get("answer_start")
        a_end = p.get("answer_end")
        if (
            not isinstance(a_start, (int, float))
            or not isinstance(a_end, (int, float))
        ):
            malformed.append(p)
            continue
        valid.append(p)
    return valid, malformed


def _convert_produced_interview_pairs(pairs: list[dict]) -> list[dict]:
    """Convert produced_interview LLM output to standard qa_pairs format.

    Input:  [{answer_start, answer_end}, ...]
    Output: [{question_start, question_end, answers: [...]}, ...]

    Since the interviewer's audio was edited out, question_start == question_end
    (zero-width marker at the beginning of the answer).
    """
    qa_pairs = []
    for p in pairs:
        a_start = p["answer_start"]
        a_end = p["answer_end"]
        qa_pairs.append({
            "question_start": a_start,
            "question_end": a_start,  # zero-width: no audible question
            "answers": [{"answer_start": a_start, "answer_end": a_end}],
        })
    return qa_pairs


# ---------------------------------------------------------------------------
# Gap-based segmentation for produced interviews
# ---------------------------------------------------------------------------

_GAP_THRESHOLD = 3.0  # seconds — gaps >= this indicate a new question


def _segment_by_gaps(
    segments: list[dict],
    gap_threshold: float = _GAP_THRESHOLD,
) -> list[dict]:
    """Split transcript segments into topic blocks at timestamp gaps.

    For produced interviews where the interviewer's audio is edited out,
    large gaps (>= *gap_threshold* seconds) between consecutive transcript
    segments indicate points where a new question was asked.

    Returns standard qa_pairs format with zero-width question markers.
    """
    if not segments:
        return []

    # Sort segments by start time
    sorted_segs = sorted(segments, key=lambda s: s.get("start", 0))

    # Find gap positions
    block_start = sorted_segs[0]["start"]
    qa_pairs: list[dict] = []

    for i in range(1, len(sorted_segs)):
        prev_end = sorted_segs[i - 1].get("end", sorted_segs[i - 1]["start"])
        curr_start = sorted_segs[i]["start"]
        gap = curr_start - prev_end

        if gap >= gap_threshold:
            # End the current block and start a new one
            block_end = prev_end
            qa_pairs.append({
                "question_start": block_start,
                "question_end": block_start,  # zero-width
                "answers": [{"answer_start": block_start, "answer_end": block_end}],
            })
            block_start = curr_start

    # Final block
    block_end = sorted_segs[-1].get("end", sorted_segs[-1]["start"])
    qa_pairs.append({
        "question_start": block_start,
        "question_end": block_start,
        "answers": [{"answer_start": block_start, "answer_end": block_end}],
    })

    return qa_pairs


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
    system_prompt = EXTRACT_PROMPTS.get(event_type, EXTRACT_PROMPTS["other"])

    has_diarization = any(s.get("speaker") for s in segments)
    use_diarized = event_type in USE_DIARIZATION_FOR and has_diarization
    label = "diarized" if use_diarized else "plain"
    format_fn = format_diarized_transcript if use_diarized else format_plain_transcript

    # Chunk long transcripts for better coverage
    CHUNK_DURATION = 600.0   # 10 minutes per chunk
    OVERLAP = 60.0           # 1 minute overlap
    total_duration = segments[-1]["end"] if segments else 0

    if not segments:
        return None, 0.0, "(empty transcript — no segments)"

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
        if not chunk:
            print(f"    Chunk {ci}/{len(chunks)}: empty — skipping")
            continue
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
            if event_type == "produced_interview":
                valid, malformed = _validate_produced_interview_pairs(result)
            else:
                valid, malformed = _validate_qa_pairs(result)
            if malformed:
                print(f"    -> {len(malformed)} malformed pair(s) in chunk {ci} "
                      f"(null timestamps) — retrying...")
                retry_prompt = user_prompt + _SCHEMA_REMINDER
                t0r = time.time()
                raw2 = call_ollama(
                    ollama_url, model, retry_prompt,
                    system=system_prompt, temperature=temperature,
                )
                elapsed_r = time.time() - t0r
                total_elapsed += elapsed_r
                raw_parts.append(raw2)
                result2 = extract_json(raw2)
                if isinstance(result2, list):
                    if event_type == "produced_interview":
                        valid, still_bad = _validate_produced_interview_pairs(result2)
                    else:
                        valid, still_bad = _validate_qa_pairs(result2)
                    if still_bad:
                        print(f"    -> retry: {len(still_bad)} still malformed "
                              f"— dropping them")
                    print(f"    -> retry: {len(valid)} valid pairs in {elapsed_r:.1f}s")
                else:
                    print(f"    -> retry failed to parse ({elapsed_r:.1f}s): {raw2[:200]}")
            else:
                print(f"    -> {len(valid)} pairs in {elapsed:.1f}s")
            all_pairs.extend(valid)
        else:
            print(f"    -> FAILED to parse chunk {ci} ({elapsed:.1f}s): {raw[:200]}")

    print(f"  Total extraction time: {total_elapsed:.1f}s")

    if not all_pairs:
        return None, total_elapsed, "\n---\n".join(raw_parts)

    # Convert produced_interview format to standard qa_pairs before normalization
    if event_type == "produced_interview":
        all_pairs = _convert_produced_interview_pairs(all_pairs)

    qa_pairs = normalize_qa_pairs(all_pairs, skip_merge=(event_type == "produced_interview"))
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


def normalize_qa_pairs(pairs: list[dict], *, skip_merge: bool = False) -> list[dict]:
    """Normalize, merge, and sort Q&A pairs.

    1. Convert legacy flat format (answer_start, answer_end at top level)
       to the ``answers`` array format.
    2. Strip any name / affiliation fields.
    3. Merge entries that share overlapping question timestamps (within 15 s).
       Skipped when *skip_merge* is True (e.g. produced_interview segments).
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
    if skip_merge:
        merged = list(sorted(pairs, key=lambda x: x.get("question_start") or 0))
    else:
        MERGE_THRESHOLD = 15.0  # seconds
        merged = []
        for p in sorted(pairs, key=lambda x: x.get("question_start") or 0):
            qs = p.get("question_start") or 0
            qe = p.get("question_end") or 0
            matched = False
            for m in merged:
                mqs = m.get("question_start") or 0
                mqe = m.get("question_end") or 0
                if abs(qs - mqs) <= MERGE_THRESHOLD or (mqs <= qs <= mqe):
                    # Merge: extend question window and add answers
                    m["question_start"] = min(m.get("question_start") or 0, p.get("question_start") or 0)
                    m["question_end"] = max(m.get("question_end") or 0, p.get("question_end") or 0)
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
        m["answers"] = sorted(m["answers"], key=lambda a: a.get("answer_start") or 0)

    sorted_pairs = sorted(merged, key=lambda x: x.get("question_start") or 0)

    # Step 5: absorb chunk-overlap artifacts — entries whose question
    # window overlaps with a prior entry's question window (same question
    # detected in overlapping chunks).  We ONLY merge when the new pair's
    # question significantly overlaps the previous pair's question or answer
    # range AND the new pair starts within the overlap region.  Adjacent
    # pairs (next question starts right after previous answer) must NOT
    # be merged — they are separate Q&A exchanges.
    OVERLAP_TOLERANCE = 5.0  # seconds of slack for chunk-boundary artifacts
    cleaned: list[dict] = []
    for p in sorted_pairs:
        if cleaned:
            prev = cleaned[-1]
            prev_qe = prev.get("question_end") or 0
            prev_answer_end = max(
                (a.get("answer_end") or 0 for a in prev.get("answers", [])), default=0
            )
            qs = p.get("question_start") or 0
            qe = p.get("question_end") or 0

            # Only merge if the NEW question starts BEFORE the previous
            # question ends (genuine duplicate) — not merely before the
            # previous answer ends (which would be a separate exchange).
            is_dup_question = qs < prev_qe - OVERLAP_TOLERANCE

            if is_dup_question:
                # Merge: extend question window, fold in new answers
                prev["question_end"] = max(prev_qe or 0, qe or 0)
                existing_starts = {a.get("answer_start") for a in prev["answers"]}
                for a in p.get("answers", []):
                    a_start = a.get("answer_start", 0)
                    if not any(abs(a_start - es) < 5.0 for es in existing_starts if es is not None):
                        prev["answers"].append(a)
                        existing_starts.add(a_start)
                prev["answers"] = sorted(
                    prev["answers"], key=lambda a: a.get("answer_start") or 0
                )
                continue
        cleaned.append(p)

    # Step 6: fix / filter pairs with overlapping Q/A timestamps.
    # The LLM sometimes sets answer_start = question_start. Fix these
    # by bumping answer_start to question_end. Drop answers that are
    # entirely contained within the question with no extension beyond.
    valid: list[dict] = []
    for p in cleaned:
        qs = p.get("question_start") or 0
        qe = p.get("question_end") or 0
        answers = p.get("answers", [])

        good_answers = []
        for a in answers:
            a_start = a.get("answer_start") or 0
            a_end = a.get("answer_end") or 0

            # If answer starts before question ends but extends beyond,
            # fix the start to be at question_end
            if a_start < qe and a_end > qe + 1.0:
                a["answer_start"] = qe
                good_answers.append(a)
            # Answer starts after question — always valid
            elif a_start >= qe - 1.0:
                good_answers.append(a)
            # Answer completely inside question range but much longer
            elif (a_end - a_start) > 3 * max(qe - qs, 1) and (a_end - a_start) > 10:
                a["answer_start"] = qe
                good_answers.append(a)
            # Otherwise it's garbage — skip this answer

        if good_answers:
            p["answers"] = good_answers
            valid.append(p)

    # Step 7: trim answer ranges that overlap into the NEXT pair's question.
    # When the LLM tracks interleaved threads it sometimes assigns answer
    # chunks that temporally belong to the next exchange.  Drop answers
    # that start after the next question begins; trim answers that merely
    # extend past the next question start.
    for i in range(len(valid) - 1):
        next_qs = valid[i + 1].get("question_start") or 0
        trimmed = []
        for a in valid[i].get("answers", []):
            a_start = a.get("answer_start") or 0
            a_end = a.get("answer_end") or 0
            if a_start >= next_qs:
                continue  # answer belongs to the next exchange
            if a_end > next_qs:
                a["answer_end"] = next_qs  # trim overlap
                a_end = next_qs
            if a_end - a_start > 2.0:
                trimmed.append(a)
        valid[i]["answers"] = trimmed

    # Remove pairs that lost all answers after trimming
    valid = [p for p in valid if p.get("answers")]

    return valid


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

        if qs == qe:
            # Produced interview: no audible question, show as topic segment
            print(f"\n  Q{i}. [topic segment @ {qs:.1f}s]:")
        else:
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
    cls_path = CLASSIFY_DIR / (transcript_path.stem + ".classify.json")
    if cls_path.exists():
        try:
            cls = json.loads(cls_path.read_text(encoding="utf-8"))
            return cls["event_type"], cls.get("confidence", "low")
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def process_transcript(
    transcript_path: Path,
    *,
    model: str,
    ollama_url: str,
    temperature: float,
    event_type_override: str | None,
    preview: bool,
) -> bool:
    """Extract Q&A from a single transcript.  Returns True on success."""
    print(f"\nTranscript: {transcript_path.name}")

    data = load_transcript(transcript_path)
    segments = data["segments"]

    # Resolve event type: CLI flag > .classify.json > inline classification
    if event_type_override:
        event_type = event_type_override
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
                segments, ollama_url, model, temperature,
                transcript_filename=transcript_path.name,
            )
            print(f"  Event type: {event_type} (confidence: {confidence})")

    # Extraction
    if not segments:
        print(f"  Skipping Q&A extraction — transcript has no segments.")
        _record_empty(
            transcript_path,
            event_type=event_type,
            confidence=confidence,
            model=model,
            reason="empty_transcript",
        )
        print(f"  Logged to {QA_EMPTY_JSONL.name}")
        output = {
            "transcript_file": transcript_path.name,
            "model": model,
            "event_type": event_type,
            "event_type_confidence": confidence,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "skipped": True,
            "skip_reason": "Empty transcript (no segments).",
            "qa_pairs": [],
        }
        out_path = QA_DIR / (transcript_path.stem + ".qa.json")
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved (skipped): {out_path}")
        return True

    if event_type in SKIP_EXTRACTION_TYPES:
        print(f"  Skipping Q&A extraction for event type '{event_type}' (no Q&A expected).")
        # Save a minimal output indicating skip
        output = {
            "transcript_file": transcript_path.name,
            "model": model,
            "event_type": event_type,
            "event_type_confidence": confidence,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "skipped": True,
            "skip_reason": f"Event type '{event_type}' does not contain Q&A content.",
            "qa_pairs": [],
        }
        out_path = QA_DIR / (transcript_path.stem + ".qa.json")
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved (skipped): {out_path}")
        return True

    if event_type == "produced_interview":
        # Deterministic gap-based segmentation — no LLM needed
        qa_pairs = _segment_by_gaps(segments)
        elapsed = 0.0
        raw = ""
        print(f"\n  {len(qa_pairs)} topic segments detected (gap-based, threshold={_GAP_THRESHOLD}s)")
    else:
        qa_pairs, elapsed, raw = run_extraction(
            segments, event_type, ollama_url, model, temperature=temperature,
        )

    if qa_pairs is None:
        print("\nFAILED to parse JSON from response.")
        print("Raw response (first 1000 chars):")
        print(raw[:1000])
        _record_empty(
            transcript_path,
            event_type=event_type,
            confidence=confidence,
            model=model,
            reason="extraction_failed",
        )
        return False

    # Track empty results before summarising
    if not qa_pairs:
        print(f"\n  0 Q&A pairs found in {elapsed:.1f}s")
        _record_empty(
            transcript_path,
            event_type=event_type,
            confidence=confidence,
            model=model,
            reason="no_qa_found",
        )
    else:
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
    if preview:
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
        "model": model,
        "event_type": event_type,
        "event_type_confidence": confidence,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "qa_pairs": qa_pairs,
    }
    if coverage["gaps"]:
        output["coverage_gaps"] = coverage["gaps"]
    out_path = QA_DIR / (transcript_path.stem + ".qa.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5b: Extract Q&A time boundaries from transcripts"
    )
    parser.add_argument(
        "--transcript", type=Path, default=None,
        help="Process a single transcript JSON instead of all transcripts.",
    )
    parser.add_argument("--model", default=env_or_default("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    parser.add_argument("--ollama-url", default=env_or_default("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--event-type", choices=EVENT_TYPES, default=None,
        help="Skip classification and use this event type directly (single-file mode only).",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Print a human-readable preview with text sliced from transcript.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if a .qa.json already exists.",
    )
    args = parser.parse_args()

    ensure_directories()

    print("=" * 70)
    print("Stage 5b: Q&A Extraction")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Ollama URL:  {args.ollama_url}")
    print(f"  Temperature: {args.temperature}")

    # Collect transcripts to process
    if args.transcript:
        if not args.transcript.exists():
            print(f"ERROR: File not found: {args.transcript}")
            sys.exit(1)
        transcripts = [args.transcript]
    else:
        transcripts = sorted(
            p for p in TRANSCRIPTS_DIR.glob("*.json")
            if ".qa" not in p.name and ".classify" not in p.name
        )

    print(f"  Transcripts found: {len(transcripts)}")

    # Filter already-processed unless --force
    if not args.force:
        pending = []
        for t in transcripts:
            qa_path = QA_DIR / (t.stem + ".qa.json")
            if not qa_path.exists():
                pending.append(t)
            else:
                # Re-process if qa_pairs is empty (e.g. previously skipped
                # produced_content that should now be produced_interview)
                try:
                    qa_data = json.loads(qa_path.read_text(encoding="utf-8"))
                    if not qa_data.get("qa_pairs"):
                        pending.append(t)
                except (json.JSONDecodeError, OSError):
                    pending.append(t)
    else:
        pending = list(transcripts)

    skipped = len(transcripts) - len(pending)
    if skipped:
        print(f"  Already extracted: {skipped} (use --force to redo)")
    print(f"  To process: {len(pending)}")

    if not pending:
        print("\nNothing to do.")
        return

    successes = 0
    failures = 0
    total_start = time.time()

    for i, transcript_path in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}]")
        ok = process_transcript(
            transcript_path,
            model=args.model,
            ollama_url=args.ollama_url,
            temperature=args.temperature,
            event_type_override=args.event_type,
            preview=args.preview,
        )
        if ok:
            successes += 1
        else:
            failures += 1

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Extraction complete: {successes} ok, {failures} failed in {total_time:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
