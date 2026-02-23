#!/usr/bin/env python3
"""Stage 5b: Extract Q&A time boundaries from a classified transcript.

Reads the classification from <stem>.classify.json (produced by 5a); skips
the file if not found.  Applies an event-type-specific LLM prompt
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
from astro_ia_harvest.ollama_client import call_ollama  # noqa: E402
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
    "other",
)

# Event types that should skip Q&A extraction entirely
SKIP_EXTRACTION_TYPES = {"produced_content"}

# When diarization labels are available, use them for these event types.
# Speaker labels help the model identify who is asking vs answering.
USE_DIARIZATION_FOR = {"press_conference", "media_interview", "panel", "student_qa"}

# ---------------------------------------------------------------------------
# Extraction prompts  (timestamps-only — no names / affiliations)
# ---------------------------------------------------------------------------

EXTRACT_PROMPT_STUDENT_QA = """\
You are a precise transcript analyst. Your job is to identify the time boundaries of \
ALL question-and-answer pairs from this school downlink / education event where \
students ask an astronaut questions.

The transcript has timestamps in seconds and may include speaker labels (SPEAKER_XX). \
The typical pattern: a moderator or student introduces the question, the student asks \
it, then the astronaut answers. There may be a satellite delay (~6 seconds) between \
the question and the answer, or the event may be direct (no delay). There are \
typically 10-20 questions — process the ENTIRE transcript to the very end and do not \
stop early.

If speaker labels are present, use them to determine boundaries:
- The moderator introduces students — this is NOT part of the question.
- The student's question starts when the student speaker begins and ends when they \
finish speaking.
- The astronaut's answer starts when a DIFFERENT speaker (the astronaut) begins \
responding and ends when that speaker finishes before the next question cycle.

CRITICAL RULES:
1. question_start is the timestamp where the STUDENT starts their actual question \
(not the moderator's introduction).
2. question_end is where the student finishes speaking.
3. answer_start is where the astronaut begins responding.
4. answer_end is where the astronaut finishes, before the next moderator introduction \
or student question.
5. Skip preamble, voice checks, Mission Control chatter, and closing remarks.
6. Output items in CHRONOLOGICAL ORDER by question_start.

Do NOT include any text, names, or affiliations — ONLY timestamps.
Every timestamp MUST be a number (float).
IMPORTANT: Process the ENTIRE transcript. Do NOT stop early.

Output ONE line per Q&A pair in this exact pipe-delimited format:
  question_start|question_end|answer_start|answer_end

Example:
  899.7|927.7|928.8|982.4

No headers, no commentary, no blank lines, no JSON, no markdown.\
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
Do NOT include the questioner's own statements in the answers.
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
as a separate answer on the same line.

Do NOT include any text, names, or affiliations — ONLY timestamps.
IMPORTANT: Process the ENTIRE transcript from start to finish. Do NOT stop early.
A typical NASA press conference has 10-15 questions. If you have found fewer than 8, \
re-scan the transcript.

Output ONE line per Q&A pair in this exact pipe-delimited format:
  question_start|question_end|answer1_start|answer1_end[|answer2_start|answer2_end]

Examples:
  Single answerer:     899.7|927.7|928.8|982.4
  Multiple answerers:  899.7|927.7|928.8|982.4|983.0|1010.5

No headers, no commentary, no blank lines, no JSON, no markdown.\
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
speaker). Do NOT include the host's own statements in the answers.
4. If the host speaks again after the astronaut, that is either a new question or \
a transition — NOT part of the previous answer.
5. Do NOT include the host's introduction, greetings, sign-off, or Mission Control \
voice checks as Q&A pairs.
6. Only include actual QUESTIONS (interrogative statements or clear prompts) \
followed by substantive answers from a different speaker.
7. If both astronauts answer the same question sequentially, include EACH answer \
on the same line.
8. Output items in CHRONOLOGICAL ORDER by question_start timestamp.

Your task:
1. Identify EVERY question from the interviewer and the astronaut's answer(s).
2. Use speaker labels to determine boundaries — when the speaker changes from the \
astronaut back to the host, the answer ends.

Do NOT include any text, names, or affiliations — ONLY timestamps.
IMPORTANT: Process the ENTIRE transcript from start to finish. Do NOT stop early.
A typical media interview has 5-15 questions.

Output ONE line per Q&A pair in this exact pipe-delimited format:
  question_start|question_end|answer1_start|answer1_end[|answer2_start|answer2_end]

Examples:
  Single answerer:     899.7|927.7|928.8|982.4
  Multiple answerers:  899.7|927.7|928.8|982.4|983.0|1010.5

No headers, no commentary, no blank lines, no JSON, no markdown.\
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

Do NOT include any text, names, or affiliations — ONLY timestamps.
IMPORTANT: Process the ENTIRE transcript. Do NOT stop early.

Output ONE line per Q&A pair in this exact pipe-delimited format:
  question_start|question_end|answer1_start|answer1_end[|answer2_start|answer2_end]

Examples:
  Single answerer:     899.7|927.7|928.8|982.4
  Multiple answerers:  899.7|927.7|928.8|982.4|983.0|1010.5

No headers, no commentary, no blank lines, no JSON, no markdown.\
"""

EXTRACT_PROMPTS = {
    "student_qa": EXTRACT_PROMPT_STUDENT_QA,
    "press_conference": EXTRACT_PROMPT_PRESS_CONFERENCE,
    "media_interview": EXTRACT_PROMPT_MEDIA_INTERVIEW,
    "panel": EXTRACT_PROMPT_GENERIC,
    "other": EXTRACT_PROMPT_GENERIC,
}

# Appended to the retry prompt when the first response was unparseable.
_RETRY_REMINDER = (
    "\n\nYOUR PREVIOUS RESPONSE COULD NOT BE PARSED. "
    "You MUST output ONLY pipe-delimited lines with no other text.\n"
    "Format: question_start|question_end|answer_start|answer_end\n"
    "Example: 899.7|927.7|928.8|982.4\n"
    "Multiple answerers: 899.7|927.7|928.8|982.4|983.0|1010.5\n"
    "Every value must be a decimal number. No headers, no commentary, no JSON.\n"
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


def parse_pipe_qa(raw: str) -> list[dict]:
    """Parse pipe-delimited Q&A output from the LLM into structured dicts.

    Expected format (one line per Q&A pair):
        question_start|question_end|answer1_start|answer1_end[|answer2_start|answer2_end]

    Returns a list of dicts matching the qa_pairs schema.  Malformed lines are
    silently skipped so the pipeline never crashes on bad LLM output.
    """
    pairs: list[dict] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip lines that are clearly commentary (start with a letter, bracket, etc.)
        if line[0].isalpha() or line.startswith("[") or line.startswith("{"):
            continue
        parts = line.split("|")
        floats: list[float] = []
        for p in parts:
            p = p.strip()
            try:
                floats.append(float(p))
            except (ValueError, TypeError):
                break  # stop at first non-float field
        # Need at least 4 values (q_start, q_end, a_start, a_end) and even count
        if len(floats) < 4 or len(floats) % 2 != 0:
            continue
        answers = [
            {"answer_start": floats[i], "answer_end": floats[i + 1]}
            for i in range(2, len(floats), 2)
        ]
        pairs.append({
            "question_start": floats[0],
            "question_end": floats[1],
            "answers": answers,
        })
    return pairs


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
            num_predict=8192,
        )
        elapsed = time.time() - t0
        total_elapsed += elapsed
        raw_parts.append(raw)

        parsed = parse_pipe_qa(raw)
        valid, malformed = _validate_qa_pairs(parsed)
        if not valid and raw.strip():
            # Nothing parsed — retry once with format reminder
            print(f"    -> 0 pairs parsed from chunk {ci} — retrying...")
            retry_prompt = user_prompt + _RETRY_REMINDER
            t0r = time.time()
            raw2 = call_ollama(
                ollama_url, model, retry_prompt,
                system=system_prompt, temperature=temperature,
                num_predict=8192,
            )
            elapsed_r = time.time() - t0r
            total_elapsed += elapsed_r
            raw_parts.append(raw2)
            parsed2 = parse_pipe_qa(raw2)
            valid, malformed = _validate_qa_pairs(parsed2)
            if valid:
                print(f"    -> retry: {len(valid)} pairs in {elapsed_r:.1f}s")
            else:
                print(f"    -> retry also failed ({elapsed_r:.1f}s): {raw2[:200]}")
        elif malformed:
            print(f"    -> {len(valid)} valid, {len(malformed)} malformed (dropped) "
                  f"in {elapsed:.1f}s")
        else:
            print(f"    -> {len(valid)} pairs in {elapsed:.1f}s")
        all_pairs.extend(valid)

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

    # Step 3: merge entries with overlapping/nearby question START timestamps.
    # This deduplicates the same question detected in overlapping chunks.
    # IMPORTANT: We compare against the ORIGINAL question_start only (stored
    # in _orig_qs) to prevent cascading merges that snowball a growing
    # question_end window across the entire transcript.
    MERGE_THRESHOLD = 15.0  # seconds
    merged: list[dict] = []
    for p in sorted(pairs, key=lambda x: x.get("question_start") or 0):
        qs = p.get("question_start") or 0
        matched = False
        for m in merged:
            m_orig_qs = m.get("_orig_qs") or m.get("question_start") or 0
            if abs(qs - m_orig_qs) <= MERGE_THRESHOLD:
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
            p["_orig_qs"] = qs
            merged.append(p)

    # Remove internal tracking key
    for m in merged:
        m.pop("_orig_qs", None)

    # Step 4: sort chronologically and sort each answers list by answer_start
    for m in merged:
        m["answers"] = sorted(m["answers"], key=lambda a: a.get("answer_start") or 0)

    sorted_pairs = sorted(merged, key=lambda x: x.get("question_start") or 0)

    # Step 5: absorb chunk-overlap artifacts — entries whose question
    # window overlaps with a prior entry's question window (same question
    # detected in overlapping chunks).  We ONLY merge when the new pair's
    # question_start falls BEFORE the previous pair's ORIGINAL question_end.
    # We track the original question_end to prevent cascading merges.
    OVERLAP_TOLERANCE = 5.0  # seconds of slack for chunk-boundary artifacts
    cleaned: list[dict] = []
    for p in sorted_pairs:
        if cleaned:
            prev = cleaned[-1]
            prev_orig_qe = prev.get("_orig_qe") or prev.get("question_end") or 0
            qs = p.get("question_start") or 0
            qe = p.get("question_end") or 0

            # Only merge if the NEW question starts BEFORE the previous
            # question's ORIGINAL end (genuine duplicate from chunk overlap).
            is_dup_question = qs < prev_orig_qe - OVERLAP_TOLERANCE

            if is_dup_question:
                # Merge: extend question window, fold in new answers
                prev["question_end"] = max(prev.get("question_end") or 0, qe or 0)
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
        p["_orig_qe"] = p.get("question_end") or 0
        cleaned.append(p)

    # Remove internal tracking key
    for c in cleaned:
        c.pop("_orig_qe", None)

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
            # Zero-width question (e.g. diarization gap)
            print(f"\n  Q{i}. [@ {qs:.1f}s]:")
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
            print("  No .classify.json found — skipping.")
            return False

    # Extraction
    MIN_TEXT_CHARS = 200  # skip transcripts with negligible text content
    total_text = sum(len(s.get("text", "")) for s in segments)
    if not segments or total_text < MIN_TEXT_CHARS:
        reason = "empty_transcript" if not segments else "insufficient_text"
        detail = (
            "Empty transcript (no segments)."
            if not segments
            else f"Transcript too short ({total_text} chars, min {MIN_TEXT_CHARS})."
        )
        print(f"  Skipping Q&A extraction — {detail}")
        _record_empty(
            transcript_path,
            event_type=event_type,
            confidence=confidence,
            model=model,
            reason=reason,
        )
        print(f"  Logged to {QA_EMPTY_JSONL.name}")
        output = {
            "transcript_file": transcript_path.name,
            "model": model,
            "event_type": event_type,
            "event_type_confidence": confidence,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "skipped": True,
            "skip_reason": detail,
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

    qa_pairs, elapsed, raw = run_extraction(
        segments, event_type, ollama_url, model, temperature=temperature,
    )

    if qa_pairs is None:
        print("\nFAILED to parse Q&A from response.")
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
        "--force", nargs="?", const="all", default=None,
        metavar="EVENT_TYPE",
        help="Re-extract even if a .qa.json already exists. "
             "Optionally specify an event type (e.g. --force student_qa) "
             "to only re-extract that type. Use --force alone to redo all.",
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
    force_type = args.force  # None, "all", or an event type like "student_qa"
    if force_type and force_type not in ("all", *EVENT_TYPES):
        print(f"ERROR: Unknown event type for --force: {force_type}")
        print(f"  Valid types: {', '.join(EVENT_TYPES)}")
        sys.exit(1)

    if force_type:
        print(f"  Force mode: {'all types' if force_type == 'all' else force_type}")

    pending = []
    forced = 0
    for t in transcripts:
        qa_path = QA_DIR / (t.stem + ".qa.json")
        if not qa_path.exists():
            pending.append(t)
            continue

        # Existing .qa.json — decide whether to re-process
        try:
            qa_data = json.loads(qa_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pending.append(t)
            continue

        # Always re-process if qa_pairs is empty (e.g. previously skipped
        # type that was reclassified after re-running 5a)
        if not qa_data.get("qa_pairs"):
            pending.append(t)
            continue

        # --force: re-process based on force scope
        if force_type == "all":
            pending.append(t)
            forced += 1
        elif force_type:
            # Only force files whose classified event_type matches
            existing_type = qa_data.get("event_type", "")
            if existing_type == force_type:
                pending.append(t)
                forced += 1

    skipped = len(transcripts) - len(pending)
    if skipped:
        print(f"  Already extracted: {skipped} (use --force to redo)")
    if forced:
        print(f"  Forced re-extraction: {forced}")
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
