"""Shared Q&A extraction utilities.

Used by Stage 5b (chunk-based extraction) and Stage 5x (per-segment extraction).
Contains parsing, normalization, and quality-filtering logic so neither script
duplicates these functions.
"""

from __future__ import annotations

from collections import defaultdict


# ---------------------------------------------------------------------------
# Quality-filter constants
# ---------------------------------------------------------------------------

BOILERPLATE_PHRASES: tuple[str, ...] = (
    "are you ready for the event",
    "this concludes the event",
    "this concludes our event",
    "thank you to all participants",
    "station, this is houston",
    "houston, this is station",
    "we are ready for the event",
    "ready for the event",
    "press star 1 to ask",
    "press star one to ask",
    "our next question comes from",
    "our first question comes from",
)

MIN_QUESTION_WORDS = 4


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

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
        if not line or line.upper() == "NONE":
            continue
        # Skip lines that are clearly commentary
        if line[0].isalpha() or line.startswith("[") or line.startswith("{"):
            continue
        parts = line.split("|")
        floats: list[float] = []
        for p in parts:
            p = p.strip()
            try:
                floats.append(float(p))
            except (ValueError, TypeError):
                break
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
# Normalization
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
    MERGE_THRESHOLD = 15.0  # seconds
    merged: list[dict] = []
    for p in sorted(pairs, key=lambda x: x.get("question_start") or 0):
        qs = p.get("question_start") or 0
        matched = False
        for m in merged:
            m_orig_qs = m.get("_orig_qs") or m.get("question_start") or 0
            if abs(qs - m_orig_qs) <= MERGE_THRESHOLD:
                m["question_start"] = min(m.get("question_start") or 0, p.get("question_start") or 0)
                m["question_end"] = max(m.get("question_end") or 0, p.get("question_end") or 0)
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

    for m in merged:
        m.pop("_orig_qs", None)

    # Step 4: sort chronologically
    for m in merged:
        m["answers"] = sorted(m["answers"], key=lambda a: a.get("answer_start") or 0)

    sorted_pairs = sorted(merged, key=lambda x: x.get("question_start") or 0)

    # Step 5: absorb chunk-overlap artifacts
    OVERLAP_TOLERANCE = 5.0
    cleaned: list[dict] = []
    for p in sorted_pairs:
        if cleaned:
            prev = cleaned[-1]
            prev_orig_qe = prev.get("_orig_qe") or prev.get("question_end") or 0
            qs = p.get("question_start") or 0
            qe = p.get("question_end") or 0
            if qs < prev_orig_qe - OVERLAP_TOLERANCE:
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

    for c in cleaned:
        c.pop("_orig_qe", None)

    # Step 6: fix / filter pairs with overlapping Q/A timestamps
    valid: list[dict] = []
    for p in cleaned:
        qs = p.get("question_start") or 0
        qe = p.get("question_end") or 0
        answers = p.get("answers", [])
        good_answers = []
        for a in answers:
            a_start = a.get("answer_start") or 0
            a_end = a.get("answer_end") or 0
            if a_start < qe and a_end > qe + 1.0:
                a["answer_start"] = qe
                good_answers.append(a)
            elif a_start >= qe - 1.0:
                good_answers.append(a)
            elif (a_end - a_start) > 3 * max(qe - qs, 1) and (a_end - a_start) > 10:
                a["answer_start"] = qe
                good_answers.append(a)
        if good_answers:
            p["answers"] = good_answers
            valid.append(p)

    # Step 7: trim answer ranges that overlap into the NEXT pair's question
    for i in range(len(valid) - 1):
        next_qs = valid[i + 1].get("question_start") or 0
        trimmed = []
        for a in valid[i].get("answers", []):
            a_start = a.get("answer_start") or 0
            a_end = a.get("answer_end") or 0
            if a_start >= next_qs:
                continue
            if a_end > next_qs:
                a["answer_end"] = next_qs
                a_end = next_qs
            if a_end - a_start > 2.0:
                trimmed.append(a)
        valid[i]["answers"] = trimmed

    valid = [p for p in valid if p.get("answers")]
    return valid


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

def _dominant_speaker(segments: list[dict], start: float, end: float) -> str | None:
    """Return the speaker with the most talk-time in [start, end], or None."""
    durations: dict[str, float] = defaultdict(float)
    for seg in segments:
        sp = seg.get("speaker")
        if not sp:
            continue
        ov_start = max(seg["start"], start)
        ov_end = min(seg["end"], end)
        if ov_end > ov_start:
            durations[sp] += ov_end - ov_start
    if not durations:
        return None
    return max(durations, key=lambda k: durations[k])


def apply_quality_filters(
    qa_pairs: list[dict],
    segments: list[dict],
    *,
    slice_fn,
) -> list[dict]:
    """Apply content-level quality filters to extracted Q&A pairs.

    Filters applied (in order):
    - Drop pairs with no answers
    - Drop pairs with empty question text
    - Drop questions shorter than MIN_QUESTION_WORDS words
    - Drop mission-control / event-logistics boilerplate
    - Drop voice-check and tech-check greetings
    - Drop self-answered pairs (all answers from same speaker as question)

    ``slice_fn`` is ``transcript_utils.slice_segments`` — passed in to avoid a
    circular import between this module and transcript_utils.
    """
    clean: list[dict] = []
    for pair in qa_pairs:
        q_start = pair.get("question_start", 0)
        q_end = pair.get("question_end", 0)
        answers = pair.get("answers", [])

        if not answers:
            continue

        q_text = slice_fn(segments, q_start, q_end).strip()

        if not q_text:
            continue

        q_words = q_text.split()

        if len(q_words) < MIN_QUESTION_WORDS:
            continue

        q_lower = q_text.lower()
        if any(phrase in q_lower for phrase in BOILERPLATE_PHRASES):
            continue

        if len(q_words) <= 10:
            q_trimmed = q_lower.rstrip("?.! ")
            if any((
                "hear me" in q_trimmed,
                "hear us" in q_trimmed,
                q_trimmed.startswith(("hello", "hi ", "hey ", "good morning",
                                      "good afternoon", "good evening")),
                "checking" in q_trimmed and any(
                    w in q_trimmed for w in ("audio", "video", "connection")
                ),
                "are we connected" in q_trimmed,
                "can you see" in q_trimmed,
                "can you hear" in q_trimmed,
            )):
                continue

        q_speaker = _dominant_speaker(segments, q_start, q_end)
        if q_speaker:
            # Only drop pairs where ALL *known* answer speakers match the questioner.
            # Using `if sp` to filter Nones caused a bug: all() on an empty
            # iterable returns True, incorrectly dropping pairs with no diarization.
            known_ans_speakers = [
                sp for sp in
                (_dominant_speaker(segments, a["answer_start"], a["answer_end"])
                 for a in answers)
                if sp is not None
            ]
            if known_ans_speakers and all(sp == q_speaker for sp in known_ans_speakers):
                continue

        clean.append(pair)

    dropped = len(qa_pairs) - len(clean)
    if dropped:
        print(f"  Quality filters: dropped {dropped}/{len(qa_pairs)} low-quality pair(s)")
    return clean


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def compute_coverage(qa_pairs: list[dict], segments: list[dict]) -> dict:
    """Analyse how well the Q&A pairs cover the transcript timeline.

    Returns dict with qa_start, qa_end, covered_pct, and gaps > 2 min.
    """
    if not qa_pairs:
        return {"qa_start": 0, "qa_end": 0, "covered_pct": 0, "gaps": []}

    first_q = qa_pairs[0].get("question_start", 0)
    last_answers = qa_pairs[-1].get("answers", [])
    last_end = max((a.get("answer_end", 0) for a in last_answers), default=0)
    total_span = last_end - first_q if last_end > first_q else 1

    covered = 0.0
    prev_end = first_q
    gaps: list[dict] = []
    for p in qa_pairs:
        qs = p.get("question_start", 0)
        answers = p.get("answers", [])
        ae = max((a.get("answer_end", 0) for a in answers), default=qs)
        gap_dur = qs - prev_end
        if gap_dur > 120:
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
