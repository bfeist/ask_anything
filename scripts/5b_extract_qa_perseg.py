#!/usr/bin/env python3
"""Stage 5b: Per-segment Q&A extraction.

Two-pass approach:

  Pass 1 — Heuristic scan (zero LLM calls):
    Every transcript segment is tested against lightweight text rules
    (question marks, interrogative words) to produce a list of *candidate*
    anchors — segments that might be questions.  This is intentionally
    high-recall; false positives are expected and handled in Pass 2.

  Pass 2 — LLM confirmation (one call per candidate group):
    A focused context window (configurable pre/post seconds around each
    anchor) is sent to the LLM with the candidate segment highlighted.
    The LLM answers one narrow question: "Is this a real question with an
    answer from a different speaker?  If yes, give me the timestamps."
    Each call is small (~1-2 min of transcript) and independent, so they
    can be fully parallelised with --workers N.

Usage:
  uv run python scripts/5b_extract_qa_perseg.py
  uv run python scripts/5b_extract_qa_perseg.py --transcript data/transcripts/<file>.json
  uv run python scripts/5b_extract_qa_perseg.py --workers 4
  uv run python scripts/5b_extract_qa_perseg.py --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from astro_ia_harvest.config import (  # noqa: E402
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    NO_AUDIO_JSONL,
    NON_ENGLISH_JSONL,
    QA_DIR,
    QA_EMPTY_JSONL,
    TRANSCRIPTS_DIR,
    ensure_directories,
    env_or_default,
)
from astro_ia_harvest.jsonl_utils import load_jsonl  # noqa: E402
from astro_ia_harvest.ollama_client import call_ollama  # noqa: E402
from astro_ia_harvest.qa_utils import (  # noqa: E402
    apply_quality_filters,
    compute_coverage,
    normalize_qa_pairs,
    parse_pipe_qa,
    _validate_qa_pairs,
)
from astro_ia_harvest.transcript_utils import (  # noqa: E402
    format_diarized_transcript,
    format_plain_transcript,
    load_transcript,
    slice_segments,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default context window around each candidate anchor
DEFAULT_WINDOW_PRE = 20.0   # seconds of context before the candidate
DEFAULT_WINDOW_POST = 90.0  # seconds of context after the candidate

# Candidate segments within this gap are merged into one confirmation call
DEFAULT_GROUP_GAP = 10.0  # seconds

# Interrogative words/phrases that mark the start of a likely question
_INTERROGATIVE_STARTS = re.compile(
    r"^\s*(what|who|when|where|how|why|which|whose|whom"
    # Auxiliary verb questions — expanded beyond just "you" to include any subject
    r"|can\s+\w+|could\s+\w+|will\s+\w+|would\s+\w+|should\s+\w+|might\s+\w+"
    r"|do\s+\w+|did\s+\w+|does\s+\w+"
    r"|have\s+\w+|has\s+\w+"
    r"|is\s+\w+|are\s+\w+|was\s+\w+|were\s+\w+"
    # Directives / invitations
    r"|tell\s+(?:us|me)\b|talk\s+(?:us|me)\s+through|describe\s+\w+"
    r"|walk\s+(?:us|me)\s+through|share\s+(?:what|how|why|with)\b"
    # First-person intro phrases
    r"|i(?:\s*'m|\s+am|\s+was)\s+(?:curious|wondering)\b"
    r"|i\s+(?:was\s+wondering|would\s+(?:like|love)\s+to|have\s+a\s+question|wanted\s+to\s+(?:ask|know))\b"
    # Student / audience intro: "My question is…" / "My name is…, my question…"
    r"|my\s+(?:question|name)\b"
    # Short-form questions
    r"|any\b"
    # "Since <you/it/we>…" preamble leading to a question
    r"|since\b"
    # "Now, <question-word>…" moderator/interviewer pivot
    r"|now\s*,?\s*(?:what|how|why|when|where|who|will|would|can|could|are|is|were|was|do|does|did|for|tell)\b"
    # "So what/how/…" or "So can/could/…" with optional comma
    r"|so\s*,?\s*(?:what|how|why|when|where|who|tell|can|could|will|would|should|do|does|did|is|are|was|were|have|has)\b"
    # "But how/what/…" — common interviewer follow-up pivot
    r"|but\s+(?:what|how|why|when|where|who|can|could|will|would|do|does|did|is|are|was|were)\b"
    # "For all/the/you/us… <question>" — moderator addressing a specific person
    r"|for\s+(?:all|the|each|every|those|you|us|any)\b"
    # "This question / This one's from…"
    r"|this\s+(?:question|one)\b"
    # "If you…" conditional question setup
    r"|if\s+you\b"
    # "Anything about/that…?" — shorthand direct question
    r"|anything\b"
    # Moderator intro: "We have a question from…"
    r"|we\s+(?:have|had)\s+a\b"
    r")\b",
    re.IGNORECASE,
)

# Leading salutation/name prefix to strip before checking interrogative start.
# Handles patterns like "Jim,", "Administrator,", "Bob and Doug," etc.
_SALUTATION_PREFIX = re.compile(
    r"^\s*(?:[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+){0,2},\s*){1,3}",
)

# ---------------------------------------------------------------------------
# System prompts for confirmation
# ---------------------------------------------------------------------------

_CONFIRM_SYSTEM = """\
You are analysing a short excerpt from a NASA video transcript.

A candidate segment is marked with >>> ... <<< delimiters.

Your task: determine whether that candidate is a GENUINE QUESTION that receives
a SUBSTANTIVE ANSWER from a DIFFERENT speaker in the surrounding text.

GENUINE QUESTION criteria (ALL must be true):
1. The candidate segment contains an interrogative sentence, or a clear prompt for
   information / opinion directed at another person.
2. A DIFFERENT speaker responds with ≥2 sentences of substantive content (not just
   "thank you" or "sure" or "go ahead").
3. This is not a readiness/tech check ("Can you hear me?", "Are you ready?"),
   greeting, sign-off, moderator hand-off, or Mission Control protocol phrase.
4. The exchange is not primarily about divisive political or social controversy
   (abortion, partisan elections, religious disputes, etc.). Questions about the
   astronaut's personal life, hobbies, family, feelings, or any other non-space
   topic ARE acceptable as long as they meet criteria 1-3.

When determining question_start, include any lead-up sentence(s) immediately
before the interrogative that provide context or set up the question — typically
1-2 sentences. For example, if the interviewer says "You've been up there for six
months. What's the hardest part?" the question_start should begin at "You've been
up there..." not at "What's the hardest part?". Only include lead-up if it is
clearly part of the same turn and directly relates to the question being asked.

If all criteria are met, output ONE pipe-delimited line:
  question_start|question_end|answer_start|answer_end

If multiple speakers give sequential answers to the SAME question:
  question_start|question_end|a1_start|a1_end|a2_start|a2_end

If criteria are NOT met, output exactly:
  NONE

ALL values must be decimal numbers (seconds from start of video).
question_start / question_end span the full question utterance.
answer_start / answer_end span only the answerer's response.
No other text. No commentary. No markdown fences.
"""

# Appended when the LLM returns an unparseable response.
_RETRY_REMINDER = (
    "\n\nYOUR PREVIOUS RESPONSE COULD NOT BE PARSED. "
    "Output ONLY a pipe-delimited line like: 899.7|927.7|928.8|982.4\n"
    "or the word NONE. No other text.\n"
)


# ---------------------------------------------------------------------------
# Empty-result tracking
# ---------------------------------------------------------------------------

def _record_empty(
    transcript_path: Path,
    *,
    model: str,
    reason: str,
) -> None:
    record = {
        "transcript_file": transcript_path.name,
        "model": model,
        "reason": reason,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with QA_EMPTY_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Pass 1: heuristic candidate scan
# ---------------------------------------------------------------------------

def find_candidates(segments: list[dict]) -> list[dict]:
    """Return segments that look like they might be questions.

    Criteria (any one is sufficient):
    - Text contains '?'
    - Text begins with an interrogative word/phrase
    - Text begins with a name/salutation prefix followed by an interrogative
      (e.g. "Jim, what do you think..." or "Administrator, tell us about...")

    Deliberately high-recall; Pass 2 will filter false positives.
    """
    results = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        if "?" in text:
            results.append(seg)
            continue
        # Try raw text first, then strip any leading salutation/name prefix
        text_for_match = _SALUTATION_PREFIX.sub("", text).strip()
        if _INTERROGATIVE_STARTS.match(text) or _INTERROGATIVE_STARTS.match(text_for_match):
            results.append(seg)
    return results


def group_candidates(
    candidates: list[dict],
    gap_threshold: float = DEFAULT_GROUP_GAP,
) -> list[list[dict]]:
    """Cluster candidates that are temporally close into confirmation groups.

    Segments within ``gap_threshold`` seconds of each other are merged so a
    single multi-segment question generates only one LLM call.
    """
    if not candidates:
        return []
    groups: list[list[dict]] = [[candidates[0]]]
    for cand in candidates[1:]:
        if cand["start"] - groups[-1][-1]["end"] <= gap_threshold:
            groups[-1].append(cand)
        else:
            groups.append([cand])
    return groups


# ---------------------------------------------------------------------------
# Pass 2: LLM confirmation
# ---------------------------------------------------------------------------

def _build_confirm_prompt(
    group: list[dict],
    all_segments: list[dict],
    *,
    window_pre: float,
    window_post: float,
    use_diarized: bool,
) -> str:
    """Build the user prompt for one confirmation call.

    The context window is [group_start - window_pre, group_end + window_post].
    Candidate segments are marked with >>> ... <<< in the transcript text.
    """
    g_start = group[0]["start"] - window_pre
    g_end = group[-1]["end"] + window_post
    ctx_segs = [s for s in all_segments if s["end"] >= g_start and s["start"] < g_end]

    candidate_set = {(s["start"], s["end"]) for s in group}
    fmt_fn = format_diarized_transcript if use_diarized else format_plain_transcript

    # Build a custom transcript string with >>> / <<< around candidates.
    # We format each segment individually and inject markers.
    lines = []
    for seg in ctx_segs:
        key = (seg["start"], seg["end"])
        text = seg.get("text", "").strip()
        ts = f"[{seg['start']:.1f}]"
        sp = seg.get("speaker", "")
        prefix = f"{sp}: " if sp else ""
        if key in candidate_set:
            lines.append(f"{ts} {prefix}>>> {text} <<<")
        else:
            lines.append(f"{ts} {prefix}{text}")

    transcript_block = "\n".join(lines)
    g_anchor_start = group[0]["start"]
    g_anchor_end = group[-1]["end"]

    return (
        f"Transcript excerpt ({g_start:.0f}s – {g_end:.0f}s):\n\n"
        f"{transcript_block}\n\n"
        f"Candidate segment(s) are at {g_anchor_start:.1f}s – {g_anchor_end:.1f}s "
        f"(marked with >>> <<<)."
    )


def confirm_candidate(
    gi: int,
    group: list[dict],
    all_segments: list[dict],
    *,
    ollama_url: str,
    model: str,
    temperature: float,
    window_pre: float,
    window_post: float,
    use_diarized: bool,
) -> tuple[int, list[dict], float, str]:
    """Run one LLM confirmation call.  Returns (gi, valid_pairs, elapsed, raw_response)."""
    prompt = _build_confirm_prompt(
        group, all_segments,
        window_pre=window_pre,
        window_post=window_post,
        use_diarized=use_diarized,
    )
    t0 = time.time()
    raw = call_ollama(
        ollama_url, model, prompt,
        system=_CONFIRM_SYSTEM,
        temperature=temperature,
        num_predict=256,
    )
    elapsed = time.time() - t0

    parsed = parse_pipe_qa(raw)
    valid, _ = _validate_qa_pairs(parsed)

    if not valid and raw.strip() and raw.strip().upper() != "NONE":
        # Retry once
        retry_prompt = prompt + _RETRY_REMINDER
        t0r = time.time()
        raw2 = call_ollama(
            ollama_url, model, retry_prompt,
            system=_CONFIRM_SYSTEM,
            temperature=temperature,
            num_predict=256,
        )
        elapsed += time.time() - t0r
        parsed2 = parse_pipe_qa(raw2)
        valid, _ = _validate_qa_pairs(parsed2)
        raw = raw2

    return gi, valid, elapsed, raw.strip()


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def run_extraction(
    segments: list[dict],
    ollama_url: str,
    model: str,
    temperature: float,
    *,
    window_pre: float = DEFAULT_WINDOW_PRE,
    window_post: float = DEFAULT_WINDOW_POST,
    group_gap: float = DEFAULT_GROUP_GAP,
    num_workers: int = 1,
    verbose: bool = False,
) -> tuple[list[dict] | None, float, int, int]:
    """Full two-pass extraction.

    Returns (qa_pairs | None, total_elapsed, num_candidates, num_confirmed).
    """
    if not segments:
        return None, 0.0, 0, 0

    use_diarized = any(s.get("speaker") for s in segments)

    # ---- Pass 1 ----
    candidates = find_candidates(segments)
    groups = group_candidates(candidates, gap_threshold=group_gap)
    print(f"  Pass 1: {len(candidates)} candidate segment(s) → "
          f"{len(groups)} group(s) to confirm")

    if not groups:
        return [], 0.0, 0, 0

    # ---- Pass 2 ----
    total_elapsed = 0.0
    all_pairs: list[dict] = []
    confirmed = 0

    def _call(gi: int, group: list[dict]) -> tuple[int, list[dict], float, str]:
        return confirm_candidate(
            gi, group, segments,
            ollama_url=ollama_url,
            model=model,
            temperature=temperature,
            window_pre=window_pre,
            window_post=window_post,
            use_diarized=use_diarized,
        )

    print(f"  Pass 2: confirming {len(groups)} group(s) "
          f"({num_workers} worker(s))...")

    indexed = list(enumerate(groups, 1))

    if num_workers > 1 and len(indexed) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_call, gi, grp): gi for gi, grp in indexed}
            results = sorted(
                (f.result() for f in as_completed(futures)),
                key=lambda r: r[0],
            )
    else:
        results = [_call(gi, grp) for gi, grp in indexed]

    for gi, pairs, elapsed, raw_resp in results:
        total_elapsed += elapsed
        grp = groups[gi - 1]
        g_start = grp[0]["start"]
        g_end = grp[-1]["end"]
        if pairs:
            confirmed += 1
            print(f"    Group {gi:3d}: [{g_start:.1f}s-{g_end:.1f}s] "
                  f"→ {len(pairs)} pair(s) ({elapsed:.1f}s)")
        elif verbose:
            cand_text = " / ".join(s.get("text", "").strip()[:50] for s in grp)
            print(f"    Group {gi:3d}: [{g_start:.1f}s-{g_end:.1f}s] "
                  f"→ NONE ({elapsed:.1f}s) [{cand_text[:70]}]")
        all_pairs.extend(pairs)

    print(f"  Pass 2: {confirmed}/{len(groups)} groups confirmed "
          f"in {total_elapsed:.1f}s total")

    if not all_pairs:
        return [], total_elapsed, len(candidates), confirmed

    qa_pairs = normalize_qa_pairs(all_pairs)
    return qa_pairs, total_elapsed, len(candidates), confirmed


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_transcript(
    transcript_path: Path,
    *,
    model: str,
    ollama_url: str,
    temperature: float,
    window_pre: float,
    window_post: float,
    group_gap: float,
    num_workers: int,
    preview: bool,
    verbose: bool = False,
) -> bool:
    """Extract Q&A from a single transcript.  Returns True on success."""
    print(f"\nTranscript: {transcript_path.name}")

    data = load_transcript(transcript_path)
    segments = data["segments"]

    MIN_TEXT_CHARS = 200
    total_text = sum(len(s.get("text", "")) for s in segments)
    if not segments or total_text < MIN_TEXT_CHARS:
        reason = "empty_transcript" if not segments else "insufficient_text"
        print(f"  Skipping — {reason}")
        _record_empty(transcript_path, model=model, reason=reason)
        return True

    # Two-pass extraction
    qa_pairs, elapsed, n_candidates, n_confirmed = run_extraction(
        segments, ollama_url, model, temperature,
        window_pre=window_pre,
        window_post=window_post,
        group_gap=group_gap,
        num_workers=num_workers,
        verbose=preview or verbose,
    )

    if qa_pairs is None:
        print("  FAILED to extract Q&A.")
        _record_empty(transcript_path, model=model, reason="extraction_failed")
        return False

    # Quality filters
    if qa_pairs:
        qa_pairs = apply_quality_filters(qa_pairs, segments, slice_fn=slice_segments)

    if not qa_pairs:
        print(f"  0 Q&A pairs after filtering (candidates={n_candidates}, "
              f"confirmed={n_confirmed})")
        _record_empty(transcript_path, model=model, reason="no_qa_found")
        return True

    print(f"\n  {len(qa_pairs)} Q&A pairs "
          f"(candidates={n_candidates}, confirmed={n_confirmed}, "
          f"elapsed={elapsed:.1f}s)")
    for i, qa in enumerate(qa_pairs, 1):
        answers = qa.get("answers", [])
        qs = qa.get("question_start", "?")
        qe = qa.get("question_end", "?")
        n_ans = len(answers)
        last_end = answers[-1].get("answer_end", "?") if answers else "?"
        print(f"  {i:2d}. [{qs:>7}s-{qe:>7}s]  {n_ans} answer(s), ends {last_end}s")

    if preview:
        _print_preview(qa_pairs, segments)

    coverage = compute_coverage(qa_pairs, segments)

    output = {
        "transcript_file": transcript_path.name,
        "model": model,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "extraction_method": "per_segment",
        "pass1_candidates": n_candidates,
        "pass2_confirmed": n_confirmed,
        "qa_pairs": qa_pairs,
    }
    if coverage.get("gaps"):
        output["coverage_gaps"] = coverage["gaps"]

    out_path = QA_DIR / (transcript_path.stem + ".qa.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {out_path}")
    return True



def _print_preview(qa_pairs: list[dict], segments: list[dict]) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {len(qa_pairs)} Q&A pairs")
    print(f"{'=' * 70}")
    for i, qa in enumerate(qa_pairs, 1):
        qs = qa.get("question_start", 0)
        qe = qa.get("question_end", 0)
        q_text = slice_segments(segments, qs, qe)[:150]
        print(f"\n  Q{i}. [{qs:.1f}s-{qe:.1f}s]:")
        print(f"       {q_text}...")
        for j, ans in enumerate(qa.get("answers", []), 1):
            a_s = ans.get("answer_start", 0)
            ae = ans.get("answer_end", 0)
            a_text = slice_segments(segments, a_s, ae)[:150]
            prefix = f"  A{j}" if len(qa["answers"]) > 1 else "   A"
            print(f"    {prefix}: [{a_s:.1f}s-{ae:.1f}s]:")
            print(f"       {a_text}...")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5b: Per-segment Q&A extraction"
    )
    parser.add_argument(
        "--transcript", type=Path, default=None,
        help="Process a single transcript JSON instead of all transcripts.",
    )
    parser.add_argument("--model", default=env_or_default("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    parser.add_argument("--ollama-url", default=env_or_default("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="Parallel Ollama workers for Pass 2 confirmation calls (default: 1).",
    )
    parser.add_argument(
        "--window-pre", type=float, default=DEFAULT_WINDOW_PRE, metavar="SEC",
        help=f"Seconds of context before each candidate (default: {DEFAULT_WINDOW_PRE}).",
    )
    parser.add_argument(
        "--window-post", type=float, default=DEFAULT_WINDOW_POST, metavar="SEC",
        help=f"Seconds of context after each candidate (default: {DEFAULT_WINDOW_POST}).",
    )
    parser.add_argument(
        "--group-gap", type=float, default=DEFAULT_GROUP_GAP, metavar="SEC",
        help=f"Max gap (s) between candidates to merge into one call (default: {DEFAULT_GROUP_GAP}).",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Print a human-readable preview with text sliced from transcript.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print NONE (rejected) groups in Pass 2 output.",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Re-extract even if .qa.json already exists.",
    )
    args = parser.parse_args()

    ensure_directories()

    print("=" * 70)
    print("Stage 5b: Per-Segment Q&A Extraction")
    print("=" * 70)
    print(f"  Model:        {args.model}")
    print(f"  Ollama URL:   {args.ollama_url}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Workers:      {args.workers}")
    print(f"  Window pre:   {args.window_pre}s")
    print(f"  Window post:  {args.window_post}s")
    print(f"  Group gap:    {args.group_gap}s")

    if args.transcript:
        if not args.transcript.exists():
            print(f"ERROR: File not found: {args.transcript}")
            sys.exit(1)
        transcripts = [args.transcript]
    else:
        transcripts = sorted(
            p for p in TRANSCRIPTS_DIR.glob("*.json")
            if ".qa" not in p.name
        )

    print(f"  Transcripts:  {len(transcripts)}")

    # Build skip sets from no-audio and non-English JSONL files
    non_english_stems: set[str] = {
        r["video_stem"] for r in load_jsonl(NON_ENGLISH_JSONL) if "video_stem" in r
    }
    no_audio_stems: set[str] = {
        r["video_stem"] for r in load_jsonl(NO_AUDIO_JSONL) if "video_stem" in r
    }
    skip_stems = non_english_stems | no_audio_stems

    before_skip = len(transcripts)
    transcripts = [t for t in transcripts if t.stem not in skip_stems]
    skipped_media = before_skip - len(transcripts)
    if skipped_media:
        print(f"  Skipped (no audio / non-English): {skipped_media}")

    # Filter already-processed unless --force
    force = args.force

    # Build set of stems already recorded as empty (no Q&A found / skipped)
    empty_records = load_jsonl(QA_EMPTY_JSONL)
    already_empty: set[str] = {
        Path(r["transcript_file"]).stem
        for r in empty_records
        if "transcript_file" in r
    }

    pending = []
    forced = 0
    for t in transcripts:
        qa_path = QA_DIR / (t.stem + ".qa.json")
        if qa_path.exists():
            # Has a real output file with pairs — only re-run if forced
            try:
                qa_data = json.loads(qa_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pending.append(t)
                continue
            if force:
                pending.append(t)
                forced += 1
        elif t.stem in already_empty:
            # Previously attempted but produced no pairs — skip unless forced
            if force:
                pending.append(t)
                forced += 1
        else:
            pending.append(t)

    skipped = len(transcripts) - len(pending)
    if skipped:
        print(f"  Already processed: {skipped} (use --force to redo)")
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
            window_pre=args.window_pre,
            window_post=args.window_post,
            group_gap=args.group_gap,
            num_workers=args.workers,
            preview=args.preview,
            verbose=args.verbose,
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
