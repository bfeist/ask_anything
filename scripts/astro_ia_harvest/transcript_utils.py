"""Shared transcript loading, formatting, and analysis utilities."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def load_transcript(path: Path) -> dict:
    """Load a transcript JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def seg_text(seg: dict, prefer_english: bool = True) -> str:
    """Return the best available text for a segment.

    When *prefer_english* is True (default), returns the English translation
    ``text_en`` if present, otherwise falls back to ``text``.
    """
    if prefer_english and seg.get("text_en"):
        return seg["text_en"].strip()
    return (seg.get("text") or "").strip()


def format_plain_transcript(
    segments: list[dict],
    *,
    max_time: float | None = None,
    prefer_english: bool = True,
) -> str:
    """Timestamped text only — no speaker labels."""
    lines: list[str] = []
    for seg in segments:
        if max_time is not None and seg["start"] >= max_time:
            break
        ts = f"[{seg['start']:.1f}s]"
        lines.append(f"{ts} {seg_text(seg, prefer_english)}")
    return "\n".join(lines)


def format_diarized_transcript(
    segments: list[dict],
    *,
    max_time: float | None = None,
    prefer_english: bool = True,
) -> str:
    """Timestamped text with speaker labels."""
    lines: list[str] = []
    for seg in segments:
        if max_time is not None and seg["start"] >= max_time:
            break
        ts = f"[{seg['start']:.1f}s]"
        sp = seg.get("speaker", "?")
        lines.append(f"{ts} {sp}: {seg_text(seg, prefer_english)}")
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


def slice_segments(
    segments: list[dict],
    start: float,
    end: float,
    prefer_english: bool = True,
) -> str:
    """Return concatenated text from all segments overlapping [start, end].

    Uses word-level timestamps when available for precise boundary slicing.
    Each word is included at most once based on its midpoint falling within
    the [start, end) window, preventing text duplication at boundaries.
    Falls back to segment-level slicing if word timestamps are unavailable.
    """
    parts: list[str] = []
    for seg in segments:
        # Quick skip: segment doesn't overlap window at all
        if seg["end"] <= start or seg["start"] >= end:
            continue
        words = seg.get("words")
        if words:
            # Word-level precision: include a word if its midpoint is in [start, end)
            for w in words:
                w_start = w.get("start")
                w_end = w.get("end")
                if w_start is None or w_end is None:
                    continue
                midpoint = (w_start + w_end) / 2
                if start <= midpoint < end:
                    parts.append(w.get("word", ""))
        else:
            # Fallback: segment-level with midpoint check
            midpoint = (seg["start"] + seg["end"]) / 2
            if start <= midpoint < end:
                parts.append(seg_text(seg, prefer_english))
    return " ".join(parts)
