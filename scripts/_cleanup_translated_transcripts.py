#!/usr/bin/env python3
"""One-time cleanup: convert existing translated transcripts to English-only.

For transcripts where Whisper misdetected the language:
- If text_en exists and text is garbage → use text_en as text
- If text is already English → just drop text_en
- Drop per-segment 'language' fields
- Set top-level language to "en", remove 'translated' flag

Transcripts with hallucinated garbage in BOTH text and text_en are flagged
for re-transcription.
"""
import json
import re
from pathlib import Path

TRANSCRIPTS_DIR = Path(__file__).resolve().parents[1] / "data" / "transcripts"

# Known hallucination patterns from bad language detection
GARBAGE_PATTERNS = [
    re.compile(r"^Teksting av", re.IGNORECASE),
    re.compile(r"^Undertekster av", re.IGNORECASE),
    re.compile(r"^Musik\s+Musik", re.IGNORECASE),
    re.compile(r"^🎵"),
    re.compile(r"^\.\.\.\s*\.\.\.\s*\.\.\."),  # "... ... ..."
]


def is_garbage(text: str) -> bool:
    """Check if a segment's text looks like hallucinated junk."""
    text = text.strip()
    if not text:
        return True
    for pat in GARBAGE_PATTERNS:
        if pat.match(text):
            return True
    return False


def clean_transcript(path: Path) -> str | None:
    """Clean a transcript file in-place. Returns status message or None if skipped."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("translated", False) and data.get("language", "en") == "en":
        return None  # Already clean

    orig_lang = data.get("language", "?")
    segments = data.get("segments", [])
    changes = 0
    garbage_dropped = 0

    for seg in segments:
        text = seg.get("text", "").strip()
        text_en = seg.get("text_en", "").strip()

        if text_en:
            # Always prefer the English translation
            seg["text"] = text_en
            if text_en != text:
                changes += 1
        
        if is_garbage(seg.get("text", "").strip()):
            # Even after potential swap, if text is still junk → drop
            seg["_drop"] = True
            garbage_dropped += 1

        # Remove multi-language fields
        seg.pop("text_en", None)
        seg.pop("language", None)

    # Remove marked segments
    data["segments"] = [s for s in segments if not s.get("_drop")]
    for s in data["segments"]:
        s.pop("_drop", None)

    # Update top-level metadata
    data["language"] = "en"
    data.pop("translated", None)
    data["num_segments"] = len(data["segments"])

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return (
        f"  {path.name}\n"
        f"    was: {orig_lang} | swapped: {changes} | garbage dropped: {garbage_dropped} "
        f"| final segs: {len(data['segments'])}"
    )


def main():
    print("Cleaning translated transcripts to English-only…\n")

    cleaned = 0
    needs_retranscribe = []

    for path in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        result = clean_transcript(path)
        if result:
            print(result)
            cleaned += 1

            # Check if remaining content is mostly empty after cleanup
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            total = len(data["segments"])
            non_empty = sum(1 for s in data["segments"] if s.get("text", "").strip())
            if total < 5 or non_empty < total * 0.5:
                needs_retranscribe.append(path.name)

    print(f"\nCleaned: {cleaned} transcripts")

    if needs_retranscribe:
        print(f"\n⚠ These transcripts have mostly garbage and need re-transcription:")
        for name in needs_retranscribe:
            print(f"  - {name}")
        print(f"\nRe-transcribe with: uv run python 4_transcribe_videos.py --force --file <video_path>")


if __name__ == "__main__":
    main()
