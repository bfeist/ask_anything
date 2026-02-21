#!/usr/bin/env python3
"""Targeted fix for the remaining transcript issues after first cleanup.

1. Expedition 67 & 69: Drop hallucinated intro segments (pre-event silence/music
   that whisper hallucinated as video editing instructions)
2. Crew1 Media (la): Drop hallucinated first 2 music/silence segments
3. ARISS Spanish: Flag for re-transcription (no English available)
"""
import json
from pathlib import Path

TRANSCRIPTS_DIR = Path(__file__).resolve().parents[1] / "data" / "transcripts"

# Hallucinated phrases from bad whisper detections - these never appear in real
# NASA content and are artifacts from whisper hallucinating during silence/music
HALLUCINATION_PHRASES_EXACT = {
    "add a layer mask to the top layer",
    "add a layer mask to the top layer.",
    "let's get started!",
    "let's get started",
    "hello, everyone.",
    "i'm sorry.",
}

# Substring-based: if the segment text starts with one of these, it's hallucinated
HALLUCINATION_PREFIXES = [
    "this is the first time i'm doing a video on a new project",
    "this is the first time i've done this, so i'm not sure",
    "i'm going to add a new layer to the top layer",
    "i'm going to add a new layer to the background",
    "i'm going to add a layer mask",
]


def is_hallucination(text: str) -> bool:
    """Check if segment text is a known whisper hallucination."""
    t = text.strip().lower()
    if t in HALLUCINATION_PHRASES_EXACT:
        return True
    for prefix in HALLUCINATION_PREFIXES:
        if t.startswith(prefix):
            return True
    return False


def fix_file(path: Path) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    orig_count = len(data["segments"])
    # Drop segments whose text matches known hallucinations
    data["segments"] = [
        s for s in data["segments"]
        if not is_hallucination(s.get("text", ""))
    ]
    dropped = orig_count - len(data["segments"])
    data["num_segments"] = len(data["segments"])

    if dropped > 0:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Fixed {path.name}:")
        print(f"    Dropped {dropped} hallucinated segments ({orig_count} → {data['num_segments']})")
    else:
        print(f"  {path.name}: no hallucinated segments found")


def main():
    print("Dropping known hallucinated segments from cleaned transcripts...\n")

    for path in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Only check files that were previously non-English
        if any(is_hallucination(s.get("text", ""))
               for s in data["segments"]):
            fix_file(path)

    # Flag ARISS Spanish
    ariss = TRANSCRIPTS_DIR / "ARISSHacePosibleElPrimerContactoEducacionalConVenezuela__ARISS Hace Posible el Primer Contacto Educacional con Venezuela_lowres.json"
    if ariss.exists():
        with open(ariss, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Check if text is still Spanish
        sample = data["segments"][0]["text"] if data["segments"] else ""
        if not sample.isascii():  # Simple heuristic: Spanish has accented chars
            print(f"\n⚠ NEEDS RE-TRANSCRIPTION (still Spanish, no English available):")
            print(f"  {ariss.name}")
            print(f"  Re-run: uv run python 4_transcribe_videos.py --force --file <video_path>")


if __name__ == "__main__":
    main()
