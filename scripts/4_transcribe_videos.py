#!/usr/bin/env python3
"""Step 4: Transcribe downloaded videos using WhisperX with optional diarization.

Scans both the primary download directory (D:\\ask_anything_ia_videos_raw) and the
legacy directory (D:\\ISSiRT_ia_videos_raw) for .mp4 files.  Files that already
have a matching transcript JSON in data/transcripts/ are skipped.

Output per video:  data/transcripts/<video_stem>.json
Each JSON contains full WhisperX segment data with aligned timestamps and,
when a HuggingFace token is available, speaker diarization labels.

Requirements:
  - whisperx (pip install whisperx)
  - PyTorch with CUDA support
  - For diarization: HF_TOKEN env var with accepted terms for
    pyannote/segmentation-3.0 and pyannote/speaker-diarization-3.1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

# Monkey-patch: lightning_fabric passes weights_only=None to torch.load, which
# PyTorch >=2.6 treats as True (strict mode).  pyannote checkpoints contain
# custom objects (e.g. TorchVersion) that strict mode rejects.  Override the
# default to False so pyannote model weights load correctly.
import lightning_fabric.utilities.cloud_io as _lf_cloud_io

_lf_original_load = _lf_cloud_io._load


def _lf_patched_load(path_or_url, map_location=None, weights_only=None):
    if weights_only is None:
        weights_only = False
    return _lf_original_load(path_or_url, map_location=map_location, weights_only=weights_only)


_lf_cloud_io._load = _lf_patched_load

import whisperx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from astro_ia_harvest.config import (  # noqa: E402
    DOWNLOAD_DIR,
    EXISTING_DOWNLOAD_DIR,
    TRANSCRIPTS_DIR,
    TRANSCRIPT_LOG_JSONL,
    ensure_directories,
)
from astro_ia_harvest.jsonl_utils import append_jsonl  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_BATCH_SIZE = 16  # reduce if VRAM is tight
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-3.1"
LANGUAGE = "en"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def transcript_stem(video_path: Path) -> str:
    """Derive a stable transcript filename stem from a video path."""
    return video_path.stem


def find_videos(*dirs: Path) -> list[Path]:
    """Collect all .mp4 files from the given directories."""
    videos: list[Path] = []
    for d in dirs:
        if d.exists():
            videos.extend(sorted(d.rglob("*.mp4")))
    return videos


def already_transcribed(video_path: Path) -> bool:
    """Return True if a transcript JSON already exists for this video."""
    out = TRANSCRIPTS_DIR / f"{transcript_stem(video_path)}.json"
    return out.exists()


def load_transcribed_set() -> set[str]:
    """Return set of video stems that already have transcripts."""
    if not TRANSCRIPTS_DIR.exists():
        return set()
    return {p.stem for p in TRANSCRIPTS_DIR.glob("*.json")}


def save_transcript(video_path: Path, result: dict, meta: dict) -> Path:
    """Write the WhisperX result to a JSON file and return the path."""
    out_path = TRANSCRIPTS_DIR / f"{transcript_stem(video_path)}.json"

    # Build clean serializable segments
    segments = []
    for seg in result.get("segments", []):
        clean_seg: dict = {
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg.get("text", "").strip(),
        }
        if "speaker" in seg:
            clean_seg["speaker"] = seg["speaker"]
        # Keep word-level timestamps if present (useful for precise seeking)
        if "words" in seg:
            clean_seg["words"] = [
                {
                    "word": w.get("word", ""),
                    "start": round(w["start"], 3) if "start" in w else None,
                    "end": round(w["end"], 3) if "end" in w else None,
                    "score": round(w.get("score", 0), 3),
                    **({"speaker": w["speaker"]} if "speaker" in w else {}),
                }
                for w in seg["words"]
            ]
        segments.append(clean_seg)

    payload = {
        "video_file": video_path.name,
        "video_path": str(video_path),
        "model": meta["model"],
        "language": LANGUAGE,
        "diarization": meta["diarization"],
        "compute_type": meta["compute_type"],
        "transcribed_at": datetime.now(timezone.utc).isoformat(),
        "processing_time_s": round(meta["processing_time_s"], 2),
        "num_segments": len(segments),
        "segments": segments,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


# ---------------------------------------------------------------------------
# Core transcription
# ---------------------------------------------------------------------------

def transcribe_video(
    video_path: Path,
    *,
    model_name: str = DEFAULT_WHISPER_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    hf_token: str | None = None,
    diarize_model_name: str = DEFAULT_DIARIZE_MODEL,
    device: str = "cuda",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> Path:
    """Run WhisperX transcription + alignment + optional diarization on a video.

    Returns the path to the saved transcript JSON.
    """
    t0 = time.time()
    print(f"\n{'─' * 70}")
    print(f"Transcribing: {video_path.name}")
    print(f"  Path:      {video_path}")
    print(f"  Model:     {model_name}  Compute: {compute_type}  Device: {device}")

    # 1. Load audio
    print("  Loading audio …")
    audio = whisperx.load_audio(str(video_path))

    # 2. Transcribe
    print("  Running whisper transcription …")
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=LANGUAGE,
    )
    result = model.transcribe(audio, batch_size=batch_size, language=LANGUAGE)
    num_raw = len(result.get("segments", []))
    print(f"  Raw segments: {num_raw}")

    # Free model VRAM before alignment
    del model
    torch.cuda.empty_cache()

    # 3. Align (for accurate per-word timestamps)
    print("  Aligning timestamps …")
    align_model, align_meta = whisperx.load_align_model(
        language_code=LANGUAGE, device=device
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        align_meta,
        audio,
        device,
        return_char_alignments=False,
    )
    del align_model
    torch.cuda.empty_cache()

    # 4. Diarization (optional)
    diarized = False
    if hf_token:
        print("  Running speaker diarization …")
        try:
            diarize_model = whisperx.diarize.DiarizationPipeline(
                model_name=diarize_model_name, use_auth_token=hf_token, device=device
            )
            diarize_kwargs: dict = {}
            if min_speakers is not None:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_kwargs["max_speakers"] = max_speakers
            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            diarized = True
            del diarize_model
            torch.cuda.empty_cache()
            # Count unique speakers
            speakers = {s.get("speaker") for s in result.get("segments", []) if s.get("speaker")}
            print(f"  Speakers detected: {len(speakers)} — {sorted(speakers)}")
        except Exception as exc:
            print(f"  ⚠ Diarization failed: {exc}")
            print("  Continuing without speaker labels.")
    else:
        print("  Diarization skipped (no HF_TOKEN).")

    elapsed = time.time() - t0

    # 5. Save
    meta = {
        "model": model_name,
        "compute_type": compute_type,
        "diarization": diarized,
        "processing_time_s": elapsed,
    }
    out_path = save_transcript(video_path, result, meta)

    num_final = len(result.get("segments", []))
    print(f"  Final segments: {num_final}")
    print(f"  Saved: {out_path}")
    print(f"  Time: {elapsed:.1f}s")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe downloaded videos using WhisperX"
    )
    parser.add_argument(
        "--model", default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for whisper inference (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--compute-type", default=DEFAULT_COMPUTE_TYPE,
        help=f"Compute type for CTranslate2 (default: {DEFAULT_COMPUTE_TYPE})",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max videos to transcribe (0 = all)",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Transcribe a single specific video file (full path)",
    )
    parser.add_argument(
        "--min-speakers", type=int, default=None,
        help="Minimum number of speakers for diarization",
    )
    parser.add_argument(
        "--max-speakers", type=int, default=None,
        help="Maximum number of speakers for diarization",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-transcribe even if transcript already exists",
    )
    parser.add_argument(
        "--diarize-model", default=DEFAULT_DIARIZE_MODEL,
        help=f"Pyannote diarization model (default: {DEFAULT_DIARIZE_MODEL})",
    )
    parser.add_argument(
        "--no-diarize", action="store_true",
        help="Skip diarization even if HF_TOKEN is set",
    )
    args = parser.parse_args()

    ensure_directories()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠ WARNING: No CUDA GPU detected. Transcription will be very slow.")

    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    if args.no_diarize:
        hf_token = None

    print("=" * 70)
    print("Step 4: Video Transcription (WhisperX)")
    print("=" * 70)
    print(f"  Device:       {device}")
    print(f"  Model:        {args.model}")
    print(f"  Compute type: {args.compute_type}")
    print(f"  Diarization:  {'yes — ' + args.diarize_model if hf_token else 'no (HF_TOKEN not set)'}")
    print(f"  Transcripts:  {TRANSCRIPTS_DIR}")

    # Collect videos to process
    if args.file:
        target = Path(args.file)
        if not target.exists():
            print(f"ERROR: File not found: {target}")
            sys.exit(1)
        videos = [target]
    else:
        videos = find_videos(DOWNLOAD_DIR, EXISTING_DOWNLOAD_DIR)

    print(f"  Videos found: {len(videos)}")

    # Filter already-transcribed
    if not args.force:
        videos = [v for v in videos if not already_transcribed(v)]

    if args.limit > 0:
        videos = videos[: args.limit]

    print(f"  To transcribe: {len(videos)}")

    if not videos:
        print("\nNothing to transcribe. All videos already have transcripts.")
        return

    # Process
    successes = 0
    failures = 0
    total_start = time.time()

    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}]")
        try:
            out_path = transcribe_video(
                video,
                model_name=args.model,
                batch_size=args.batch_size,
                compute_type=args.compute_type,
                hf_token=hf_token,
                diarize_model_name=args.diarize_model,
                device=device,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
            # Log success
            append_jsonl(TRANSCRIPT_LOG_JSONL, {
                "video_file": video.name,
                "video_path": str(video),
                "transcript_path": str(out_path),
                "status": "ok",
                "ts": int(time.time()),
            })
            successes += 1
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")
            append_jsonl(TRANSCRIPT_LOG_JSONL, {
                "video_file": video.name,
                "video_path": str(video),
                "status": "error",
                "error": str(exc),
                "ts": int(time.time()),
            })
            failures += 1

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Transcription complete: {successes} ok, {failures} failed in {total_time:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
