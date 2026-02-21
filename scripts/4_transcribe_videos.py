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
DEFAULT_LANGUAGE = "auto"  # "auto" = detect from audio; or force e.g. "en", "ru"


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
        if seg.get("text_en"):
            clean_seg["text_en"] = seg["text_en"].strip()
        if "language" in seg:
            clean_seg["language"] = seg["language"]
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
        "language": meta.get("detected_language", "en"),
        "translated": meta.get("translated", False),
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

def _merge_translations(
    original_segments: list[dict],
    translated_segments: list[dict],
    detected_lang: str,
) -> None:
    """Add ``text_en`` and ``language`` to original segments by timestamp overlap."""
    for orig in original_segments:
        orig["language"] = detected_lang
        # Find translated segments that overlap with this original segment
        overlapping = [
            t for t in translated_segments
            if t["end"] > orig["start"] and t["start"] < orig["end"]
        ]
        if overlapping:
            eng_text = " ".join(t.get("text", "").strip() for t in overlapping)
            if eng_text:
                orig["text_en"] = eng_text


def transcribe_video(
    video_path: Path,
    *,
    model_name: str = DEFAULT_WHISPER_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    language: str = DEFAULT_LANGUAGE,
    hf_token: str | None = None,
    diarize_model_name: str = DEFAULT_DIARIZE_MODEL,
    device: str = "cuda",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> Path:
    """Run WhisperX transcription + alignment + optional diarization on a video.

    When *language* is ``"auto"`` the language is detected from the audio.
    If the detected language is not English a second translation pass is run
    to produce ``text_en`` fields for each segment.

    Returns the path to the saved transcript JSON.
    """
    t0 = time.time()
    print(f"\n{'─' * 70}")
    print(f"Transcribing: {video_path.name}")
    print(f"  Path:      {video_path}")
    print(f"  Model:     {model_name}  Compute: {compute_type}  Device: {device}")
    print(f"  Language:  {language}")

    auto_detect = language == "auto"

    # 1. Load audio
    print("  Loading audio …")
    audio = whisperx.load_audio(str(video_path))

    # 2. Transcribe (original language)
    print("  Running whisper transcription …")
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        **({}  if auto_detect else {"language": language}),
    )
    transcribe_kwargs: dict = {"batch_size": batch_size}
    if not auto_detect:
        transcribe_kwargs["language"] = language
    result = model.transcribe(audio, **transcribe_kwargs)
    detected_lang = result.get("language", language if not auto_detect else "en")
    num_raw = len(result.get("segments", []))
    print(f"  Detected language: {detected_lang}")
    print(f"  Raw segments: {num_raw}")

    # 3. Translation pass (if non-English and we want English translations)
    translated_result = None
    if detected_lang != "en":
        print(f"  Running English translation pass (detected {detected_lang})…")
        translated_result = model.transcribe(
            audio, batch_size=batch_size, task="translate",
        )
        print(f"  Translation segments: {len(translated_result.get('segments', []))}")

    # Free model VRAM before alignment
    del model
    torch.cuda.empty_cache()

    # 4. Align original text
    lang_for_align = detected_lang if auto_detect else language
    print(f"  Aligning timestamps ({lang_for_align})…")
    try:
        align_model, align_meta = whisperx.load_align_model(
            language_code=lang_for_align, device=device
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
    except Exception as exc:
        print(f"  ⚠ Alignment model not available for '{lang_for_align}': {exc}")
        print("  Continuing with unaligned timestamps.")

    # 5. Align translated text (if we have a translation pass)
    if translated_result is not None:
        print("  Aligning English translation…")
        try:
            align_model_en, align_meta_en = whisperx.load_align_model(
                language_code="en", device=device
            )
            translated_result = whisperx.align(
                translated_result["segments"],
                align_model_en,
                align_meta_en,
                audio,
                device,
                return_char_alignments=False,
            )
            del align_model_en
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  ⚠ English alignment failed: {exc}")
            print("  Translations will use unaligned timestamps.")

    # 6. Diarization (optional)
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

    # 7. Merge translations onto main segments
    translated = False
    if translated_result is not None:
        print("  Merging English translations into segments…")
        trans_segs = translated_result.get("segments", translated_result if isinstance(translated_result, list) else [])
        if isinstance(translated_result, dict) and "segments" in translated_result:
            trans_segs = translated_result["segments"]
        main_segs = result.get("segments", result if isinstance(result, list) else [])
        if isinstance(result, dict) and "segments" in result:
            main_segs = result["segments"]
        _merge_translations(main_segs, trans_segs, detected_lang)
        translated = True

    elapsed = time.time() - t0

    # 8. Save
    meta = {
        "model": model_name,
        "compute_type": compute_type,
        "diarization": diarized,
        "detected_language": detected_lang,
        "translated": translated,
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
        "--language", default=DEFAULT_LANGUAGE,
        help=(
            f"Language code for transcription (default: {DEFAULT_LANGUAGE}). "
            "Use 'auto' for automatic detection, or an ISO 639-1 code like 'en', 'ru', 'ja'."
        ),
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
    print(f"  Language:     {args.language}")
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
                language=args.language,
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
