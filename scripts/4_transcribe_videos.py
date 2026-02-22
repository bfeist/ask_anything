#!/usr/bin/env python3
"""Step 4: Transcribe downloaded videos using WhisperX with optional diarization.

Scans both the primary download directory (D:\\ask_anything_ia_videos_raw) and the
legacy directory (D:\\ISSiRT_ia_videos_raw) for .mp4 files.  Files that already
have a matching transcript JSON in data/transcripts/ are skipped.

Before transcription, a multi-segment language detection pass samples audio
from several points throughout the file (not just the first 30 seconds) to
reliably determine whether the content is English.  Non-English videos are
logged to data/non_english_videos.jsonl and skipped.  Subsequent runs
automatically skip videos already in that file.

English audio is transcribed normally; alignment uses the English wav2vec2
model so phonemes match correctly.

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
import logging
import os
import subprocess
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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

# Suppress noisy loggers
logging.getLogger("whisperx.alignment").setLevel(logging.ERROR)
logging.getLogger("whisperx.asr").setLevel(logging.WARNING)
logging.getLogger("whisperx.vads.pyannote").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning_fabric").setLevel(logging.WARNING)

# Suppress version-mismatch and deprecation warnings from pyannote / torchaudio
warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Model was trained with.*")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
warnings.filterwarnings("ignore", message=".*Bad things might happen.*")
warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from astro_ia_harvest.config import (  # noqa: E402
    DOWNLOAD_DIR,
    NO_AUDIO_JSONL,
    NON_ENGLISH_JSONL,
    TRANSCRIPTS_DIR,
    TRANSCRIPT_LOG_JSONL,
    ensure_directories,
)
from astro_ia_harvest.jsonl_utils import append_jsonl, load_jsonl  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_BATCH_SIZE = 16  # reduce if VRAM is tight
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-3.1"
LANG_DETECT_SAMPLE_SECONDS = 30  # duration of each sample window
LANG_DETECT_NUM_SAMPLES = 5     # number of windows to sample
LANG_DETECT_ENGLISH_THRESHOLD = 0.5  # fraction of samples that must be English


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NonEnglishError(Exception):
    """Raised when language detection determines a video is not English."""

    def __init__(self, video_path: Path, detected_lang: str, details: dict, elapsed: float):
        self.video_path = video_path
        self.detected_lang = detected_lang
        self.details = details
        self.elapsed = elapsed
        super().__init__(
            f"Non-English audio detected: {detected_lang} "
            f"(English in {details['english_fraction']:.0%} of samples)"
        )


class NoAudioError(Exception):
    """Raised when a video file has no audio stream."""

    def __init__(self, video_path: Path):
        self.video_path = video_path
        super().__init__(f"No audio stream found in {video_path.name}")


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


def load_non_english_set() -> set[str]:
    """Return set of video stems previously flagged as non-English."""
    records = load_jsonl(NON_ENGLISH_JSONL)
    return {r["video_stem"] for r in records if "video_stem" in r}


def load_no_audio_set() -> set[str]:
    """Return set of video stems previously flagged as having no audio stream."""
    records = load_jsonl(NO_AUDIO_JSONL)
    return {r["video_stem"] for r in records if "video_stem" in r}


def probe_has_audio(video_path: Path) -> bool:
    """Return True if the file has at least one audio stream, False otherwise.

    Uses ffprobe to inspect the container without decoding any media.
    Falls back to True on probe errors so transcription still attempts the file.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return bool(result.stdout.strip())
    except Exception:
        return True  # cannot probe — let whisperx try


def detect_language_multi_sample(
    audio: np.ndarray,
    model,
    *,
    sample_rate: int = 16_000,
    num_samples: int = LANG_DETECT_NUM_SAMPLES,
    sample_seconds: int = LANG_DETECT_SAMPLE_SECONDS,
) -> tuple[str, dict]:
    """Detect language by sampling multiple windows from the middle third.

    NASA videos typically have an English intro (slate, host remarks) and
    sometimes an English outro.  The actual interview content lives in the
    middle.  To avoid being fooled by English bookends, we sample *only*
    from the middle third of the audio.

    Returns (language_code, details_dict) where *details_dict* contains
    per-sample results and the vote tally for logging.
    """
    total_samples = len(audio)
    window_len = sample_seconds * sample_rate

    # Use only the middle third of the audio to avoid English intros/outros
    third = total_samples // 3
    usable_start = third
    usable_end = 2 * third
    duration_sec = round(total_samples / sample_rate, 1)
    print(f"    Audio duration: {duration_sec}s — sampling middle third "
          f"({round(usable_start / sample_rate, 1)}s – "
          f"{round(usable_end / sample_rate, 1)}s)")

    if usable_end - usable_start < window_len:
        # Very short audio — just use the whole thing as one sample
        offsets = [max(0, (total_samples - window_len) // 2)]
    else:
        usable_range = usable_end - usable_start - window_len
        step = usable_range // max(num_samples - 1, 1)
        offsets = [usable_start + step * i for i in range(num_samples)]

    sample_results = []
    for idx, offset in enumerate(offsets):
        chunk = audio[offset : offset + window_len]
        # Use faster-whisper's language detection via the model
        try:
            segments, info = model.model.transcribe(
                chunk, task="transcribe", without_timestamps=True,
                # Only detect language, don't actually decode much
            )
            # Consume the generator to get info populated
            _ = next(segments, None)
            lang = info.language
            prob = info.language_probability
        except Exception:
            lang = "unknown"
            prob = 0.0
        offset_sec = round(offset / sample_rate, 1)
        sample_results.append({
            "offset_s": offset_sec,
            "language": lang,
            "probability": round(prob, 3),
        })
        print(f"    Sample {idx + 1}/{len(offsets)} @ {offset_sec}s → {lang} ({prob:.1%})")

    # Majority vote
    lang_counts = Counter(s["language"] for s in sample_results)
    majority_lang, majority_count = lang_counts.most_common(1)[0]
    en_count = lang_counts.get("en", 0)
    en_fraction = en_count / len(sample_results)

    details = {
        "samples": sample_results,
        "vote_counts": dict(lang_counts),
        "majority_language": majority_lang,
        "english_fraction": round(en_fraction, 3),
    }
    return majority_lang, details


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
        "language": "en",
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
    stream: bool = False,
    video_num: int | None = None,
    video_total: int | None = None,
) -> Path:
    """Run language detection + WhisperX transcription + alignment + diarization.

    First runs multi-segment language detection.  If the audio is determined
    to be non-English, raises ``NonEnglishError`` so the caller can log it
    and skip transcription.

    Returns the path to the saved transcript JSON.
    """
    t0 = time.time()
    counter = f"[{video_num}/{video_total}] " if video_num is not None and video_total is not None else ""
    print(f"\n{'─' * 70}")
    print(f"{counter}Transcribing: {video_path.name}")
    print(f"  Path:      {video_path}")
    print(f"  Model:     {model_name}  Compute: {compute_type}  Device: {device}")

    # 0. Pre-flight: verify audio stream exists before loading GPU resources
    print("  Probing audio streams …")
    if not probe_has_audio(video_path):
        raise NoAudioError(video_path)

    # 1. Load audio
    print("  Loading audio …")
    audio = whisperx.load_audio(str(video_path))

    # 2. Multi-segment language detection
    print("  Running multi-segment language detection …")
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
    )
    detected_lang, lang_details = detect_language_multi_sample(audio, model)
    en_fraction = lang_details["english_fraction"]
    print(f"  Language detection result: {detected_lang} "
          f"(English in {en_fraction:.0%} of samples)")

    if en_fraction < LANG_DETECT_ENGLISH_THRESHOLD:
        # Non-English — free resources and signal to caller
        del model
        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        raise NonEnglishError(
            video_path=video_path,
            detected_lang=detected_lang,
            details=lang_details,
            elapsed=elapsed,
        )

    # 3. Transcribe (English)
    # Explicitly pass language="en" to suppress WhisperX's internal first-30s
    # language detection, which is unreliable (e.g. it can misdetect English
    # intros as Norwegian nn, Latin, etc.).  We've already confirmed English
    # via multi-segment detection above.
    if stream:
        print("  Running whisper transcribe (language=en forced, streaming) …")
        print("  " + "─" * 66)
        sys.stdout.flush()
        # Use the underlying faster-whisper generator so each segment is
        # printed to the terminal as soon as it is decoded.
        segments_gen, _ = model.model.transcribe(
            audio,
            language="en",
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
        )
        raw_segments: list[dict] = []
        for seg in segments_gen:
            words = [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "score": w.probability,
                }
                for w in (seg.words or [])
            ]
            raw_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": words,
            })
            print(
                f"  [{seg.start:7.1f}s → {seg.end:7.1f}s]  {seg.text.strip()}",
                flush=True,
            )
        result = {"segments": raw_segments, "language": "en"}
        print("  " + "─" * 66)
    else:
        print("  Running whisper transcribe (language=en forced, batched) …")
        result = model.transcribe(audio, batch_size=batch_size, language="en")
    num_raw = len(result.get("segments", []))
    print(f"  Raw segments: {num_raw}")

    # Free model VRAM before alignment
    del model
    torch.cuda.empty_cache()

    # 4. Align timestamps (English audio → English text, phonemes match)
    print("  Aligning timestamps (en)…")
    try:
        align_model, align_meta = whisperx.load_align_model(
            language_code="en", device=device
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
        print(f"  ⚠ English alignment failed: {exc}")
        print("  Continuing with unaligned timestamps.")

    # 5. Diarization (optional)
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

    # 6. Save
    meta = {
        "model": model_name,
        "compute_type": compute_type,
        "diarization": diarized,
        "detected_language": "en",
        "lang_detection_details": lang_details,
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
        "--force-lang-check", action="store_true",
        help="Re-run language detection on videos previously flagged as non-English",
    )
    parser.add_argument(
        "--diarize-model", default=DEFAULT_DIARIZE_MODEL,
        help=f"Pyannote diarization model (default: {DEFAULT_DIARIZE_MODEL})",
    )
    parser.add_argument(
        "--no-diarize", action="store_true",
        help="Skip diarization even if HF_TOKEN is set",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream transcription output segment-by-segment (slower but shows live progress)",
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
    print(f"  Output lang:  English (non-English videos skipped)")
    print(f"  Diarization:  {'yes \u2014 ' + args.diarize_model if hf_token else 'no (HF_TOKEN not set)'}")
    print(f"  Transcripts:  {TRANSCRIPTS_DIR}")
    print(f"  Non-English:  {NON_ENGLISH_JSONL}")
    print(f"  Streaming:    {'yes' if args.stream else 'no (batched)'}")

    # Collect videos to process
    if args.file:
        target = Path(args.file)
        if not target.exists():
            print(f"ERROR: File not found: {target}")
            sys.exit(1)
        videos = [target]
    else:
        videos = find_videos(DOWNLOAD_DIR)

    print(f"  Videos found: {len(videos)}")

    # Filter already-transcribed
    if not args.force:
        before = len(videos)
        videos = [v for v in videos if not already_transcribed(v)]
        skipped_transcribed = before - len(videos)
        if skipped_transcribed:
            print(f"  Skipped (already transcribed): {skipped_transcribed}")

    # Filter videos previously flagged as non-English
    if not args.force_lang_check:
        non_english_stems = load_non_english_set()
        before = len(videos)
        videos = [v for v in videos if v.stem not in non_english_stems]
        skipped_lang = before - len(videos)
        if skipped_lang:
            print(f"  Skipped (non-English): {skipped_lang}")
    else:
        non_english_stems = set()

    # Filter videos previously flagged as having no audio
    no_audio_stems = load_no_audio_set()
    before = len(videos)
    videos = [v for v in videos if v.stem not in no_audio_stems]
    skipped_no_audio = before - len(videos)
    if skipped_no_audio:
        print(f"  Skipped (no audio stream): {skipped_no_audio}")

    if args.limit > 0:
        videos = videos[: args.limit]

    print(f"  To transcribe: {len(videos)}")

    if not videos:
        print("\nNothing to transcribe. All videos already have transcripts.")
        return

    # Process
    # Use a live queue so that newly downloaded videos discovered between
    # transcriptions are automatically picked up and appended.
    successes = 0
    failures = 0
    non_english_count = 0
    no_audio_count = 0
    total_start = time.time()
    queue: list[Path] = list(videos)
    initial_queue_size = len(queue)
    seen_videos: set[Path] = set(queue)  # track every path ever enqueued

    i = 0
    while i < len(queue):
        video = queue[i]
        i += 1
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
                stream=args.stream,
                video_num=i,
                video_total=initial_queue_size,
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
        except NonEnglishError as exc:
            print(f"  \u2717 NON-ENGLISH: {exc}")
            append_jsonl(NON_ENGLISH_JSONL, {
                "video_file": video.name,
                "video_stem": video.stem,
                "video_path": str(video),
                "detected_language": exc.detected_lang,
                "english_fraction": exc.details["english_fraction"],
                "vote_counts": exc.details["vote_counts"],
                "samples": exc.details["samples"],
                "detection_time_s": round(exc.elapsed, 2),
                "ts": int(time.time()),
            })
            non_english_count += 1
        except NoAudioError as exc:
            print(f"  \u2717 NO AUDIO: {exc}")
            append_jsonl(NO_AUDIO_JSONL, {
                "video_file": video.name,
                "video_stem": video.stem,
                "video_path": str(video),
                "ts": int(time.time()),
            })
            no_audio_count += 1
        except Exception as exc:
            # Truncate verbose errors (e.g. ffmpeg stderr dumps) to first line / 200 chars
            err_msg = str(exc).split("\n")[0].strip()[:200]
            print(f"  ✗ FAILED: {err_msg}")
            append_jsonl(TRANSCRIPT_LOG_JSONL, {
                "video_file": video.name,
                "video_path": str(video),
                "status": "error",
                "error": str(exc),
                "ts": int(time.time()),
            })
            failures += 1

        # Rescan download directories for videos added since the last check
        if not args.file:
            transcribed = load_transcribed_set()
            non_english_stems = load_non_english_set()
            no_audio_stems = load_no_audio_set()
            for new_video in find_videos(DOWNLOAD_DIR):
                if new_video not in seen_videos and (
                    args.force or new_video.stem not in transcribed
                ) and (
                    args.force_lang_check or new_video.stem not in non_english_stems
                ) and new_video.stem not in no_audio_stems:
                    print(f"  + Discovered new video: {new_video.name}")
                    queue.append(new_video)
                    seen_videos.add(new_video)
            if len(queue) > i:
                print(f"  Queue: {i} done, {len(queue) - i} remaining (including any new)")

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Transcription complete: {successes} ok, {non_english_count} non-English, "
          f"{no_audio_count} no-audio, {failures} failed in {total_time:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
