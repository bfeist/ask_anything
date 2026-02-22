from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env from project root (silently ignored if it doesn't exist)
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
DOWNLOAD_DIR = Path(r"D:\ask_anything_ia_videos_raw")

IA_METADATA_JSONL = DATA_DIR / "ia_video_metadata.jsonl"
IA_PROGRESS_FILE = DATA_DIR / "ia_identifiers_seen.txt"
CLASSIFIED_JSONL = DATA_DIR / "classified_candidates.jsonl"
DOWNLOAD_LOG_CSV = DATA_DIR / "download_log.csv"
DOWNLOAD_FAILURES_JSONL = DATA_DIR / "download_failures.jsonl"

# Transcription outputs
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
TRANSCRIPT_LOG_JSONL = DATA_DIR / "transcript_log.jsonl"
NON_ENGLISH_JSONL = DATA_DIR / "non_english_videos.jsonl"
NO_AUDIO_JSONL = DATA_DIR / "no_audio_videos.jsonl"

# Stage 5 outputs
CLASSIFY_DIR = DATA_DIR / "classify"
QA_DIR = DATA_DIR / "qa"
QA_TEXT_DIR = DATA_DIR / "qa_text"
QA_EMPTY_JSONL = DATA_DIR / "qa_empty.jsonl"  # transcripts that produced no Q&A pairs

# Stage 6 outputs
SEARCH_INDEX_DIR = DATA_DIR / "search_index"

# Existing cache from the other project. Files found here will be treated as already downloaded.
EXISTING_DOWNLOAD_DIR = Path(r"D:\ISSiRT_ia_videos_raw")

IA_ROWS_PER_REQUEST = 100
IA_UPLOADERS = [
    "john.l.stoll@nasa.gov",  # Primary NASA TV uploader (John Stoll, JSC PAO)
    "elizabeth.k.weissinger@nasa.gov",  # JSC PAO uploader (Beth Weissinger)
    "dexter.herbert-1@nasa.gov",  # JSC PAO uploader (Dexter Herbert)
    "edmond.a.toma@nasa.gov",  # JSC PAO uploader (Edmond Toma)
    "e.toma@nasa.gov",  # JSC PAO uploader (older account)
]

# Extensions that can map to playable video assets.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mxf", ".m4v", ".avi", ".mpg", ".mpeg", ".webm"}

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "gemma3:12b"


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFY_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    QA_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    SEARCH_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()
