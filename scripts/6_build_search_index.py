#!/usr/bin/env python3
"""Stage 6: Build semantic search index from Q&A text files.

Reads every .qa_text.json produced by Stage 5c, extracts question text, and
generates sentence embeddings using `all-MiniLM-L6-v2`.  The output is a set
of static files that enable fully client-side semantic search:

  data/search_index/
    index_meta.json   – model info, dimensions, question count, build timestamp
    questions.json    – question metadata with video timing (no answer text)
    embeddings.bin    – float16 binary blob (num_questions × 384 × 2 bytes)

Embeddings are stored as float16 to halve file size with no measurable
impact on ranking quality.  At query time the browser loads a matching ONNX
model via transformers.js, embeds the user's query, widens the stored
float16 embeddings to float32, and computes cosine similarity — zero server
traffic required.

This script is a PURE INDEX BUILDER — it contains no content filtering.
All quality decisions are made upstream in Stage 5b during extraction.

Usage:
  uv run python scripts/6_build_search_index.py
  uv run python scripts/6_build_search_index.py --force
  uv run python scripts/6_build_search_index.py --qa-text-file data/qa_text/<file>.qa_text.json
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from astro_ia_harvest.config import (  # noqa: E402
    QA_TEXT_DIR,
    SEARCH_INDEX_DIR,
    ensure_directories,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 64  # sentences per GPU/CPU batch

INDEX_META_FILE = "index_meta.json"
QUESTIONS_FILE = "questions.json"
EMBEDDINGS_FILE = "embeddings.bin"


# ---------------------------------------------------------------------------

def load_qa_text(path: Path) -> dict | None:
    """Load and validate a single .qa_text.json file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  WARNING: Could not read {path.name}: {exc}")
        return None

    if "qa_pairs" not in data:
        print(f"  WARNING: No qa_pairs in {path.name}")
        return None

    return data


def extract_questions(qa_text: dict, source_filename: str) -> list[dict]:
    """Pull indexable question records from a qa_text document.

    Each record carries enough metadata to reference back to the original
    video at the correct timestamp — but intentionally omits answer text
    to keep the index compact.

    No content filtering is applied here; all quality decisions are made
    upstream in Stage 5b.
    """
    questions: list[dict] = []

    for pair in qa_text.get("qa_pairs", []):
        q = pair.get("question", {})
        text = (q.get("text") or "").strip()

        if not text:
            continue

        # Build answer timing references (no text)
        answer_timings = [
            {"start": ans["start"], "end": ans["end"]}
            for ans in pair.get("answers", [])
        ]

        questions.append({
            "text": text,
            "source_file": source_filename,
            "event_type": qa_text.get("event_type", "unknown"),
            "pair_index": pair.get("index", 0),
            "question_start": q.get("start"),
            "question_end": q.get("end"),
            "answers": answer_timings,
        })

    return questions


def load_model():
    """Load the sentence-transformers model (downloads on first run)."""
    from sentence_transformers import SentenceTransformer

    print(f"\n  Loading model: {MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    return model


def encode_questions(model, texts: list[str]) -> np.ndarray:
    """Encode a list of question strings into a 2-D numpy array of embeddings.

    Returns shape (len(texts), EMBEDDING_DIM) with float32 dtype.
    """
    print(f"\n  Encoding {len(texts)} questions (batch_size={BATCH_SIZE})…")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # pre-normalise so cosine sim = dot product
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"  Encoding complete in {elapsed:.1f}s ({len(texts) / max(elapsed, 0.01):.0f} q/s)")
    return embeddings.astype(np.float32)


def save_embeddings_bin(embeddings: np.ndarray, path: Path) -> None:
    """Write embeddings as a flat float16 binary file.

    The model produces float32 embeddings which are downcast to float16
    before writing.  This halves the file size with no measurable effect
    on cosine-similarity ranking (verified: top-10 overlap = 10/10 across
    random query samples).

    Layout: row-major, num_questions × embedding_dim × 2 bytes.
    The browser reads this with ``new Float32Array(new Float16Array(buf))``
    or equivalent widen-on-load approach.
    """
    emb16 = embeddings.astype(np.float16)
    path.write_bytes(emb16.tobytes())
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Saved embeddings: {path}  ({size_mb:.2f} MB, float16)")


def save_questions_json(questions: list[dict], path: Path) -> None:
    """Write the question metadata array as pretty-printed JSON."""
    # Add sequential IDs that match the row index in embeddings.bin
    for i, q in enumerate(questions):
        q["id"] = i

    path.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Saved questions:  {path}  ({len(questions)} entries)")


def save_index_meta(num_questions: int, path: Path) -> None:
    """Write the index metadata file."""
    meta = {
        "version": 2,
        "model": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "embedding_dtype": "float16",
        "num_questions": num_questions,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "questions": QUESTIONS_FILE,
            "embeddings": EMBEDDINGS_FILE,
        },
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Saved metadata:   {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_index(qa_text_files: list[Path], *, force: bool = False) -> None:
    """Build the full search index from a list of .qa_text.json files."""
    meta_path = SEARCH_INDEX_DIR / INDEX_META_FILE
    questions_path = SEARCH_INDEX_DIR / QUESTIONS_FILE
    embeddings_path = SEARCH_INDEX_DIR / EMBEDDINGS_FILE

    if not force and meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"\n  Existing index has {existing.get('num_questions', '?')} questions")
        print(f"  Built at: {existing.get('built_at', '?')}")
        print("  Use --force to rebuild.")
        return

    # ------------------------------------------------------------------
    # 1. Collect questions from all qa_text files
    # ------------------------------------------------------------------
    all_questions: list[dict] = []
    files_processed = 0
    files_skipped = 0

    for i, path in enumerate(qa_text_files, 1):
        print(f"\n[{i}/{len(qa_text_files)}] {path.name}")
        qa_text = load_qa_text(path)
        if qa_text is None:
            files_skipped += 1
            continue

        questions = extract_questions(qa_text, source_filename=path.name)
        print(f"  Extracted {len(questions)} indexable questions")
        all_questions.extend(questions)
        files_processed += 1

    if not all_questions:
        print("\nERROR: No questions extracted. Nothing to index.")
        return

    print(f"\n{'-' * 60}")
    print(f"Total: {len(all_questions)} questions from {files_processed} files "
          f"({files_skipped} skipped)")
    print(f"{'-' * 60}")

    # ------------------------------------------------------------------
    # 1b. Deduplicate questions by exact text
    # ------------------------------------------------------------------
    # NASA sometimes uploads the same event under multiple IA identifiers,
    # producing identical transcripts and therefore identical question
    # text.  When the same question appears from multiple source files,
    # keep the instance from the source that contributed the most
    # questions overall (best event coverage).
    from collections import Counter
    source_question_counts = Counter(q["source_file"] for q in all_questions)

    seen_texts: dict[str, int] = {}  # normalized text -> index in deduped list
    deduped: list[dict] = []
    duplicates_removed = 0

    for q in all_questions:
        text = q["text"]
        key = text.strip().lower()
        if key in seen_texts:
            # Already saw this text — keep the one from the richer source
            existing_idx = seen_texts[key]
            existing_source = deduped[existing_idx]["source_file"]
            new_source = q["source_file"]
            if source_question_counts[new_source] > source_question_counts[existing_source]:
                deduped[existing_idx] = q  # replace with better source
            duplicates_removed += 1
        else:
            seen_texts[key] = len(deduped)
            deduped.append(q)

    if duplicates_removed:
        print(f"\n  Deduplication: removed {duplicates_removed} duplicate question texts")
        print(f"  Questions after dedup: {len(deduped)} (was {len(all_questions)})")
    all_questions = deduped

    # ------------------------------------------------------------------
    # 2. Generate embeddings
    # ------------------------------------------------------------------
    model = load_model()
    texts = [q["text"] for q in all_questions]
    embeddings = encode_questions(model, texts)

    assert embeddings.shape == (len(all_questions), EMBEDDING_DIM), (
        f"Unexpected shape {embeddings.shape}"
    )

    # ------------------------------------------------------------------
    # 3. Save outputs
    # ------------------------------------------------------------------
    print(f"\n  Writing index to: {SEARCH_INDEX_DIR}")
    save_questions_json(all_questions, questions_path)
    save_embeddings_bin(embeddings, embeddings_path)
    save_index_meta(len(all_questions), meta_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6: Build semantic search index from Q&A text files"
    )
    parser.add_argument(
        "--qa-text-file", type=Path, default=None,
        help="Process a single .qa_text.json instead of all files in data/qa_text/.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild the entire index even if it already exists.",
    )
    args = parser.parse_args()

    ensure_directories()

    print("=" * 70)
    print("Stage 6: Build Semantic Search Index")
    print("=" * 70)
    print(f"  QA text dir:    {QA_TEXT_DIR}")
    print(f"  Output dir:     {SEARCH_INDEX_DIR}")
    print(f"  Model:          {MODEL_NAME}")
    print(f"  Embedding dim:  {EMBEDDING_DIM}")

    if args.qa_text_file:
        if not args.qa_text_file.exists():
            print(f"ERROR: File not found: {args.qa_text_file}")
            sys.exit(1)
        candidates = [args.qa_text_file]
    else:
        candidates = sorted(QA_TEXT_DIR.glob("*.qa_text.json"))

    print(f"  Source files:    {len(candidates)}")

    if not candidates:
        print("\nNo .qa_text.json files found. Run Stage 5c first.")
        return

    build_index(candidates, force=args.force)

    print(f"\n{'=' * 70}")
    print("Search index build complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
