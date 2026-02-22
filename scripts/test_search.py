#!/usr/bin/env python3
"""Test harness: query the semantic search index from the command line.

Loads the index built by Stage 6, embeds an arbitrary query using the same
sentence-transformers model, and returns the top-N most similar questions
ranked by cosine similarity.

Usage:
  uv run python scripts/test_search.py "What is it like to sleep in space?"
  uv run python scripts/test_search.py "How do astronauts exercise?" --top 20
  uv run python scripts/test_search.py --interactive
  uv run python scripts/test_search.py --batch-test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from astro_ia_harvest.config import SEARCH_INDEX_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def load_index(index_dir: Path) -> tuple[dict, list[dict], np.ndarray]:
    """Load the full search index.

    Returns (meta, questions, embeddings) where embeddings is a 2-D
    float32 numpy array of shape (num_questions, embedding_dim).
    """
    meta_path = index_dir / "index_meta.json"
    questions_path = index_dir / "questions.json"
    embeddings_path = index_dir / "embeddings.bin"

    for p in (meta_path, questions_path, embeddings_path):
        if not p.exists():
            print(f"ERROR: Missing index file: {p}")
            print("       Run Stage 6 first:  uv run python scripts/6_build_search_index.py")
            sys.exit(1)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    dim = meta["embedding_dim"]
    raw = np.frombuffer(embeddings_path.read_bytes(), dtype=np.float32)
    embeddings = raw.reshape(meta["num_questions"], dim)

    return meta, questions, embeddings


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    query: str,
    model,
    embeddings: np.ndarray,
    questions: list[dict],
    top_k: int = 10,
    threshold: float = 0.0,
) -> list[dict]:
    """Embed a query and return top-K matching questions by cosine similarity.

    Embeddings are pre-normalised (L2 norm = 1), so cosine similarity
    reduces to a simple dot product.
    """
    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)  # shape (1, dim)

    # Cosine similarity via dot product (embeddings are already unit-norm)
    scores = (embeddings @ query_vec.T).squeeze()  # shape (num_questions,)

    # Rank & filter
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < threshold:
            break
        q = dict(questions[idx])
        q["score"] = round(score, 4)
        results.append(q)

    return results


def format_result(rank: int, result: dict) -> str:
    """Pretty-print a single search result."""
    lines = [
        f"  #{rank}  score={result['score']:.4f}",
        f"       Q: \"{result['text']}\"",
        f"       Source: {result['source_file']}",
        f"       Event: {result['event_type']}  |  Pair #{result['pair_index']}",
        f"       Timing: question {result['question_start']:.1f}s–{result['question_end']:.1f}s",
    ]
    for i, ans in enumerate(result.get("answers", []), 1):
        lines.append(f"       Answer {i}: {ans['start']:.1f}s–{ans['end']:.1f}s")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch self-test
# ---------------------------------------------------------------------------

BATCH_QUERIES = [
    # Semantic matches expected (common astronaut Q&A topics)
    "What is it like to sleep in space?",
    "How do astronauts exercise on the space station?",
    "What do you eat in space?",
    "How long does it take to get to the International Space Station?",
    "What happens if you get sick in space?",
    "Do you miss your family while in space?",
    "What does Earth look like from space?",
    "How do you go to the bathroom in zero gravity?",
    "What experiments are you doing on the space station?",
    "How did you become an astronaut?",
    # Slightly unusual / edge-case queries
    "rocket launch experience feelings",
    "effects of microgravity on the human body",
    "spacewalk extravehicular activity",
    "water recycling life support systems",
    "international cooperation space exploration",
]


def run_batch_test(model, embeddings: np.ndarray, questions: list[dict]) -> None:
    """Run a battery of test queries and report results."""
    print(f"\n{'=' * 70}")
    print(f"Batch Self-Test  ({len(BATCH_QUERIES)} queries × {len(questions)} questions)")
    print(f"{'=' * 70}")

    total_time = 0.0
    all_top_scores = []

    for query in BATCH_QUERIES:
        t0 = time.time()
        results = search(query, model, embeddings, questions, top_k=5)
        elapsed = time.time() - t0
        total_time += elapsed

        top_score = results[0]["score"] if results else 0.0
        all_top_scores.append(top_score)

        print(f"\n  Query: \"{query}\"")
        print(f"  Time: {elapsed * 1000:.1f}ms  |  Top score: {top_score:.4f}")
        for i, r in enumerate(results[:3], 1):
            print(f"    {i}. [{r['score']:.4f}] {r['text'][:90]}…" if len(r['text']) > 90
                  else f"    {i}. [{r['score']:.4f}] {r['text']}")

    # Summary
    scores_arr = np.array(all_top_scores)
    print(f"\n{'─' * 70}")
    print(f"  Queries:        {len(BATCH_QUERIES)}")
    print(f"  Avg top score:  {scores_arr.mean():.4f}")
    print(f"  Min top score:  {scores_arr.min():.4f}")
    print(f"  Max top score:  {scores_arr.max():.4f}")
    print(f"  Total time:     {total_time * 1000:.1f}ms")
    print(f"  Avg per query:  {total_time / len(BATCH_QUERIES) * 1000:.1f}ms")
    print(f"{'─' * 70}")


# ---------------------------------------------------------------------------
# Cross-similarity analysis
# ---------------------------------------------------------------------------

def run_cross_similarity(embeddings: np.ndarray, questions: list[dict], top_k: int = 5) -> None:
    """Find the most similar question *pairs* in the entire index.

    Useful for spotting near-duplicates across different videos/events.
    """
    print(f"\n{'=' * 70}")
    print("Cross-Similarity Analysis (finding duplicate / near-duplicate questions)")
    print(f"{'=' * 70}")

    # Full pairwise cosine similarity (embeddings already unit-norm)
    sim_matrix = embeddings @ embeddings.T
    n = sim_matrix.shape[0]

    # Zero out the diagonal (self-similarity = 1.0)
    np.fill_diagonal(sim_matrix, 0.0)

    # Find top-K most similar pairs
    flat_indices = np.argsort(sim_matrix.ravel())[::-1][:top_k * 2]  # ×2 because symmetric

    seen_pairs: set[tuple[int, int]] = set()
    pairs_shown = 0
    for flat_idx in flat_indices:
        i, j = divmod(int(flat_idx), n)
        pair_key = (min(i, j), max(i, j))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        score = float(sim_matrix[i, j])
        q_i = questions[i]
        q_j = questions[j]

        pairs_shown += 1
        print(f"\n  Pair #{pairs_shown}  similarity={score:.4f}")
        print(f"    A: \"{q_i['text'][:100]}\"")
        print(f"       Source: {q_i['source_file'][:80]}")
        print(f"    B: \"{q_j['text'][:100]}\"")
        print(f"       Source: {q_j['source_file'][:80]}")

        if pairs_shown >= top_k:
            break


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode(model, embeddings: np.ndarray, questions: list[dict], top_k: int) -> None:
    """REPL loop for testing queries interactively."""
    print(f"\nInteractive search mode (type 'quit' to exit)")
    print(f"Index: {len(questions)} questions  |  Top-K: {top_k}")
    print(f"{'─' * 60}")

    while True:
        try:
            query = input("\n  Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("  Bye!")
            break

        t0 = time.time()
        results = search(query, model, embeddings, questions, top_k=top_k)
        elapsed = time.time() - t0

        print(f"  ({elapsed * 1000:.1f}ms, {len(results)} results)\n")
        for i, r in enumerate(results, 1):
            print(format_result(i, r))
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test harness for the semantic search index",
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="Free-text question to search for.",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of results to return (default: 10).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Minimum cosine similarity score (default: 0.0).",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Enter interactive REPL mode.",
    )
    parser.add_argument(
        "--batch-test", action="store_true",
        help="Run a battery of test queries and report aggregate stats.",
    )
    parser.add_argument(
        "--cross-sim", action="store_true",
        help="Find the most similar question pairs across all videos.",
    )
    parser.add_argument(
        "--cross-sim-top", type=int, default=15,
        help="Number of top similar pairs to show (default: 15).",
    )
    parser.add_argument(
        "--index-dir", type=Path, default=SEARCH_INDEX_DIR,
        help=f"Path to the search index directory (default: {SEARCH_INDEX_DIR}).",
    )
    args = parser.parse_args()

    # Load index
    print(f"Loading search index from: {args.index_dir}")
    meta, questions, embeddings = load_index(args.index_dir)
    print(f"  Model: {meta['model']}")
    print(f"  Questions: {meta['num_questions']}")
    print(f"  Embedding dim: {meta['embedding_dim']}")
    print(f"  Built at: {meta['built_at']}")

    # Load the same model for query encoding
    from sentence_transformers import SentenceTransformer

    print(f"\nLoading model: {meta['model']}")
    model = SentenceTransformer(meta["model"])

    if args.batch_test:
        run_batch_test(model, embeddings, questions)
        return

    if args.cross_sim:
        run_cross_similarity(embeddings, questions, top_k=args.cross_sim_top)
        return

    if args.interactive:
        interactive_mode(model, embeddings, questions, top_k=args.top)
        return

    if not args.query:
        parser.print_help()
        print("\nProvide a query, or use --interactive / --batch-test / --cross-sim")
        sys.exit(1)

    # Single query
    t0 = time.time()
    results = search(args.query, model, embeddings, questions,
                     top_k=args.top, threshold=args.threshold)
    elapsed = time.time() - t0

    print(f"\n  Query: \"{args.query}\"")
    print(f"  Results: {len(results)} (in {elapsed * 1000:.1f}ms)\n")

    for i, r in enumerate(results, 1):
        print(format_result(i, r))
        print()


if __name__ == "__main__":
    main()
