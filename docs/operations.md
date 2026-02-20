# Operations

## Setup

1. Create environment and install deps:

```bash
uv sync
```

2. Ensure Ollama is running with a model (default: `gemma3:12b`).

## Common Runs

Run full pipeline:

```bash
uv run python scripts/run_pipeline.py --steps 1 2 3 --model gemma3:12b
```

Run classifier only:

```bash
uv run python scripts/2_classify_candidates.py --model gemma3:12b
```

Download only after classification exists:

```bash
uv run python scripts/3_download_lowres.py
```

Short iterative test pass:

```bash
uv run python scripts/1_scan_ia_metadata.py --max-queries 1 --max-pages 1 --max-items 3
uv run python scripts/2_classify_candidates.py --limit 10
uv run python scripts/3_download_lowres.py --limit 5 --dry-run
```

## Incremental Behavior

- Step 1 uses `data/ia_identifiers_seen.txt`.
- Step 2 skips `(identifier, filename)` already written to `data/classified_candidates.jsonl`.
- Step 3 skips videos already found in:
  - `D:\ask_anything_ia_videos_raw`
  - `D:\ISSiRT_ia_videos_raw`

## Reset Points

If you want a fresh rerun:

- Delete `data/ia_identifiers_seen.txt` and `data/ia_video_metadata.jsonl` for fresh IA scan.
- Delete `data/classified_candidates.jsonl` for fresh relevance classification.
- Keep download dirs if you still want dedupe against local copies.
