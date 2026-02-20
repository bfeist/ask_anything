# Astronaut IA Interview Harvester

This project scans Internet Archive for NASA-related videos, classifies likely interview/Q&A content, and downloads only low-res versions for downstream transcription.

## What This Solves

- Collects large amounts of interview-style astronaut footage.
- Filters out likely non-dialogue assets (for example passive camera feeds).
- Avoids redownloading files already present in the download location.
- Keeps each step incremental and resumable.

## Pipeline

1. `scripts/1_scan_ia_metadata.py`
- IA search and metadata collection.
- Output: `data/ia_video_metadata.jsonl`

2. `scripts/2_classify_candidates.py`
- Ollama-based interview/Q&A relevance classification.
- Output: `data/classified_candidates.jsonl`

3. `scripts/3_download_lowres.py`
- Downloads low-res candidates first (`.ia.mp4` preferred).
- Outputs:
	- `downloads/*.mp4`
	- `data/download_log.csv`
	- `data/download_failures.jsonl`

Orchestrator:

- `scripts/run_pipeline.py`

## Quick Start

```bash
uv sync
```

Run full pipeline:

```bash
uv run python scripts/run_pipeline.py --steps 1 2 3 --model gemma3:12b
```

Short iterative validation run (recommended first):

```bash
uv run python scripts/1_scan_ia_metadata.py --max-queries 1 --max-pages 1 --max-items 3
uv run python scripts/2_classify_candidates.py --limit 10
uv run python scripts/3_download_lowres.py --limit 5 --dry-run
uv run python scripts/run_pipeline.py --steps 1 2 3 --step1-max-queries 1 --step1-max-pages 1 --step1-max-items 2 --step2-limit 5 --step3-limit 3 --step3-dry-run
```

Run only classification with a different model:

```bash
uv run python scripts/2_classify_candidates.py --model gemma3:12b
```

## Documentation

- `docs/architecture.md`
- `docs/classification_strategy.md`
- `docs/operations.md`