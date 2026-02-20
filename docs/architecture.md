# Architecture

## Goal

Harvest large volumes of astronaut interview and Q&A style footage from Internet Archive (IA), while avoiding unrelated passive feeds and avoiding redownloading media that already exists in `D:\ISSiRT_ia_videos_raw`.

## Project Layout

```
scripts/                       # All Python code lives here (src/ reserved for future Vite project)
  astro_ia_harvest/            # Shared library package
    config.py                  # Paths, API settings, uploader list
    ia_api.py                  # IA search + metadata fetching
    ollama_client.py           # Ollama LLM classification
    download_utils.py          # URL construction, dedup key normalization
    jsonl_utils.py             # JSONL read/write helpers
  1_scan_ia_metadata.py        # Step 1 entry point
  2_classify_candidates.py     # Step 2 entry point
  3_download_lowres.py         # Step 3 entry point
  run_pipeline.py              # Orchestrator
data/                          # Runtime data (gitignored)
docs/                          # Architecture and operations docs
```

## Pipeline Steps

1. `scripts/1_scan_ia_metadata.py`
- Queries IA by uploader only (the five known NASA JSC PAO accounts in `config.py`).
- Fetches item metadata and writes one JSONL record per candidate video file.
- Stores identifier progress in `data/ia_identifiers_seen.txt` for incremental reruns.

2. `scripts/2_classify_candidates.py`
- Sends each filename/title/description/subject to Ollama for keep/reject classification.
- Writes a persistent classification JSONL with decision traces and model used.
- If Ollama is unreachable, records default to reject and can be re-classified later.

3. `scripts/3_download_lowres.py`
- Downloads only records marked `likely_relevant=true`.
- Tries low-res IA variant first (`.ia.mp4`), then fallback variants.
- Skips if the source video key already exists in local project downloads or the legacy directory.

4. `scripts/run_pipeline.py`
- Orchestrates step execution and passes classifier runtime options.

## Data Flow

- Step 1 output: `data/ia_video_metadata.jsonl`
- Step 2 output: `data/classified_candidates.jsonl`
- Step 3 outputs:
  - video files under `D:\ask_anything_ia_videos_raw`
  - `data/download_log.csv`
  - `data/download_failures.jsonl`

## Why This Design

- Decouples expensive classification from scanning and downloading.
- Supports one-time long classification runs with resumability.
- Avoids dependency on Imagery Online and start-time metadata entirely.
