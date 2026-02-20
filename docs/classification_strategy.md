# Classification Strategy

## Approach

All classification is done via Ollama (local LLM). The model receives the filename,
title, description, and subject for each candidate video file and returns a keep/reject
decision with confidence and reasoning. Correctly identifies interviews, press conferences,
discussions, profile pieces, and Q&A sessions while rejecting animations, resource reels,
docking footage, and operational B-roll.

## Ollama Prompt Contract

The Ollama classifier expects strict JSON:

```json
{
  "decision": "keep|reject",
  "confidence": 0.0,
  "reason": "short text"
}
```

If JSON parsing fails, the script falls back to word-matching the raw response body (`reason: fallback_parse`, `confidence: 0.5`). If Ollama itself throws an error, the record is rejected outright.

## Fallback Behaviour

If Ollama is unreachable the exception is caught, records are marked `decision=reject` with `method=ollama_fallback`. Re-run the classifier later to pick them up.

## Tuning Guidance

- Change the model: `--model <name>` (default: gemma3:12b).
- Adjust the prompt in `scripts/astro_ia_harvest/ollama_client.py`.
