# Classification Strategy

## Approach

All classification is done via Ollama (local LLM). The model receives the filename,
title, description, and subject for each candidate video file and returns a keep/reject
decision with confidence and reasoning.

The classifier is tightly aligned with the downstream event types defined in
`5a_classify_event.py`. It specifically targets:

- **student_qa** — School downlinks, ARISS contacts, education inflight events
- **press_conference** — News conferences, pre/post-flight briefings
- **media_interview** — Astronaut interviews with named TV/radio/media outlets
- **panel** — Panel discussions or roundtables

Everything else (Space to Ground segments, launch/landing ops coverage, spacewalk
feeds, resource reels, highlights packages, ceremonies, raw camera feeds, training
footage, etc.) is rejected to avoid downloading thousands of irrelevant files.

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
