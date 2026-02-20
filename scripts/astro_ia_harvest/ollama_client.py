from __future__ import annotations

import json
import re

import requests


def classify_filename_with_ollama(
    *,
    ollama_url: str,
    model: str,
    filename: str,
    title: str,
    description: str,
    subject: str,
    timeout: int = 90,
) -> tuple[str, float, str]:
    """
    Returns (decision, confidence, reason).
    decision is keep/reject.
    """
    prompt = f"""You are classifying NASA-related video files for transcript harvesting.
Keep files likely to contain spoken interview/Q&A or press-briefing style dialogue.
Reject files likely to be passive camera feeds, scenic views, b-roll only, music-only, or technical loops.

Respond in strict JSON with keys:
- decision: \"keep\" or \"reject\"
- confidence: number from 0.0 to 1.0
- reason: short string

filename: {filename}
title: {title}
description: {description}
subject: {subject}
"""

    resp = requests.post(
        ollama_url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 4096},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json().get("response", "").strip()

    parsed = _extract_json(body)
    if not parsed:
        decision = "reject" if "reject" in body.lower() else "keep"
        return decision, 0.5, "fallback_parse"

    decision = str(parsed.get("decision", "reject")).strip().lower()
    if decision not in {"keep", "reject"}:
        decision = "reject"
    confidence = _to_confidence(parsed.get("confidence"))
    reason = str(parsed.get("reason", "")).strip() or "no_reason"
    return decision, confidence, reason


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _to_confidence(value: object) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.5
    if num < 0:
        return 0.0
    if num > 1:
        return 1.0
    return num
