from __future__ import annotations

import json
import re

import requests


# ---------------------------------------------------------------------------
# Generic Ollama helpers
# ---------------------------------------------------------------------------

def call_ollama(
    ollama_url: str,
    model: str,
    user_prompt: str,
    *,
    system: str = "",
    temperature: float = 0.1,
    num_ctx: int = 32768,
    num_predict: int = 8192,
    timeout: int = 600,
) -> str:
    """Call Ollama ``/api/generate`` and return the response text."""
    payload: dict = {
        "model": model,
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    if system:
        payload["system"] = system
    resp = requests.post(ollama_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def extract_json(text: str) -> dict | list | None:
    """Robustly extract a JSON object or array from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # JSON array
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Filename classification (used by stage 2)
# ---------------------------------------------------------------------------

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
    prompt = f"""You are classifying NASA video files to find interactive Q&A events worth transcribing.

CRITICAL: The **filename** is the most reliable indicator of what a video contains. \
The title, description, and subject fields often describe the parent *collection* \
(e.g. "Crew-11 Content", "Resource Reel") rather than the individual video. \
Always prioritize the filename over title/description/subject. If the filename \
clearly indicates a news conference, interview, or education event, KEEP it \
regardless of what the other fields say.

The downstream pipeline only processes these event types:
  1. student_qa — School downlinks, ARISS contacts, education inflight events where students ask an astronaut questions. Key signals in filename: "Education Inflight", "Inflight with [school/org]", "EDU_Inflight", "ARISS", "school", "student", "ham radio", any organization or media outlet name after "Inflight" (e.g. "Inflight_HCHSA", "Inflight_NOGGIN", "Inflight_CNBC").
  2. press_conference — News conferences, pre-launch/post-flight/post-mission press briefings, flight readiness reviews with Q&A. Key signals in filename: "News_Conference", "News Conference", "Press_Conference", "Postflight", "Post-Flight", "Post_Flight", "Pre-Launch", "Flight_Readiness", "Mission_Overview_News".
  3. media_interview — An astronaut discussing life in space with a specific TV station, newspaper, or radio outlet. Key signals in filename: "Discusses_Life_In_Space", "Talks_with", "Discuss", name of a TV/radio station (e.g. WTKR-TV, NPR, CNBC, KHQ-TV).
  4. panel — Panel discussions or roundtables with multiple speakers.

KEEP files whose **filename** clearly matches one of the four types above.

REJECT everything else, including:
  - Space to Ground weekly recap segments (narrated, no Q&A)
  - Launch, splashdown, landing, docking, undocking coverage
  - Spacewalk / EVA coverage
  - Highlights packages or montages
  - B-roll collections (but NOT if the filename says News Conference, Interview, etc.)
  - Change of command ceremonies, welcome events, arrival events
  - Raw camera feeds, Earth views, flyovers
  - Animations, simulations
  - Training footage
  - Film magazine scans (Apollo, Gemini, etc.)
  - General "On-Orbit" content without a named event partner
  - Diary camera / GoPro footage
  - "Meet the astronaut" profile videos, "Science in Orbit" montages
  - Anything else without clear Q&A or interview structure in the filename

When in doubt, REJECT — it is much cheaper to miss a borderline file than to download and process thousands of irrelevant ones.

Respond in strict JSON with keys:
- decision: "keep" or "reject"
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
