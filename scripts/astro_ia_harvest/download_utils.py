from __future__ import annotations

import re
from pathlib import Path


def canonical_video_key(name: str) -> str:
    """Normalize a filename to a stable key so renamed historical files still match.

    Strips leading ISO-like date prefixes (e.g. ``2019-02-07T16-34-39_``) so
    that files from the ISSiRT archive (which carry such prefixes) match the
    same videos downloaded here without them.
    """
    raw = Path(name).name.lower()
    # Strip date-time prefix: 2019-02-07T16-34-39_ (ISSiRT naming convention)
    raw = re.sub(r"^\d{4}-\d{2}-\d{2}t\d{2}-\d{2}-\d{2}_", "", raw)
    raw = raw.replace(".ia.mp4", "")
    raw = raw.replace("_lowres", "")
    while True:
        stem, dot, ext = raw.rpartition(".")
        if not dot:
            break
        if ext in {"mp4", "mxf", "mov", "m4v", "avi", "mpg", "mpeg", "webm"}:
            raw = stem
            continue
        break
    return re.sub(r"[^a-z0-9]+", "", raw)


def build_candidate_urls(identifier: str, filename: str) -> list[tuple[str, str]]:
    """Return URL candidates in priority order with labels."""
    base = f"https://archive.org/download/{identifier}"
    name = Path(filename).name
    stem = _strip_video_extensions(name)

    urls: list[tuple[str, str]] = []
    urls.append((f"{base}/{stem}.ia.mp4", "lowres_variant"))
    urls.append((f"{base}/{stem}.mp4", "mp4_variant"))
    if name.lower().endswith(".mp4"):
        urls.append((f"{base}/{name}", "original_mp4_name"))
    return urls


def build_output_name(identifier: str, filename: str) -> str:
    stem = _strip_video_extensions(Path(filename).name)
    return f"{identifier}__{stem}_lowres.mp4"


def _strip_video_extensions(name: str) -> str:
    n = name
    for ext in (".ia.mp4", ".mp4", ".mxf", ".mov", ".m4v", ".avi", ".mpg", ".mpeg", ".webm"):
        if n.lower().endswith(ext):
            n = n[: -len(ext)]
    return n.rstrip(".")
