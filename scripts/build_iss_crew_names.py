#!/usr/bin/env python3
"""Extract unique ISS crew surnames from crew_arr_dep.json for use in iss_topic_filter.

Reads the ISSiRT crew arrivals/departures file and produces a deduplicated,
sorted list of surnames. Short or common names that would cause false positives
in filename matching are flagged for review.

Output: scripts/astro_ia_harvest/iss_crew_names.json
"""
from __future__ import annotations

import json
from pathlib import Path

CREW_SOURCE = Path(r"F:\_repos\ISSiRT_assets\ISSiRT_web_assets\crew_arr_dep.json")
OUTPUT = Path(__file__).resolve().parent / "astro_ia_harvest" / "iss_crew_names.json"

# Names that are too short or too common to safely use as standalone patterns
# in filename matching (would cause false positives).
# These are excluded from automatic matching unless they appear with a first name.
MIN_SURNAME_LENGTH = 4
AMBIGUOUS_NAMES = {
    # Common English words or very short strings that appear in non-crew contexts
    "lee",    # very common word/name
    "wolf",   # common word
    "lu",     # too short
    "li",     # too short  
    "ross",   # common name, appears in non-ISS contexts
    "ford",   # common word/brand
    "long",   # common word
    "scott",  # extremely common first name used as surname
    "brown",  # very common name
    "jones",  # very common name
    "smith",  # very common name  
    "clark",  # very common name
    "hill",   # common word  
    "barry",  # common first name
    "hart",   # common word/name
    "du",     # too short
    "kim",    # very common name, too short
    "king",   # common word/name
}


def main() -> None:
    with open(CREW_SOURCE, encoding="utf-8") as f:
        crew = json.load(f)

    print(f"Source records: {len(crew)}")

    # Collect unique surnames
    all_surnames: dict[str, set[str]] = {}  # lowercase -> set of first names
    for c in crew:
        last = c["name_last"].strip()
        first = c["name_first"].strip()
        key = last.lower()
        if key not in all_surnames:
            all_surnames[key] = set()
        all_surnames[key].add(first)

    print(f"Unique surnames: {len(all_surnames)}")

    # Split into usable and ambiguous
    usable = {}
    ambiguous = {}
    too_short = {}
    for name, firsts in sorted(all_surnames.items()):
        if name in AMBIGUOUS_NAMES:
            ambiguous[name] = firsts
        elif len(name) < MIN_SURNAME_LENGTH:
            too_short[name] = firsts
        else:
            usable[name] = firsts

    print(f"\nUsable as standalone surname pattern: {len(usable)}")
    print(f"Ambiguous (need first+last combo):    {len(ambiguous)}")
    print(f"Too short (< {MIN_SURNAME_LENGTH} chars):               {len(too_short)}")

    # Show ambiguous names
    if ambiguous:
        print("\n  Ambiguous names (excluded from standalone matching):")
        for name, firsts in sorted(ambiguous.items()):
            print(f"    {name} ({', '.join(sorted(firsts))})")

    if too_short:
        print(f"\n  Too short (< {MIN_SURNAME_LENGTH} chars):")
        for name, firsts in sorted(too_short.items()):
            print(f"    {name} ({', '.join(sorted(firsts))})")

    # Build output structure
    output = {
        "_comment": "Auto-generated from crew_arr_dep.json. Do not edit manually.",
        "_source": str(CREW_SOURCE),
        "surnames": sorted(usable.keys()),
        "first_last_combos": {
            name: sorted(firsts) for name, firsts in sorted({**ambiguous, **too_short}.items())
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWritten to {OUTPUT}")
    print(f"  {len(output['surnames'])} standalone surnames")
    print(f"  {len(output['first_last_combos'])} first+last combo entries")


if __name__ == "__main__":
    main()
