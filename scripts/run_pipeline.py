#!/usr/bin/env python3
"""Run one or more pipeline steps for IA astronaut interview harvesting."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = {
    1: ROOT / "scripts" / "1_scan_ia_metadata.py",
    2: ROOT / "scripts" / "2_classify_candidates.py",
    3: ROOT / "scripts" / "3_download_lowres.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IA astronaut interview harvest pipeline")
    parser.add_argument("--steps", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--step1-max-pages", type=int, default=0)
    parser.add_argument("--step1-max-queries", type=int, default=0)
    parser.add_argument("--step1-max-items", type=int, default=0)
    parser.add_argument("--step2-limit", type=int, default=0)
    parser.add_argument("--step3-limit", type=int, default=0)
    parser.add_argument("--step3-dry-run", action="store_true")
    args = parser.parse_args()

    total_start = time.time()
    print("=" * 70)
    print("IA Astronaut Interview Harvest Pipeline")
    print("=" * 70)
    print(f"Running steps: {args.steps}")

    for step in args.steps:
        path = STEPS[step]
        cmd = [sys.executable, str(path)]
        if step == 1:
            if args.step1_max_pages > 0:
                cmd.extend(["--max-pages", str(args.step1_max_pages)])
            if args.step1_max_queries > 0:
                cmd.extend(["--max-queries", str(args.step1_max_queries)])
            if args.step1_max_items > 0:
                cmd.extend(["--max-items", str(args.step1_max_items)])
        if step == 2:
            cmd.extend(["--model", args.model, "--ollama-url", args.ollama_url])
            if args.step2_limit > 0:
                cmd.extend(["--limit", str(args.step2_limit)])
        if step == 3:
            if args.step3_limit > 0:
                cmd.extend(["--limit", str(args.step3_limit)])
            if args.step3_dry_run:
                cmd.append("--dry-run")

        print(f"\n{'=' * 70}")
        print(f"Step {step}: {path.name}")
        print(f"{'=' * 70}")

        started = time.time()
        completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if completed.returncode != 0:
            print(f"Step {step} failed with exit code {completed.returncode}")
            sys.exit(completed.returncode)

        elapsed = time.time() - started
        print(f"Step {step} completed in {elapsed:.1f}s")

    total = time.time() - total_start
    print(f"\nPipeline finished in {total:.1f}s")


if __name__ == "__main__":
    main()
