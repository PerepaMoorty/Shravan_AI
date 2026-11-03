#!/usr/bin/env python3
"""CLI: parse every KFall trial under DATA_ROOT into a timeline CSV (see
shravan_ai/config.py for how DATA_ROOT resolves)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shravan_ai import config
from shravan_ai.pipeline import run_pipeline


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"Data root:  {config.describe_data_root()}")
    print(f"Output dir: {config.OUTPUT_DIR}\n")

    summary = run_pipeline()

    print(f"\nProcessed {len(summary['processed'])} trial(s):")
    for path in summary["processed"]:
        print(f"  {path}")

    if summary["skipped"]:
        print(f"\nSkipped {len(summary['skipped'])} trial(s):")
        for msg in summary["skipped"]:
            print(f"  {msg}")

    if summary["needs_review"]:
        print(f"\n{len(summary['needs_review'])} trial(s) need manual direction review "
              f"(see {config.TRIAL_TIMELINE_DIR / '_needs_review.csv'}):")
        for msg in summary["needs_review"]:
            print(f"  {msg}")


if __name__ == "__main__":
    main()
