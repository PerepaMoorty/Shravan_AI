#!/usr/bin/env python3
"""CLI: plot every trial timeline's acceleration magnitude with
onset/impact markers, to visually confirm labels line up with sensor spikes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from shravan_ai import config
from shravan_ai.sanity_check import plot_trial


def main():
    print(f"Output dir: {config.OUTPUT_DIR}\n")

    timeline_dir = config.TRIAL_TIMELINE_DIR
    plot_dir = config.SANITY_PLOT_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    timeline_paths = sorted(p for p in timeline_dir.glob("*.csv") if not p.name.startswith("_"))
    if not timeline_paths:
        print(f"No trial timelines found in {timeline_dir}. Run run_pipeline.py first.")
        return

    for timeline_path in timeline_paths:
        timeline = pd.read_csv(timeline_path)
        out_path = plot_dir / timeline_path.name.replace(".csv", ".png")
        plot_trial(timeline, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
