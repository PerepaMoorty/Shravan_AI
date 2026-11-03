#!/usr/bin/env python3
"""CLI: run the stub model over every trial timeline, writing one model log
CSV per trial plus a lead/lag summary."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from shravan_ai import config
from shravan_ai.leadlag import compute_lead_lag, format_lead_lag
from shravan_ai.model import run_threshold_model


def main():
    print(f"Data root:  {config.describe_data_root()}")
    print(f"Output dir: {config.OUTPUT_DIR}\n")

    timeline_dir = config.TRIAL_TIMELINE_DIR
    model_log_dir = config.MODEL_LOG_DIR
    model_log_dir.mkdir(parents=True, exist_ok=True)

    timeline_paths = sorted(p for p in timeline_dir.glob("*.csv") if not p.name.startswith("_"))
    if not timeline_paths:
        print(f"No trial timelines found in {timeline_dir}. Run run_pipeline.py first.")
        return

    summary_rows = []
    for timeline_path in timeline_paths:
        timeline = pd.read_csv(timeline_path)
        model_log = run_threshold_model(timeline)

        out_path = model_log_dir / timeline_path.name.replace(".csv", "_model_log.csv")
        model_log.to_csv(out_path, index=False)

        result = compute_lead_lag(timeline, model_log)
        subject_id = timeline["subject_id"].iloc[0]
        trial_id = timeline["trial_id"].iloc[0]
        summary_rows.append(
            {
                "subject_id": subject_id,
                "trial_id": trial_id,
                "fall_direction": timeline["fall_direction"].iloc[0],
                "onset_lead_ms": result.onset_lead_ms,
                "impact_lead_ms": result.impact_lead_ms,
                "summary": format_lead_lag(result),
            }
        )
        print(f"{subject_id}/{trial_id}: {format_lead_lag(result)}")

    summary_path = model_log_dir / "_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nWrote {len(summary_rows)} model log(s) to {model_log_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
