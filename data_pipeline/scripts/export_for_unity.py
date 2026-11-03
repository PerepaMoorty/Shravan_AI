#!/usr/bin/env python3
"""Bundle pipeline output into the per-trial files Unity reads from
StreamingAssets/Trials/:
  <trial>_timeline.csv       ground truth
  <trial>_model_log.csv      stub model's firing log
  <trial>_meta.json          trial summary and onset/impact times
  <trial>_gait_features.csv  estimated cadence/stride timing
plus _index.json listing every exported trial key.

Run run_pipeline.py and run_model.py first. If data_pipeline/
unity_export_config.json exists, only the trials it selects are exported
(see shravan_ai/curation.py); otherwise every processed trial is.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from shravan_ai import config, curation
from shravan_ai.gait_features import compute_gait_features
from shravan_ai.leadlag import compute_lead_lag

DEFAULT_UNITY_STREAMING_ASSETS = (
    REPO_ROOT.parent / "unity" / "ShravanAI" / "Assets" / "StreamingAssets" / "Trials"
)


def export_trial(trial_key: str, timeline_path: Path, model_log_path: Path, out_dir: Path):
    timeline = pd.read_csv(timeline_path)
    model_log = pd.read_csv(model_log_path)
    result = compute_lead_lag(timeline, model_log)

    shutil.copyfile(timeline_path, out_dir / f"{trial_key}_timeline.csv")
    shutil.copyfile(model_log_path, out_dir / f"{trial_key}_model_log.csv")

    gait_features = compute_gait_features(timeline)
    gait_features.to_csv(out_dir / f"{trial_key}_gait_features.csv", index=False)

    meta = {
        "subject_id": str(timeline["subject_id"].iloc[0]),
        "trial_id": str(timeline["trial_id"].iloc[0]),
        "fall_direction": str(timeline["fall_direction"].iloc[0]),
        "duration_s": float(timeline["timestamp_s"].iloc[-1]),
        "has_onset": result.onset_timestamp_s is not None,
        "onset_timestamp_s": result.onset_timestamp_s or 0.0,
        "has_impact": result.impact_timestamp_s is not None,
        "impact_timestamp_s": result.impact_timestamp_s or 0.0,
    }
    (out_dir / f"{trial_key}_meta.json").write_text(json.dumps(meta, indent=2))


def _clear_previous_export(out_dir: Path) -> None:
    # Also remove Unity's .meta companions so re-curating leaves no orphans.
    for pattern in ("*_timeline.csv", "*_model_log.csv", "*_meta.json", "*_gait_features.csv"):
        for stale in out_dir.glob(pattern):
            stale.unlink()
            meta_companion = stale.with_name(stale.name + ".meta")
            if meta_companion.exists():
                meta_companion.unlink()
    for orphan_meta in out_dir.glob("*.meta"):
        if not orphan_meta.with_name(orphan_meta.stem).exists():
            orphan_meta.unlink()


def main():
    print(f"Output dir: {config.OUTPUT_DIR}\n")

    out_dir = DEFAULT_UNITY_STREAMING_ASSETS
    out_dir.mkdir(parents=True, exist_ok=True)

    timeline_dir = config.TRIAL_TIMELINE_DIR
    model_log_dir = config.MODEL_LOG_DIR

    selected_keys = None
    if curation.DEFAULT_CONFIG_PATH.exists():
        cfg = curation.load_selection_config()
        selected_keys, stats = curation.select_trials_for_unity(timeline_dir, model_log_dir, cfg)
        print(f"Curated selection from {curation.DEFAULT_CONFIG_PATH.name}:")
        print(f"  subjects used: {stats['subjects_used']}")
        for direction, keys in stats["by_direction"].items():
            print(f"  {direction}: {keys}")
        print()
    else:
        print(f"No curation config at {curation.DEFAULT_CONFIG_PATH} — exporting every processed trial.\n")

    _clear_previous_export(out_dir)

    candidate_paths = (
        [timeline_dir / f"{key}.csv" for key in selected_keys]
        if selected_keys is not None
        else sorted(p for p in timeline_dir.glob("*.csv") if not p.name.startswith("_"))
    )

    trial_keys = []
    for timeline_path in candidate_paths:
        if not timeline_path.exists():
            print(f"Skipping {timeline_path.stem}: timeline file missing")
            continue
        trial_key = timeline_path.stem
        model_log_path = model_log_dir / f"{trial_key}_model_log.csv"
        if not model_log_path.exists():
            print(f"Skipping {trial_key}: no model log found (run run_model.py first)")
            continue

        export_trial(trial_key, timeline_path, model_log_path, out_dir)
        trial_keys.append(trial_key)
        print(f"Exported {trial_key} -> {out_dir}")

    (out_dir / "_index.json").write_text(json.dumps({"trials": trial_keys}, indent=2))
    print(f"\nExported {len(trial_keys)} trial(s) to {out_dir}")


if __name__ == "__main__":
    main()
