"""Run the trained real detection model over the digital twin's curated
Unity trials, and write real `_model_log.csv` files for them --
replacing `data_pipeline`'s Phase 4 threshold stub with the actual
trained XGBoost detector from this codebase.

Usage:
    python scripts/export_detection_to_unity.py

Reads the curated trial list from
`../../unity/ShravanAI/Assets/StreamingAssets/Trials/_index.json`
(already written by `data_pipeline/scripts/export_for_unity.py`), maps
each Unity trial key (e.g. "SA06_S06T20R01") to this project's own
harmonized KFall trial, runs it through the trained detection model
(`results/detection_model/xgboost_model.json`), and overwrites that
trial's `_model_log.csv` with the model's real, independent
fall-detected event (if any) -- same CSV schema `data_pipeline`'s stub
already produces (frame_idx, timestamp_s, event, confidence), so
nothing on the Unity/C# side needs to change to read it.

Only ever emits "fall_detected" events -- "pre_impact_predicted" is
prediction's own, written separately by
`export_prediction_to_unity.py`. **Merges into `_model_log.csv` rather
than overwriting it** (drops any prior `fall_detected` row this script
itself wrote, keeps everything else -- e.g. prediction's
`pre_impact_predicted` row -- untouched): this fixes a real bug where
running this script AFTER `export_prediction_to_unity.py` used to
silently erase prediction's already-written event, contradicting that
script's own docstring claim that running order doesn't matter. Now it
genuinely doesn't.

Also writes a single global `_model_stats.json` (not per-trial -- these
are the model's held-out test-set numbers from
`results/detection_model/evaluation_report.json`, the same for every
trial) so the Unity Console can report real accuracy/precision/recall/
false-positive-rate alongside the per-event confidence values, instead
of confidence being the only number visible anywhere.

**Also writes `<trial>_detection_confidence_trace.csv`** per trial --
every window's raw fall-probability (`frame_idx, timestamp_s,
proba_fall`), not just the single threshold-crossing event -- so a live
"confidence gauge" on the Unity dashboard has a real, continuous signal
to animate instead of only a before/after snapshot at the moment of
firing. Native resolution only (one point per real window, 1.0s stride
-- no interpolation/fabrication between real model outputs); the C#
side is expected to hold/lerp between points for display, not this
script.

Run this AFTER `data_pipeline/scripts/export_for_unity.py` (which owns
ground truth: `_timeline.csv`/`_meta.json`/`_index.json`). This script
touches `_model_log.csv`, `_model_stats.json`, and
`_detection_confidence_trace.csv`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from detection.features import FEATURE_NAMES, compute_window_features
from detection.dataset import load_auxiliary_channel, load_window
from detection.model import load_model
from detection.windowing import WindowingConfig
from shared.manifest import load_manifest
from shared.windowing import generate_window_specs

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_UNITY_TRIALS_DIR = (
    REPO_ROOT.parents[1] / "unity" / "ShravanAI" / "Assets" / "StreamingAssets" / "Trials"
)

# "SA06_S06T20R01" -> subject="SA06", task=20, rep="01". This is
# data_pipeline's own trial-key convention (subject_id + "_" +
# reconstructed sensor filename stem) -- mapped here to this project's
# own (subject_id, activity_code, trial_id) manifest identity, e.g.
# ("SA06", "T20", "R01"), matching shared/io/readers_kfall.py's
# TrialMetadata convention exactly.
_UNITY_KEY_RE = re.compile(r"^(SA\d+)_S\d+T(\d+)R(\d+)$", re.IGNORECASE)

# Matches the trained model's own evaluation threshold
# (detection/model.py's evaluate_model uses the same 0.5 cutoff).
FALL_DETECTED_THRESHOLD = 0.5


def _parse_unity_key(key: str) -> tuple[str, str, str]:
    match = _UNITY_KEY_RE.match(key)
    if not match:
        raise ValueError(
            f"Unity trial key {key!r} doesn't match the expected "
            f"'SAxx_SxxTyyRzz' pattern -- can't map it to this project's manifest."
        )
    subject_id, task_str, rep_str = match.groups()
    return subject_id.upper(), f"T{int(task_str):02d}", f"R{rep_str}"


def _compute_window_probas(
    trial_row: pd.Series,
    model,
    windowing_config: WindowingConfig,
) -> list[tuple[int, float, float]]:
    """Run every window of one trial through `model`, in time order,
    returning `(frame_idx, timestamp_s, proba_fall)` for EVERY window --
    the full real signal, not just the first threshold crossing. Shared
    by `_detect_fall_event` (finds the first crossing) and
    `_write_confidence_trace` (writes all of it) so features are only
    ever computed once per trial, not twice.

    `frame_idx` is each window's END frame -- the earliest moment a
    real-time deployment could have made this call, having observed the
    full 2-second window it just classified. This is a genuine, coarser
    detection latency than a per-frame stub would have, not a bug --
    it's what a 2.0s/1.0s windowed classifier actually implies.
    """
    n_samples = round(trial_row["duration_s"] * trial_row["sample_rate_hz"])
    window_specs = generate_window_specs(n_samples, windowing_config)
    if not window_specs:
        return []

    cache: dict = {}
    results: list[tuple[int, float, float]] = []
    for spec in window_specs:
        window_row = pd.Series({
            "harmonized_path": trial_row["harmonized_path"],
            "start_frame": spec.start_frame,
            "end_frame": spec.end_frame,
        })
        window = load_window(window_row, windowing_config.window_length_samples, signal_cache=cache)
        tilt_deg_window = load_auxiliary_channel(
            window_row, "tilt_deg", windowing_config.window_length_samples, signal_cache=cache
        )
        features = compute_window_features(
            window, sample_rate_hz=windowing_config.target_rate_hz, tilt_deg_window=tilt_deg_window
        )
        X = np.array([[features[name] for name in FEATURE_NAMES]], dtype=np.float32)
        proba = float(model.predict_proba(X)[0, 1])

        frame_idx = spec.end_frame - 1
        timestamp_s = frame_idx / windowing_config.target_rate_hz
        results.append((frame_idx, timestamp_s, proba))

    return results


def _detect_fall_event(
    window_probas: list[tuple[int, float, float]], threshold: float
) -> list[tuple[int, float, str, float]]:
    """First window (by time) whose fall probability crosses
    `threshold`, or an empty list if it never does (a real, reportable
    "not detected" outcome, not an error). Takes the already-computed
    per-window probabilities from `_compute_window_probas` rather than
    recomputing them.
    """
    for frame_idx, timestamp_s, proba in window_probas:
        if proba >= threshold:
            return [(frame_idx, timestamp_s, "fall_detected", proba)]
    return []


def _write_confidence_trace(window_probas: list[tuple[int, float, float]], out_path: Path) -> None:
    """Write every window's raw fall-probability -- the full real
    signal a live confidence gauge on the Unity dashboard animates
    against, not just the single moment `threshold` was first crossed.
    """
    pd.DataFrame(window_probas, columns=["frame_idx", "timestamp_s", "proba_fall"]).to_csv(
        out_path, index=False
    )


def _merge_model_log(out_path: Path, new_events: list[tuple[int, float, str, float]]) -> None:
    """Read whatever's already at `out_path` (typically prediction's
    `pre_impact_predicted` row, if `export_prediction_to_unity.py` has
    already run), drop any prior `fall_detected` rows this script itself
    wrote (idempotent re-run), keep everything else, add the fresh
    event(s), write back sorted by `timestamp_s`. Creates the file fresh
    if it doesn't exist yet (safe to run before OR after
    export_prediction_to_unity.py, in either order -- mirrors that
    script's own `_merge_model_log`, kept as a separate copy rather than
    a shared import for the same no-cross-import reason
    `prediction/dataset.py` duplicates rather than imports
    `detection/dataset.py`'s `load_window`)."""
    columns = ["frame_idx", "timestamp_s", "event", "confidence"]
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[existing["event"] != "fall_detected"]
    else:
        existing = pd.DataFrame(columns=columns)

    new_df = pd.DataFrame(new_events, columns=columns)
    combined = pd.concat([existing, new_df], ignore_index=True).sort_values("timestamp_s")
    combined.to_csv(out_path, index=False)


def _write_model_stats(evaluation_report_path: Path, threshold: float, unity_trials_dir: Path) -> None:
    """Derive the Console-facing model-stats summary from the training run's
    own evaluation_report.json. Skips silently (with a printed note, not an
    error) if that file doesn't exist -- e.g. a fresh checkout that hasn't
    run train_detection_model.py yet -- since Unity's loader already
    tolerates a missing _model_stats.json.

    Merges into _model_stats.json rather than overwriting it: preserves an
    existing "prediction" key if scripts/export_prediction_to_unity.py has
    already run (in either order), and writes/replaces only detection's own
    top-level fields."""
    if not evaluation_report_path.exists():
        print(f"No evaluation report at {evaluation_report_path} -- skipping _model_stats.json")
        return

    report = json.loads(evaluation_report_path.read_text())
    tn, fp = report["confusion_matrix"][0]
    fn, tp = report["confusion_matrix"][1]
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    stats_path = unity_trials_dir / "_model_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    stats.update({
        "datasets": report["datasets"],
        "train_subjects": report["train_subjects"],
        "val_subjects": report["val_subjects"],
        "test_subjects": report["test_subjects"],
        "n_test_samples": report["n_test_samples"],
        "n_test_positive": report["n_test_positive"],
        "accuracy": report["accuracy"],
        "precision": report["precision"],
        "recall": report["recall"],
        "sensitivity": report["recall"],  # alias -- same quantity, clearer label for a clinical audience
        "specificity": 1.0 - false_positive_rate,
        "false_positive_rate": false_positive_rate,
        "f1": report["f1"],
        "roc_auc": report["roc_auc"],
        "decision_threshold": threshold,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    })
    stats_path.write_text(json.dumps(stats, indent=2))
    print(
        f"Wrote _model_stats.json: accuracy={stats['accuracy']:.3f} precision={stats['precision']:.3f} "
        f"recall={stats['recall']:.3f} fpr={stats['false_positive_rate']:.3f} roc_auc={stats['roc_auc']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "harmonized" / "manifest.parquet"))
    parser.add_argument(
        "--model", default=str(REPO_ROOT / "results" / "detection_model" / "xgboost_model.json")
    )
    parser.add_argument(
        "--evaluation-report",
        default=str(REPO_ROOT / "results" / "detection_model" / "evaluation_report.json"),
    )
    parser.add_argument("--unity-trials-dir", default=str(DEFAULT_UNITY_TRIALS_DIR))
    parser.add_argument("--threshold", type=float, default=FALL_DETECTED_THRESHOLD)
    args = parser.parse_args()

    unity_trials_dir = Path(args.unity_trials_dir)
    index_path = unity_trials_dir / "_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found -- run data_pipeline/scripts/export_for_unity.py "
            f"first to produce the curated trial list this script reads."
        )
    trial_keys = json.loads(index_path.read_text())["trials"]
    print(f"Found {len(trial_keys)} curated Unity trial(s) in {index_path}")

    print(f"Loading harmonized manifest from {args.manifest}...")
    kfall_df = load_manifest(args.manifest)
    kfall_df = kfall_df[kfall_df["dataset"] == "kfall"]

    print(f"Loading trained detection model from {args.model}...")
    model = load_model(args.model)
    windowing_config = WindowingConfig()

    n_detected, n_not_detected, n_missing = 0, 0, 0
    for key in trial_keys:
        subject_id, activity_code, trial_id = _parse_unity_key(key)
        match = kfall_df[
            (kfall_df["subject_id"] == subject_id)
            & (kfall_df["activity_code"] == activity_code)
            & (kfall_df["trial_id"] == trial_id)
        ]
        if len(match) == 0:
            print(f"  {key}: SKIPPED -- no matching harmonized trial "
                  f"({subject_id}/{activity_code}/{trial_id}) found in this project's manifest")
            n_missing += 1
            continue

        trial_row = match.iloc[0]
        window_probas = _compute_window_probas(trial_row, model, windowing_config)
        events = _detect_fall_event(window_probas, args.threshold)

        out_path = unity_trials_dir / f"{key}_model_log.csv"
        _merge_model_log(out_path, events)

        trace_path = unity_trials_dir / f"{key}_detection_confidence_trace.csv"
        _write_confidence_trace(window_probas, trace_path)

        if events:
            _, timestamp_s, _, confidence = events[0]
            print(f"  {key}: fall_detected at t={timestamp_s:.2f}s (confidence={confidence:.2f})")
            n_detected += 1
        else:
            print(f"  {key}: no fall detected (model never crossed threshold={args.threshold})")
            n_not_detected += 1

    print(f"\nDone: {n_detected} detected, {n_not_detected} not detected, {n_missing} skipped (no manifest match)")
    print(f"Wrote real _model_log.csv files to {unity_trials_dir}")

    _write_model_stats(Path(args.evaluation_report), args.threshold, unity_trials_dir)


if __name__ == "__main__":
    main()
