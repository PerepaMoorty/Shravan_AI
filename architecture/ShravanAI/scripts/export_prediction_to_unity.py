"""Run the trained real prediction model over the digital twin's curated
Unity trials, and write real `pre_impact_predicted` events into each
trial's `_model_log.csv` -- adding the prediction stream alongside
detection's `fall_detected` events, without erasing them.

Usage:
    python scripts/export_prediction_to_unity.py
    python scripts/export_prediction_to_unity.py --architecture convlstm

**Requires a trained checkpoint to already exist**
(`results/prediction_model/{convlstm,tiny_transformer}_model.pt` +
`evaluation_report.json`, produced by `scripts/train_prediction_model.py`).
Verified against the real checkpoints: 9/9 curated fall trials fire,
0/3 false alarms, ~1.4s mean lead time.

Reads the curated trial list from
`../../unity/ShravanAI/Assets/StreamingAssets/Trials/_index.json` (already
written by `data_pipeline/scripts/export_for_unity.py`), maps each Unity
trial key to this project's own harmonized KFall trial (prediction is
KFall-only -- `shared.manifest.query_prediction_trials` already enforces
this -- so a curated trial with no KFall match, e.g. if a SisFall trial
were ever curated, is skipped with a clear message, not an error).

**Which checkpoint gets used**: if both `convlstm_model.pt` and
`tiny_transformer_model.pt` exist, the one with the higher `test_macro_f1`
in `evaluation_report.json` is used automatically; `--architecture` forces
a specific one instead (useful if only one was trained, or to compare).

**Event selection -- uses a tuned probability THRESHOLD
(`--pre-impact-threshold`, default 0.80), not the model's raw argmax
label.** Real numbers from a held-out-test-set event-level sweep
(2026-07-22, convlstm, trained on the corrected KFall ground truth -- see
`shared/io/readers_kfall.py`'s `FALL_TASK_IDS` docstring for the
label-parser bug fixed that session, which also dramatically improved
this checkpoint's raw quality): the raw argmax label (an implicit
threshold of ~0.34) fires a false pre_impact alarm often -- an initial
same-session estimate of 86% could not be reproduced on a later, more
careful measurement; `docs/COMPREHENSIVE_STATISTICAL_EVALUATION.md`'s
rigorous event-level figure is 62.6% (275/439 ADL test trials), and
supersedes the 86% figure -- unusable either way for a demo whose whole
point is "does it ever cry wolf." Judged at a stricter, explicit threshold instead, the model's
probabilities carry real, strong signal: 0.80 cuts the false-alarm rate
to ~10% while still catching ~94% of real falls (~1.5s mean lead time).
There is no threshold in the swept range that gives both a zero
false-alarm rate AND 100% detection with this checkpoint -- picking an
operating point is a real, disclosed tradeoff, not a bug to fix. See
`--pre-impact-threshold`'s CLI help for the fuller sweep and how to pick
a different point on that curve. The event fires at the FIRST window (by
`start_frame`) whose
`proba_pre_impact` clears the threshold, using `start_frame` itself (not
`end_frame - 1`, unlike `export_detection_to_unity.py`) so the timestamp
Unity displays is the true first-warning moment. For fall trials, only
windows with `start_frame < impact_frame` are eligible (a post-impact
`pre_impact` prediction isn't advance warning, it's noise after the
fact). For ADL trials (no impact frame), any firing is eligible -- and
would be a genuine false alarm, worth showing honestly if it happens,
consistent with this demo's whole "does it ever cry wolf" framing. The
`_model_stats.json` lead-time summary is derived from these SAME
threshold-based events (not `prediction/lead_time.py`'s raw-argmax
`compute_lead_time_ms`, which `scripts/train_prediction_model.py` still
uses for its own report -- that's a legitimate, separate number
describing the raw model, not this demo's tuned operating point) so the
Console's reported detection-rate/lead-time always matches what actually
fires in the Unity trials.

Deliberately does NOT apply EMA/hysteresis smoothing, which is meant for a genuinely live, frame-by-frame streaming
display, where smoothing prevents a single noisy window from visibly
flip-flopping an on-screen state moment to moment. This script instead
precomputes one offline event timestamp from a dense array already fully
in hand. (A held-out sweep of the hysteresis approach was tried as an
alternative to a flat threshold -- 2026-07-21 -- and landed in
similar-to-slightly-better territory on the same tradeoff curve, not a
qualitatively different result, so the simpler flat threshold was kept.)

Merges into `_model_log.csv` rather than overwriting it: reads whatever's
already there (typically `export_detection_to_unity.py`'s `fall_detected`
row, if that already ran), drops any of ITS OWN prior
`pre_impact_predicted` rows (idempotent re-run), keeps every other row
untouched, adds its fresh event (if any), and writes back sorted by
`timestamp_s`. Safe to run before OR after `export_detection_to_unity.py`,
in either order.

Also merges (not overwrites) `_model_stats.json`: preserves detection's
top-level fields if already present, and writes/replaces only a nested
"prediction" key -- see that key's shape below.

**Also writes `<trial>_prediction_confidence_trace.csv`** per trial --
every dense window's full 3-class probability
(`frame_idx, timestamp_s, proba_non_fall, proba_pre_impact, proba_fall`),
not just the single moment `--pre-impact-threshold` was first crossed --
so a live "confidence gauge" on the Unity dashboard has a real,
continuous signal to animate. Native resolution only (one point per
real window, 0.1s dense stride -- no interpolation/fabrication between
real model outputs); the C# side holds/lerps between points for
display, not this script.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from prediction.features import load_augmented_window
from prediction.events import first_pre_impact_event
from prediction.labelers import FALL, NON_FALL, PRE_IMPACT
from prediction.lead_time import summarize_lead_times
from prediction.predict import load_prediction_model, predict_windows_batch
from prediction.windowing import PredictionWindowingConfig, generate_window_specs
from shared.manifest import load_manifest

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_UNITY_TRIALS_DIR = (
    REPO_ROOT.parents[1] / "unity" / "ShravanAI" / "Assets" / "StreamingAssets" / "Trials"
)
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "prediction_model"

# Same Unity-trial-key convention as export_detection_to_unity.py:
# "SA06_S06T20R01" -> subject="SA06", task=20, rep="01". Duplicated rather
# than imported from that script -- importing another script module is
# fragile, and this is a small enough helper that duplication (the same
# tradeoff prediction/dataset.py already makes vs. detection/dataset.py)
# is the more robust choice.
_UNITY_KEY_RE = re.compile(r"^(SA\d+)_S\d+T(\d+)R(\d+)$", re.IGNORECASE)


def _parse_unity_key(key: str) -> tuple[str, str, str]:
    match = _UNITY_KEY_RE.match(key)
    if not match:
        raise ValueError(
            f"Unity trial key {key!r} doesn't match the expected "
            f"'SAxx_SxxTyyRzz' pattern -- can't map it to this project's manifest."
        )
    subject_id, task_str, rep_str = match.groups()
    return subject_id.upper(), f"T{int(task_str):02d}", f"R{rep_str}"


def _pick_best_architecture(evaluation_report: dict) -> str:
    architectures = evaluation_report["architectures"]
    best = max(architectures, key=lambda a: a["test_macro_f1"])
    return best["architecture"]


def _predict_trial_windows(
    model, trial_row: pd.Series, windowing_config: PredictionWindowingConfig,
) -> pd.DataFrame:
    """Every dense window of one trial, in start_frame order, with the
    model's predicted label + ALL THREE per-class probabilities (not
    just proba_pre_impact -- kept alongside proba_non_fall/proba_fall so
    callers that want the full 3-way confidence trace, e.g. a live
    dashboard gauge, don't need a second inference pass). Empty
    DataFrame (not an error) if the trial has zero real samples."""
    n_samples = round(trial_row["duration_s"] * trial_row["sample_rate_hz"])
    window_specs = generate_window_specs(n_samples, windowing_config)
    if not window_specs:
        return pd.DataFrame(columns=[
            "start_frame", "predicted_label", "proba_non_fall", "proba_pre_impact", "proba_fall",
        ])

    cache: dict = {}
    windows = np.stack([
        load_augmented_window(
            pd.Series({
                "harmonized_path": trial_row["harmonized_path"],
                "start_frame": spec.start_frame,
                "end_frame": spec.end_frame,
            }),
            windowing_config.window_length_samples,
            signal_cache=cache,
        )
        for spec in window_specs
    ])  # (n_windows, window_length, 9)

    predictions = predict_windows_batch(model, windows)
    return pd.DataFrame({
        "start_frame": [spec.start_frame for spec in window_specs],
        "predicted_label": [p["predicted_label"] for p in predictions],
        "proba_non_fall": [p["proba_non_fall"] for p in predictions],
        "proba_pre_impact": [p["proba_pre_impact"] for p in predictions],
        "proba_fall": [p["proba_fall"] for p in predictions],
    })


def _write_confidence_trace(predictions_df: pd.DataFrame, target_rate_hz: float, out_path: Path) -> None:
    """Write every window's full 3-class probability trace -- the real
    signal a live confidence gauge on the Unity dashboard animates
    against, not just the single moment the decision threshold was
    first crossed. Native resolution only (one point per real window,
    0.1s dense stride -- no interpolation/fabrication); the C# side
    holds/lerps between points for display."""
    trace = predictions_df[["start_frame", "proba_non_fall", "proba_pre_impact", "proba_fall"]].copy()
    trace["timestamp_s"] = trace["start_frame"] / target_rate_hz
    trace = trace.rename(columns={"start_frame": "frame_idx"})
    trace[["frame_idx", "timestamp_s", "proba_non_fall", "proba_pre_impact", "proba_fall"]].to_csv(
        out_path, index=False
    )


def _merge_model_log(out_path: Path, new_events: list[tuple[int, float, str, float]]) -> None:
    """Read whatever's already at `out_path` (typically detection's
    fall_detected row), drop any prior pre_impact_predicted rows (this
    script's own -- idempotent re-run), keep everything else, add the
    fresh event(s), write back sorted by timestamp_s. Creates the file
    fresh if it doesn't exist yet (safe to run before detection's script)."""
    columns = ["frame_idx", "timestamp_s", "event", "confidence"]
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[existing["event"] != "pre_impact_predicted"]
    else:
        existing = pd.DataFrame(columns=columns)

    new_df = pd.DataFrame(new_events, columns=columns)
    combined = pd.concat([existing, new_df], ignore_index=True).sort_values("timestamp_s")
    combined.to_csv(out_path, index=False)


def _confusion_matrix_fields(cm: list[list[int]]) -> dict:
    """Flatten a 3x3 confusion matrix into explicitly named ints -- Unity's
    JsonUtility can't deserialize a jagged array-of-arrays, only plain
    fields (same reason detection's _model_stats.json uses named
    tn/fp/fn/tp rather than a nested 2x2 array)."""
    labels = [NON_FALL, PRE_IMPACT, FALL]
    fields = {}
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            fields[f"true_{true_label}_pred_{pred_label}"] = cm[i][j]
    return fields


def _per_class_metrics(cm: list[list[int]]) -> dict:
    """Precision/recall/F1 per class, derived directly from the
    confusion matrix (rather than re-parsing sklearn's text report,
    which isn't easily machine-readable) -- row i = true class i,
    column j = predicted class j."""
    cm_arr = np.array(cm, dtype=float)
    labels = [NON_FALL, PRE_IMPACT, FALL]
    per_class = {}
    for i, label in enumerate(labels):
        tp = cm_arr[i, i]
        predicted_positive = cm_arr[:, i].sum()
        actual_positive = cm_arr[i, :].sum()
        precision = float(tp / predicted_positive) if predicted_positive > 0 else 0.0
        recall = float(tp / actual_positive) if actual_positive > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    return per_class


def _write_model_stats(
    unity_trials_dir: Path,
    architecture: str,
    arch_report: dict,
    lead_summary,
    pre_impact_threshold: float,
) -> None:
    """Merges into _model_stats.json rather than overwriting: preserves
    detection's top-level fields if the file already exists (from
    export_detection_to_unity.py having run, in either order), and
    writes/replaces only the nested "prediction" key."""
    stats_path = unity_trials_dir / "_model_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    cm = arch_report["test_confusion_matrix"]
    stats["prediction"] = {
        "available": True,
        "architecture": architecture,
        "test_macro_f1": arch_report["test_macro_f1"],
        "pre_impact_decision_threshold": pre_impact_threshold,
        "confusion_matrix": _confusion_matrix_fields(cm),
        "per_class": _per_class_metrics(cm),
        "lead_time": None if lead_summary is None else {
            "n_trials": lead_summary.n_trials,
            "n_flagged": lead_summary.n_flagged,
            "detection_rate": lead_summary.detection_rate,
            "mean_lead_time_ms": lead_summary.mean_lead_time_ms,
            "median_lead_time_ms": lead_summary.median_lead_time_ms,
        },
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote prediction stats into {stats_path} (architecture={architecture})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "harmonized" / "manifest.parquet"))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--architecture", choices=["convlstm", "tiny_transformer"], default=None)
    parser.add_argument("--unity-trials-dir", default=str(DEFAULT_UNITY_TRIALS_DIR))
    parser.add_argument(
        "--pre-impact-threshold", type=float, default=0.80,
        help="Minimum proba_pre_impact required to fire a pre_impact_predicted event, evaluated "
             "at the EVENT level (one fire-or-not per trial), NOT the model's raw argmax label. "
             "Default (0.80) comes from a real held-out-test-set sweep (2026-07-22, convlstm, "
             "trained on the corrected KFall ground truth): raw argmax (~0.34 implicit threshold) "
             "fires a false alarm often (an initial 86%% estimate was not reproducible; "
             "docs/COMPREHENSIVE_STATISTICAL_EVALUATION.md's rigorous figure is 62.6%%) -- either "
             "way unusable for a demo about not crying wolf. 0.80 cuts that to ~10%% while still catching ~94%% of real falls "
             "(~1.5s mean lead time). Lower = more sensitive/more false alarms; higher = "
             "quieter/more missed falls -- there is no threshold that gives both a zero "
             "false-alarm rate and 100%% detection with this checkpoint; picking one is a real "
             "tradeoff, not a bug to fix.",
    )
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

    results_dir = Path(args.results_dir)
    report_path = results_dir / "evaluation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"{report_path} not found -- run scripts/train_prediction_model.py "
            f"first to produce a trained checkpoint + evaluation report. No "
            f"prediction model exists yet as of this script being written -- "
            f"see this script's module docstring for the pyarrow blocker "
            f"that currently prevents training from running at all."
        )
    evaluation_report = json.loads(report_path.read_text())

    architecture = args.architecture or _pick_best_architecture(evaluation_report)
    arch_report = next(a for a in evaluation_report["architectures"] if a["architecture"] == architecture)
    model_path = Path(arch_report["model_path"])
    print(f"Using architecture={architecture} (test_macro_f1={arch_report['test_macro_f1']:.3f}) "
          f"from {model_path}")
    model = load_prediction_model(model_path)

    print(f"Loading harmonized manifest from {args.manifest}...")
    kfall_df = load_manifest(args.manifest)
    kfall_df = kfall_df[kfall_df["dataset"] == "kfall"]

    windowing_config = PredictionWindowingConfig()

    n_fired, n_silent, n_missing = 0, 0, 0
    lead_times: list[float | None] = []

    for key in trial_keys:
        subject_id, activity_code, trial_id = _parse_unity_key(key)
        match = kfall_df[
            (kfall_df["subject_id"] == subject_id)
            & (kfall_df["activity_code"] == activity_code)
            & (kfall_df["trial_id"] == trial_id)
        ]
        if len(match) == 0:
            print(f"  {key}: SKIPPED -- no matching KFall trial in this project's manifest "
                  f"(prediction is KFall-only)")
            n_missing += 1
            continue

        trial_row = match.iloc[0]
        impact_frame = (
            int(trial_row["fall_impact_frame"])
            if pd.notna(trial_row.get("fall_impact_frame"))
            else None
        )

        predictions_df = _predict_trial_windows(model, trial_row, windowing_config)
        events = first_pre_impact_event(
            predictions_df, impact_frame, windowing_config.target_rate_hz, args.pre_impact_threshold
        )

        out_path = unity_trials_dir / f"{key}_model_log.csv"
        _merge_model_log(out_path, events)

        trace_path = unity_trials_dir / f"{key}_prediction_confidence_trace.csv"
        _write_confidence_trace(predictions_df, windowing_config.target_rate_hz, trace_path)

        if impact_frame is not None:
            # Derived directly from `events` (already computed above using
            # `args.pre_impact_threshold`), NOT via compute_lead_time_ms's
            # raw-argmax definition -- so the headline lead-time stat
            # reported in _model_stats.json always matches what actually
            # fires in the Unity demo, even though the threshold differs
            # from what train_prediction_model.py itself reports (that
            # script's own lead-time number legitimately stays argmax-based,
            # since it's evaluating the raw model, not this demo's tuned
            # operating point).
            if events:
                fired_frame = events[0][0]
                lead_times.append((impact_frame - fired_frame) / windowing_config.target_rate_hz * 1000.0)
            else:
                lead_times.append(None)

        if events:
            _, timestamp_s, _, confidence = events[0]
            print(f"  {key}: pre_impact_predicted at t={timestamp_s:.2f}s (confidence={confidence:.2f})")
            n_fired += 1
        else:
            print(f"  {key}: no pre_impact prediction (correctly silent, or a miss -- see lead-time below)")
            n_silent += 1

    print(f"\nDone: {n_fired} fired, {n_silent} silent, {n_missing} skipped (no manifest match)")
    print(f"Wrote pre_impact_predicted events into {unity_trials_dir}")

    lead_summary = summarize_lead_times(lead_times) if lead_times else None
    if lead_summary:
        print(f"Lead time (curated demo trials only, n={lead_summary.n_trials}): "
              f"detection_rate={lead_summary.detection_rate:.2f} "
              f"mean={lead_summary.mean_lead_time_ms}ms median={lead_summary.median_lead_time_ms}ms")

    _write_model_stats(unity_trials_dir, architecture, arch_report, lead_summary, args.pre_impact_threshold)


if __name__ == "__main__":
    main()
