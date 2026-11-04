"""Choose the pre-impact alarm threshold on VALIDATION subjects only, then
evaluate it once on the locked TEST subjects (peer review item W4).

Why: the production threshold (0.80) was picked by sweeping the test set,
so the event-level figures reported at it (93.8% detection, 9.6% false
alarms, 1,458 ms mean lead time) are optimistically biased. This script
replaces that with a protocol where the test subjects play no part in any
choice:

  1. Rebuild the exact train/val/test subject split used for training
     (same windowing config, same seed) and verify it against the split
     sizes recorded in `evaluation_report.json`.
  2. Run every trained checkpoint over every dense window of every val
     and test trial; cache the per-window probabilities to parquet.
  3. Sweep thresholds on VAL only and apply a rule fixed before looking
     at any numbers (see `select_threshold`):
       maximise event-level detection rate subject to
       false-alarm rate <= --max-false-alarm-rate (default 0.10);
       ties go to the higher threshold.
     The production architecture is also chosen on VAL (see
     `select_architecture`), not by test macro-F1.
  4. Evaluate each architecture once on TEST at its locked threshold,
     with Wilson 95% intervals. The old test-tuned 0.80 numbers are
     reported alongside, labelled as such, to quantify the optimism.

Event definitions come from `prediction.events` -- the same code the
Unity export uses -- so the numbers are directly comparable.

Usage:
    python scripts/select_threshold_on_validation.py
    python scripts/select_threshold_on_validation.py --reuse-probabilities   # skip inference
    python scripts/select_threshold_on_validation.py --max-false-alarm-rate 0.05
    python scripts/select_threshold_on_validation.py \
        --model convlstm=results/prediction_model_convlstm_retrain/evaluation_report.json

Writes to results/prediction_model/threshold_selection/:
    window_probabilities.parquet   per-window probabilities, val + test, every architecture
    threshold_selection_report.json
    sweep_<architecture>.png       val sweep (chosen point) + test curve (locked point)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prediction.dataset import build_windows_manifest
from prediction.events import (
    EventLevelResult,
    TrialPredictions,
    evaluate_event_level,
    select_architecture,
    select_threshold,
    sweep_thresholds,
    wilson_interval,
)
from prediction.labelers import LABEL_TO_INT
from prediction.predict import load_prediction_model
from prediction.split import PredictionSplitConfig, split_by_subject
from prediction.torch_dataset import PredictionWindowDataset
from prediction.windowing import PredictionWindowingConfig
from shared.manifest import load_manifest

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "prediction_model"
PREVIOUS_TEST_TUNED_THRESHOLD = 0.80
TRIAL_KEY_COLS = ["dataset", "subject_id", "activity_code", "trial_id"]
PROBA_NAMES = ["non_fall", "pre_impact", "fall"]


# ---------------------------------------------------------------- inference

def _run_inference(models: dict[str, torch.nn.Module], windows_df: pd.DataFrame,
                   windowing_config: PredictionWindowingConfig, device: torch.device,
                   batch_size: int, split_name: str) -> pd.DataFrame:
    """One pass over `windows_df`; every model sees each loaded batch, so
    the expensive window loading happens once regardless of model count."""
    ds = PredictionWindowDataset(windows_df, windowing_config.window_length_samples,
                                 windowing_config.target_rate_hz)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probas = {a: [] for a in models}
    t0, n_batches = time.time(), len(loader)
    with torch.no_grad():
        for i, (x, _) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            for arch, model in models.items():
                probas[arch].append(F.softmax(model(x), dim=1).cpu().numpy())
            if i % 20 == 0 or i == n_batches:
                print(f"    [{split_name}] batch {i}/{n_batches}  ({time.time() - t0:.0f}s elapsed)", flush=True)

    out = windows_df[TRIAL_KEY_COLS + ["global_subject_id", "window_index", "start_frame",
                                       "label", "fall_impact_frame"]].copy()
    out["split"] = split_name
    for arch, chunks in probas.items():
        p = np.concatenate(chunks)
        for name in PROBA_NAMES:
            out[f"{arch}__proba_{name}"] = p[:, LABEL_TO_INT[name]]
    return out


def _to_trials(probs_df: pd.DataFrame, architecture: str) -> list[TrialPredictions]:
    trials = []
    for key, g in probs_df.groupby(TRIAL_KEY_COLS, sort=True):
        g = g.sort_values("start_frame")
        impact = g["fall_impact_frame"].iloc[0]
        trials.append(TrialPredictions(
            trial_key=tuple(key),
            subject_id=str(g["global_subject_id"].iloc[0]),
            start_frames=g["start_frame"].to_numpy(),
            proba_pre_impact=g[f"{architecture}__proba_pre_impact"].to_numpy(),
            impact_frame=None if pd.isna(impact) else int(impact),
        ))
    return trials


# ---------------------------------------------------------------- reporting

def _result_dict(r: EventLevelResult) -> dict:
    d = asdict(r)
    d["detection_rate_ci95"] = wilson_interval(r.n_fall_warned, r.n_fall_trials)
    d["false_alarm_rate_ci95"] = wilson_interval(r.n_adl_false_alarm, r.n_adl_trials)
    return d


def _fmt(r: EventLevelResult) -> str:
    det_ci = wilson_interval(r.n_fall_warned, r.n_fall_trials)
    far_ci = wilson_interval(r.n_adl_false_alarm, r.n_adl_trials)
    lead = f"{r.mean_lead_time_ms:.0f}/{r.median_lead_time_ms:.0f} ms" if r.mean_lead_time_ms is not None else "n/a"
    return (f"thr={r.threshold:.2f}  detection={r.n_fall_warned}/{r.n_fall_trials}={r.detection_rate:.3f} "
            f"[{det_ci[0]:.3f}-{det_ci[1]:.3f}]  false_alarm={r.n_adl_false_alarm}/{r.n_adl_trials}="
            f"{r.false_alarm_rate:.3f} [{far_ci[0]:.3f}-{far_ci[1]:.3f}]  lead mean/median={lead}")


def _plot(arch: str, val_sweep, test_sweep, chosen: EventLevelResult, test_locked: EventLevelResult,
          cap: float, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, sweep, title in ((axes[0], val_sweep, "Validation (threshold chosen here)"),
                             (axes[1], test_sweep, "Test (threshold locked)")):
        thr = [r.threshold for r in sweep]
        ax.plot(thr, [r.detection_rate for r in sweep], label="Detection rate (falls warned)")
        ax.plot(thr, [r.false_alarm_rate for r in sweep], label="False-alarm rate (ADL trials)")
        ax.axvline(chosen.threshold, color="k", ls="--", lw=1, label=f"Selected = {chosen.threshold:.2f}")
        ax.set_title(title)
        ax.set_xlabel("pre_impact probability threshold")
        ax.grid(alpha=0.3)
    axes[0].axhline(cap, color="grey", ls=":", lw=1, label=f"False-alarm cap = {cap:.0%}")
    axes[1].axvline(PREVIOUS_TEST_TUNED_THRESHOLD, color="tab:red", ls=":", lw=1,
                    label=f"Previous production = {PREVIOUS_TEST_TUNED_THRESHOLD:.2f}")
    axes[1].plot([test_locked.threshold] * 2, [test_locked.detection_rate, test_locked.false_alarm_rate], "ko")
    axes[0].set_ylabel("Rate")
    for ax in axes:
        ax.legend(fontsize=8, loc="center left")
    fig.suptitle(f"{arch}: event-level threshold selection")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _resolve_model_sources(overrides: list[str] | None, results_dir: Path) -> dict[str, tuple[dict, dict]]:
    """{architecture: (its evaluation report, its entry in that report)}.
    Defaults to every architecture in <results_dir>/evaluation_report.json;
    `--model ARCH=REPORT` entries replace or add single architectures.
    Every checkpoint file must exist."""
    def entries(report_path: Path) -> dict[str, tuple[dict, dict]]:
        report = json.loads(Path(report_path).read_text())
        return {a["architecture"]: (report, a) for a in report["architectures"]}

    sources = entries(results_dir / "evaluation_report.json")
    for item in overrides or []:
        arch, _, report_path = item.partition("=")
        found = entries(Path(report_path))
        if arch not in found:
            raise ValueError(f"--model {item}: {report_path} has no {arch!r} entry (has {sorted(found)})")
        sources[arch] = found[arch]
    missing = [a for a, (_, e) in sources.items() if not Path(e["model_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoint file missing for {missing}; point them at a valid report with --model.")
    return sources


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "harmonized" / "manifest.parquet"))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--max-false-alarm-rate", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--model", action="append", metavar="ARCH=REPORT_JSON",
                        help="Take ARCH's checkpoint from the given evaluation_report.json instead of "
                             "<results-dir>/evaluation_report.json. Repeatable. Architectures not named "
                             "here still come from <results-dir>/evaluation_report.json.")
    parser.add_argument("--reuse-probabilities", action="store_true",
                        help="Skip inference and reuse window_probabilities.parquet from a previous run.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir / "threshold_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    probs_path = out_dir / "window_probabilities.parquet"

    sources = _resolve_model_sources(args.model, results_dir)
    architectures = list(sources)
    for arch, (report, entry) in sources.items():
        print(f"  {arch}: {entry['model_path']}")
    windowing_config = PredictionWindowingConfig()

    if args.reuse_probabilities:
        print(f"Reusing cached probabilities from {probs_path}")
        probs_df = pd.read_parquet(probs_path)
    else:
        print("Rebuilding dense windows and the subject split used for training...")
        windows_df = build_windows_manifest(load_manifest(args.manifest), config=windowing_config)
        train_df, val_df, test_df = split_by_subject(windows_df, PredictionSplitConfig())
        actual = (len(train_df), len(val_df), len(test_df))
        for arch, (report, _) in sources.items():
            expected = (report["train_size"], report["val_size"], report["test_size"])
            if actual != expected:
                raise RuntimeError(f"Split does not match the one {arch} was trained on: "
                                   f"{actual} vs its report {expected}. Refusing to continue.")
        print(f"  split matches training: train/val/test windows = {actual}")
        print(f"  val subjects:  {sorted(val_df['global_subject_id'].unique())}")
        print(f"  test subjects: {sorted(test_df['global_subject_id'].unique())}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running inference on {device} for {architectures}...")
        models = {}
        for arch, (_, entry) in sources.items():
            m = load_prediction_model(Path(entry["model_path"])).to(device)
            m.eval()
            models[arch] = m
        probs_df = pd.concat([
            _run_inference(models, val_df, windowing_config, device, args.batch_size, "val"),
            _run_inference(models, test_df, windowing_config, device, args.batch_size, "test"),
        ], ignore_index=True)
        probs_df.to_parquet(probs_path, index=False)
        print(f"  cached per-window probabilities to {probs_path}")

    thresholds = np.round(np.arange(0.05, 0.9901, 0.01), 2)
    rate = windowing_config.target_rate_hz
    report = {
        "protocol": {
            "selection_split": "val", "evaluation_split": "test",
            "rule": "maximise val event-level detection rate subject to val false-alarm rate <= cap; "
                    "ties -> higher threshold; if no threshold meets the cap -> lowest false-alarm rate",
            "architecture_rule": "highest val detection rate at selected threshold, then lowest val "
                                 "false-alarm rate, then longest val mean lead time",
            "max_false_alarm_rate": args.max_false_alarm_rate,
            "threshold_grid": [float(thresholds[0]), float(thresholds[-1]), 0.01],
            "previous_test_tuned_threshold": PREVIOUS_TEST_TUNED_THRESHOLD,
        },
        "subjects": {s: sorted(probs_df.loc[probs_df["split"] == s, "global_subject_id"].unique())
                     for s in ("val", "test")},
        "architectures": {},
    }

    val_choices = {}
    for arch in architectures:
        val_trials = _to_trials(probs_df[probs_df["split"] == "val"], arch)
        test_trials = _to_trials(probs_df[probs_df["split"] == "test"], arch)
        val_sweep = sweep_thresholds(val_trials, thresholds, rate)
        test_sweep = sweep_thresholds(test_trials, thresholds, rate)

        chosen, fallback = select_threshold(val_sweep, args.max_false_alarm_rate)
        val_choices[arch] = chosen
        test_locked = evaluate_event_level(test_trials, chosen.threshold, rate)
        test_previous = evaluate_event_level(test_trials, PREVIOUS_TEST_TUNED_THRESHOLD, rate)

        print(f"\n=== {arch} ===")
        print(f"  VAL  selected   : {_fmt(chosen)}" + ("   (FALLBACK: no threshold met the cap)" if fallback else ""))
        print(f"  TEST locked     : {_fmt(test_locked)}")
        print(f"  TEST at 0.80    : {_fmt(test_previous)}   <- previous production threshold (originally tuned on test)")

        _plot(arch, val_sweep, test_sweep, chosen, test_locked, args.max_false_alarm_rate,
              out_dir / f"sweep_{arch}.png")
        report["architectures"][arch] = {
            "selected_threshold": chosen.threshold,
            "used_fallback": fallback,
            "val_at_selected": _result_dict(chosen),
            "test_at_selected_locked": _result_dict(test_locked),
            "test_at_previous_production_0.80": _result_dict(test_previous),
            "val_sweep": [asdict(r) for r in val_sweep],
            "test_sweep": [asdict(r) for r in test_sweep],
        }

    production = select_architecture(val_choices)
    report["production_architecture_selected_on_val"] = production
    print(f"\nProduction architecture (chosen on val): {production} at threshold "
          f"{val_choices[production].threshold:.2f}")

    report_path = out_dir / "threshold_selection_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report: {report_path}\nPlots:  {out_dir}/sweep_<architecture>.png")


if __name__ == "__main__":
    main()
