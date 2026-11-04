"""Subject-grouped k-fold cross-validation of the prediction stream (peer
review item W3), with the W4 threshold protocol applied inside every fold.

Per fold:
  test subjects  = one fold of a seeded partition of the 32 KFall subjects
  val subjects   = --n-val-subjects chosen (seeded) from the rest
  train subjects = everything else
  1. Train each architecture with the production recipe
     (`train_prediction_model.train_one_architecture`: focal loss,
     pre_impact_boost 1.0, early stopping on val loss).
  2. Choose the alarm threshold on that fold's VAL subjects
     (`prediction.events.select_threshold`, max detection s.t. false-alarm
     rate <= cap) and evaluate it once on that fold's TEST subjects.
Every subject is a test subject exactly once, so pooled results cover all
32 subjects, and the fold-to-fold spread estimates run/split variance.

Speed: every dense window is augmented once up front into a memory-mapped
array (results/cv_prediction_cache/, reused across folds, architectures
and restarts) instead of being re-loaded every epoch. Same window
function, same tensors -- only faster.

Resumable: each (fold, architecture) writes <output-dir>/fold_k/<arch>_result.json
when it finishes; re-running skips finished pairs. Progress is written to
<output-dir>/progress.json.

Usage:
    python scripts/cross_validate_prediction.py                       # full run
    python scripts/cross_validate_prediction.py --aggregate-only      # rebuild summary
    python scripts/cross_validate_prediction.py --subjects-limit 8 --n-folds 2 \
        --n-val-subjects 2 --max-epochs 1 --output-dir results/cv_prediction_smoke   # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from select_threshold_on_validation import TRIAL_KEY_COLS, _to_trials  # noqa: E402
from train_prediction_model import train_one_architecture  # noqa: E402

from prediction.dataset import build_windows_manifest  # noqa: E402
from prediction.events import (  # noqa: E402
    evaluate_event_level, select_threshold, sweep_thresholds, wilson_interval,
)
from prediction.features import load_augmented_window  # noqa: E402
from prediction.labelers import LABEL_TO_INT  # noqa: E402
from prediction.loso import make_subject_kfolds, pick_validation_subjects  # noqa: E402
from prediction.predict import load_prediction_model  # noqa: E402
from prediction.torch_dataset import PrecomputedWindowDataset  # noqa: E402
from prediction.windowing import PredictionWindowingConfig  # noqa: E402
from shared.manifest import load_manifest  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
THRESHOLDS = np.round(np.arange(0.05, 0.9901, 0.01), 2)
ARCHITECTURES = ["convlstm", "tiny_transformer"]


# ------------------------------------------------------------------ window cache

def _fingerprint(windows_df: pd.DataFrame) -> str:
    h = hashlib.sha1()
    h.update(windows_df["harmonized_path"].astype(str).str.cat(sep="|").encode())
    h.update(windows_df["start_frame"].to_numpy().tobytes())
    return h.hexdigest()


def build_or_load_window_cache(windows_df: pd.DataFrame, cfg: PredictionWindowingConfig, cache_dir: Path) -> np.ndarray:
    """(n_windows, 9, window_length) float32 memmap of augmented windows,
    row i = windows_df row i. Rebuilt only if the window list changed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path, meta_path = cache_dir / "windows_augmented.npy", cache_dir / "windows_augmented.meta.json"
    meta = {"n_windows": len(windows_df), "window_length": cfg.window_length_samples,
            "fingerprint": _fingerprint(windows_df)}
    if path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == meta:
        print(f"Reusing window cache {path}")
        return np.load(path, mmap_mode="r")

    print(f"Building window cache for {len(windows_df):,} windows (one-off, a few minutes)...")
    tmp = cache_dir / "windows_augmented.tmp.npy"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                    shape=(len(windows_df), 9, cfg.window_length_samples))
    cache, current_path, t0 = {}, None, time.time()
    for i, row in enumerate(windows_df.itertuples(index=False)):
        if row.harmonized_path != current_path:      # windows arrive trial by trial; keep one trial in memory
            cache, current_path = {}, row.harmonized_path
        aug = load_augmented_window(pd.Series(row._asdict()), cfg.window_length_samples,
                                    sample_rate_hz=cfg.target_rate_hz, signal_cache=cache)
        arr[i] = aug.T
        if (i + 1) % 25000 == 0:
            print(f"  {i + 1:,}/{len(windows_df):,} ({time.time() - t0:.0f}s)", flush=True)
    arr.flush()
    del arr
    tmp.replace(path)
    meta_path.write_text(json.dumps(meta))
    return np.load(path, mmap_mode="r")


# ------------------------------------------------------------------ per-fold work

def _predict_probs(model, windows: np.ndarray, rows: np.ndarray, device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), batch_size):
            x = torch.from_numpy(np.asarray(windows[rows[s:s + batch_size]], dtype=np.float32)).to(device)
            out.append(F.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(out)


def _probs_frame(df: pd.DataFrame, probs: np.ndarray, arch: str) -> pd.DataFrame:
    out = df[TRIAL_KEY_COLS + ["global_subject_id", "start_frame", "label", "fall_impact_frame"]].copy()
    for name, idx in LABEL_TO_INT.items():
        out[f"{arch}__proba_{name}"] = probs[:, idx]
    return out


def _event_dict(r) -> dict:
    return {"threshold": r.threshold, "n_fall_trials": r.n_fall_trials, "n_fall_warned": r.n_fall_warned,
            "detection_rate": r.detection_rate, "n_adl_trials": r.n_adl_trials,
            "n_adl_false_alarm": r.n_adl_false_alarm, "false_alarm_rate": r.false_alarm_rate,
            "mean_lead_time_ms": r.mean_lead_time_ms, "median_lead_time_ms": r.median_lead_time_ms}


def run_fold_architecture(k, arch, train_df, val_df, test_df, trial_df, windows, cfg, fold_dir, args, device) -> dict:
    t0 = time.time()
    factory = lambda df: PrecomputedWindowDataset(df, windows)  # noqa: E731
    train_report = train_one_architecture(
        arch, train_df, val_df, test_df, trial_df, cfg, fold_dir,
        args.max_epochs, args.patience, args.batch_size, device, args.pre_impact_boost,
        dataset_factory=factory,
    )
    model = load_prediction_model(fold_dir / f"{arch}_model.pt").to(device)
    val_p = _probs_frame(val_df, _predict_probs(model, windows, val_df["cache_row"].to_numpy(), device), arch)
    test_p = _probs_frame(test_df, _predict_probs(model, windows, test_df["cache_row"].to_numpy(), device), arch)
    test_p.to_parquet(fold_dir / f"{arch}_test_probabilities.parquet", index=False)

    chosen, fallback = select_threshold(sweep_thresholds(_to_trials(val_p, arch), THRESHOLDS), args.max_false_alarm_rate)
    test_trials = _to_trials(test_p, arch)
    locked = evaluate_event_level(test_trials, chosen.threshold)
    per_subject = {
        s: _event_dict(evaluate_event_level([t for t in test_trials if t.subject_id == s], chosen.threshold))
        for s in sorted({t.subject_id for t in test_trials})
    }
    cm = np.array(train_report["test_confusion_matrix"])
    print(f"  [fold {k} {arch}] val-chosen threshold {chosen.threshold:.2f}"
          f"{' (FALLBACK)' if fallback else ''} -> test detection {locked.n_fall_warned}/{locked.n_fall_trials}"
          f" false alarms {locked.n_adl_false_alarm}/{locked.n_adl_trials} lead {locked.mean_lead_time_ms:.0f} ms", flush=True)
    return {
        "fold": k, "architecture": arch,
        "subjects": {"train": sorted(train_df["global_subject_id"].unique()),
                     "val": sorted(val_df["global_subject_id"].unique()),
                     "test": sorted(test_df["global_subject_id"].unique())},
        "best_val_loss": train_report["best_val_loss"],
        "window_level": {
            "macro_f1": train_report["test_macro_f1"],
            "accuracy": float(np.trace(cm) / cm.sum()),
            "pre_impact_precision": float(cm[1, 1] / cm[:, 1].sum()) if cm[:, 1].sum() else 0.0,
            "pre_impact_recall": float(cm[1, 1] / cm[1, :].sum()) if cm[1, :].sum() else 0.0,
            "confusion_matrix": cm.tolist(),
        },
        "raw_argmax_lead_time": train_report["lead_time"],
        "threshold_selection": {"selected_threshold": chosen.threshold, "used_fallback": fallback,
                                "val_at_selected": _event_dict(chosen)},
        "test_event_level_locked": _event_dict(locked),
        "test_per_subject_locked": per_subject,
        "wall_clock_s": time.time() - t0,
    }


# ------------------------------------------------------------------ aggregation

def _stats(values) -> dict:
    v = np.array([x for x in values if x is not None], dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


def aggregate(out_dir: Path, archs: list[str]) -> dict:
    summary = {"architectures": {}}
    results = {a: sorted((json.loads(p.read_text()) for p in out_dir.glob(f"fold_*/{a}_result.json")),
                         key=lambda r: r["fold"]) for a in archs}
    for a, rs in results.items():
        if not rs:
            continue
        ev = [r["test_event_level_locked"] for r in rs]
        warned, falls = sum(e["n_fall_warned"] for e in ev), sum(e["n_fall_trials"] for e in ev)
        fa, adl = sum(e["n_adl_false_alarm"] for e in ev), sum(e["n_adl_trials"] for e in ev)
        subj = {s: d for r in rs for s, d in r["test_per_subject_locked"].items()}
        summary["architectures"][a] = {
            "n_folds_completed": len(rs),
            "per_fold_mean_std": {
                "macro_f1": _stats(r["window_level"]["macro_f1"] for r in rs),
                "window_accuracy": _stats(r["window_level"]["accuracy"] for r in rs),
                "pre_impact_precision": _stats(r["window_level"]["pre_impact_precision"] for r in rs),
                "pre_impact_recall": _stats(r["window_level"]["pre_impact_recall"] for r in rs),
                "selected_threshold": _stats(r["threshold_selection"]["selected_threshold"] for r in rs),
                "event_detection_rate": _stats(e["detection_rate"] for e in ev),
                "event_false_alarm_rate": _stats(e["false_alarm_rate"] for e in ev),
                "mean_lead_time_ms": _stats(e["mean_lead_time_ms"] for e in ev),
            },
            "pooled_over_all_test_subjects": {
                "n_subjects": len(subj),
                "detection": [warned, falls, warned / falls, wilson_interval(warned, falls)],
                "false_alarm": [fa, adl, fa / adl, wilson_interval(fa, adl)],
                "mean_lead_time_ms": float(sum(e["mean_lead_time_ms"] * e["n_fall_warned"] for e in ev) / warned),
            },
            "per_subject_mean_std": {
                "detection_rate": _stats(d["detection_rate"] for d in subj.values()),
                "false_alarm_rate": _stats(d["false_alarm_rate"] for d in subj.values()),
            },
            "per_subject": subj,
        }
    done = [a for a in archs if results.get(a)]
    if len(done) == 2:
        a, b = done
        common = sorted({r["fold"] for r in results[a]} & {r["fold"] for r in results[b]})
        get = lambda rs, f: next(r for r in rs if r["fold"] == f)  # noqa: E731
        diff = lambda fn: [fn(get(results[b], f)) - fn(get(results[a], f)) for f in common]  # noqa: E731
        summary["paired_difference_per_fold"] = {
            "definition": f"{b} minus {a}, per fold",
            "macro_f1": diff(lambda r: r["window_level"]["macro_f1"]),
            "event_detection_rate": diff(lambda r: r["test_event_level_locked"]["detection_rate"]),
            "event_false_alarm_rate": diff(lambda r: r["test_event_level_locked"]["false_alarm_rate"]),
            "mean_lead_time_ms": diff(lambda r: r["test_event_level_locked"]["mean_lead_time_ms"]),
        }
    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Cross-validation summary (test subjects, thresholds chosen on each fold's val) ===")
    for a, s in summary["architectures"].items():
        m, p = s["per_fold_mean_std"], s["pooled_over_all_test_subjects"]
        fmt = lambda d, pct=True: (f"{d['mean']:.1%} ± {d['std']:.1%}" if pct else f"{d['mean']:.3f} ± {d['std']:.3f}")  # noqa: E731
        print(f"  {a} ({s['n_folds_completed']} folds, {p['n_subjects']} test subjects)")
        print(f"    macro-F1 {fmt(m['macro_f1'], False)} | threshold {fmt(m['selected_threshold'], False)}")
        print(f"    detection {fmt(m['event_detection_rate'])} | false alarms {fmt(m['event_false_alarm_rate'])}"
              f" | lead {m['mean_lead_time_ms']['mean']:.0f} ± {m['mean_lead_time_ms']['std']:.0f} ms")
        print(f"    pooled: detection {p['detection'][0]}/{p['detection'][1]} = {p['detection'][2]:.1%}"
              f" [{p['detection'][3][0]:.1%}-{p['detection'][3][1]:.1%}], false alarms {p['false_alarm'][0]}/"
              f"{p['false_alarm'][1]} = {p['false_alarm'][2]:.1%} [{p['false_alarm'][3][0]:.1%}-{p['false_alarm'][3][1]:.1%}]")
    return summary


# ------------------------------------------------------------------ main

def _write_progress(out_dir: Path, **fields) -> None:
    path = out_dir / "progress.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(fields, updated_at=datetime.now().isoformat(timespec="seconds"))
    path.write_text(json.dumps(state, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "harmonized" / "manifest.parquet"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "cv_prediction"))
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "results" / "cv_prediction_cache"))
    parser.add_argument("--architectures", nargs="+", default=ARCHITECTURES, choices=ARCHITECTURES)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-val-subjects", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pre-impact-boost", type=float, default=1.0)
    parser.add_argument("--max-false-alarm-rate", type=float, default=0.10)
    parser.add_argument("--subjects-limit", type=int, default=None, help="Smoke test: use only the first N subjects.")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(out_dir, args.architectures)
        return

    cfg = PredictionWindowingConfig()
    trial_df = load_manifest(args.manifest)
    windows_df = build_windows_manifest(trial_df, config=cfg)
    windows = build_or_load_window_cache(windows_df, cfg, Path(args.cache_dir))
    windows_df["cache_row"] = np.arange(len(windows_df))

    subjects = sorted(windows_df["global_subject_id"].unique())
    if args.subjects_limit:
        subjects = subjects[:args.subjects_limit]
    folds = make_subject_kfolds(subjects, args.n_folds, args.seed)
    config_record = {k: v for k, v in vars(args).items() if k != "aggregate_only"}
    (out_dir / "cv_config.json").write_text(json.dumps({**config_record, "folds": folds}, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = [(k, a) for k in range(len(folds)) for a in args.architectures]
    print(f"{len(folds)} folds x {len(args.architectures)} architectures on {device}; folds: {folds}")
    for n_done_before, (k, arch) in enumerate(pairs):
        fold_dir = out_dir / f"fold_{k}"
        fold_dir.mkdir(exist_ok=True)
        result_path = fold_dir / f"{arch}_result.json"
        if result_path.exists():
            print(f"[fold {k} {arch}] already done, skipping")
            continue
        test_subj = set(folds[k])
        val_subj = set(pick_validation_subjects([s for s in subjects if s not in test_subj],
                                                args.n_val_subjects, args.seed + k))
        train_subj = set(subjects) - test_subj - val_subj
        sel = lambda ss: windows_df[windows_df["global_subject_id"].isin(ss)].reset_index(drop=True)  # noqa: E731
        train_df, val_df, test_df = sel(train_subj), sel(val_subj), sel(test_subj)
        _write_progress(out_dir, current={"fold": k, "architecture": arch, "started_at": datetime.now().isoformat(timespec="seconds")},
                        completed=n_done_before, total=len(pairs))
        print(f"\n##### fold {k} / {arch}: train {len(train_subj)} subj ({len(train_df):,} win), "
              f"val {sorted(val_subj)}, test {sorted(test_subj)}", flush=True)
        result = run_fold_architecture(k, arch, train_df, val_df, test_df, trial_df, windows, cfg, fold_dir, args, device)
        tmp = result_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2))
        tmp.replace(result_path)
        _write_progress(out_dir, last_finished={"fold": k, "architecture": arch, "wall_clock_s": result["wall_clock_s"]})

    _write_progress(out_dir, current=None, completed=len(pairs), total=len(pairs), finished=True)
    aggregate(out_dir, args.architectures)


if __name__ == "__main__":
    main()
