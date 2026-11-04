"""Leave-one-subject-out (LOSO) cross-validation of the detection stream
(peer review item W3).

For each of the 70 subjects (32 KFall + 38 SisFall):
  test  = that subject
  val   = ~15% of the remaining subjects of EACH dataset (seeded), used only
          for XGBoost early stopping -- same role and proportion as the
          production split (detection.split, val_size=0.15 per dataset)
  train = everyone else
then the production recipe (`detection.model.train_model`, TrainingConfig
defaults, class weight from the train split) is fitted and the held-out
subject's windows are scored. Uses the cached 54-feature matrix
(results/detection_model/features_cache_all.parquet), so no feature
recomputation.

Reported at the production decision threshold (0.5):
  * pooled over all 70 held-out subjects (every window is scored by a
    model that never saw its subject): accuracy, precision, recall,
    specificity, F1, ROC-AUC -- overall, per dataset, and SisFall
    young (SA*) vs elderly (SE*);
  * per-subject spread (mean +/- std) of accuracy, recall (subjects with
    falls) and specificity.

Resumable: per-subject predictions go to <output-dir>/predictions/<subject>.parquet.

Usage:
    python scripts/cross_validate_detection.py
    python scripts/cross_validate_detection.py --aggregate-only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from detection.features import FEATURE_NAMES
from detection.model import TrainingConfig, train_model

REPO_ROOT = Path(__file__).parent.parent
DECISION_THRESHOLD = 0.5
META_COLS = ["dataset", "global_subject_id", "activity_code", "trial_id", "window_index", "label"]


def pick_val_subjects(remaining: list[str], subject_dataset: dict[str, str], frac: float, seed: int) -> set[str]:
    """`frac` of the remaining subjects within each dataset (at least one)."""
    rng = np.random.default_rng(seed)
    val = set()
    for ds in sorted(set(subject_dataset[s] for s in remaining)):
        pool = sorted(s for s in remaining if subject_dataset[s] == ds)
        n = max(1, round(frac * len(pool)))
        val.update(rng.choice(pool, size=n, replace=False).tolist())
    return val


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= DECISION_THRESHOLD).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    tn, fn = int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else None  # noqa: E731
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "n_windows": int(len(y)), "n_fall_windows": int(y.sum()),
        "accuracy": div(tp + tn, len(y)), "precision": prec, "recall": rec,
        "specificity": div(tn, tn + fp),
        "f1": div(2 * prec * rec, prec + rec) if prec is not None and rec is not None and prec + rec else None,
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _spread(values) -> dict | None:
    v = np.array([x for x in values if x is not None], dtype=float)
    if not len(v):
        return None
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n_subjects": int(len(v))}


def aggregate(out_dir: Path) -> dict:
    preds = pd.concat([pd.read_parquet(p) for p in sorted((out_dir / "predictions").glob("*.parquet"))],
                      ignore_index=True)
    y, p = preds["label"].to_numpy(), preds["fall_probability"].to_numpy()
    is_sis = preds["dataset"] == "sisfall"
    elderly = is_sis & preds["global_subject_id"].str.contains("_SE")
    groups = {"all": np.ones(len(preds), bool), "kfall": (preds["dataset"] == "kfall").to_numpy(),
              "sisfall": is_sis.to_numpy(), "sisfall_young": (is_sis & ~elderly).to_numpy(),
              "sisfall_elderly": elderly.to_numpy()}
    per_subject = {s: _metrics(g["label"].to_numpy(), g["fall_probability"].to_numpy())
                   for s, g in preds.groupby("global_subject_id")}
    summary = {
        "protocol": {"scheme": "leave-one-subject-out", "decision_threshold": DECISION_THRESHOLD,
                     "n_subjects": len(per_subject)},
        "pooled": {k: _metrics(y[m], p[m]) for k, m in groups.items()},
        "per_subject_spread": {
            k: _spread(per_subject[s][k] for s in per_subject) for k in ("accuracy", "recall", "specificity", "roc_auc")
        },
        "per_subject": per_subject,
    }
    (out_dir / "loso_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Detection LOSO (pooled over held-out subjects, threshold 0.5) ===")
    for k, m in summary["pooled"].items():
        f = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
        print(f"  {k:16s} windows={m['n_windows']:6d}  acc={f(m['accuracy'])}  prec={f(m['precision'])}  "
              f"recall={f(m['recall'])}  spec={f(m['specificity'])}  F1={f(m['f1'])}  AUC={f(m['roc_auc'])}")
    for k, s in summary["per_subject_spread"].items():
        if s:
            print(f"  per-subject {k:11s} {s['mean']:.3f} ± {s['std']:.3f} (min {s['min']:.3f}, n={s['n_subjects']})")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=str(REPO_ROOT / "results" / "detection_model" / "features_cache_all.parquet"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "cv_detection"))
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        feats = pd.read_parquet(args.features)
        subject_dataset = feats.groupby("global_subject_id")["dataset"].first().to_dict()
        subjects = sorted(subject_dataset)
        t_start = time.time()
        for i, s in enumerate(subjects):
            out = out_dir / "predictions" / f"{s}.parquet"
            if out.exists():
                continue
            remaining = [o for o in subjects if o != s]
            val = pick_val_subjects(remaining, subject_dataset, args.val_frac, args.seed + i)
            train_df = feats[feats["global_subject_id"].isin(set(remaining) - val)]
            val_df = feats[feats["global_subject_id"].isin(val)]
            test_df = feats[feats["global_subject_id"] == s]
            model = train_model(train_df, val_df, TrainingConfig())
            res = test_df[META_COLS].copy()
            res["fall_probability"] = model.predict_proba(test_df[FEATURE_NAMES].to_numpy(dtype=np.float32))[:, 1]
            res.to_parquet(out, index=False)
            print(f"  [{i + 1}/{len(subjects)}] {s}: {len(test_df)} windows "
                  f"({time.time() - t_start:.0f}s elapsed)", flush=True)
    aggregate(out_dir)


if __name__ == "__main__":
    main()
