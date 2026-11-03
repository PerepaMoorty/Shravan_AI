"""Train + evaluate the prediction pipeline's deep models, end to end.

Ties the model architectures, focal loss and lead-time metric into a
training loop. Mirrors detection's `train_detection_model.py`
in shape, adapted for a torch model instead of XGBoost.

Usage:
    python scripts/train_prediction_model.py
    python scripts/train_prediction_model.py --architecture convlstm
    python scripts/train_prediction_model.py --architecture tiny_transformer
    python scripts/train_prediction_model.py --architecture both

Pipeline: trial manifest -> dense windows -> subject-aware train/val/test
split (a SINGLE split, not full 32-fold LOSO -- see prediction/split.py's
docstring for that tradeoff) -> train with focal loss + early stopping on
val loss -> evaluate on held-out test: per-window 3-class metrics AND the
real lead-time metric per fall trial (prediction/lead_time.py) -> save
checkpoint(s) + a JSON report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from prediction.dataset import build_windows_manifest
from prediction.features import load_augmented_window
from prediction.lead_time import compute_lead_time_ms, summarize_lead_times
from prediction.losses import FocalLoss, default_alpha_weights
from prediction.models.convlstm import ConvLSTM
from prediction.models.tiny_transformer import TinyTransformer
from prediction.predict import save_prediction_model
from prediction.split import PredictionSplitConfig, split_by_subject
from prediction.torch_dataset import PredictionWindowDataset, TrialGroupedBatchSampler
from prediction.windowing import PredictionWindowingConfig
from shared.manifest import load_manifest

REPO_ROOT = Path(__file__).parent.parent

_ARCHITECTURES = {"convlstm": ConvLSTM, "tiny_transformer": TinyTransformer}


def _run_epoch(model, loader, loss_fn, device, optimizer=None) -> float:
    """One pass over `loader`. Trains if `optimizer` is given, else
    only evaluates (no grad, no weight update) -- same function for
    both so the train/val loop below can't accidentally diverge in
    behavior between the two modes.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, n_batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            loss = loss_fn(model(x), y)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _predict_all(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            all_preds.append(model(x).argmax(dim=1).cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def _compute_test_lead_times(
    model: torch.nn.Module,
    test_windows_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    windowing_config: PredictionWindowingConfig,
    device: torch.device,
) -> list[float | None]:
    """Per fall trial actually present in the held-out test split: run
    every one of its dense windows through `model`, then find the ms
    before impact it first flagged `pre_impact`
    (`prediction.lead_time.compute_lead_time_ms`) -- the real metric
    this whole pipeline exists to optimize, computed on genuinely
    held-out subjects, not a proxy.
    """
    model.eval()
    fall_trials = trial_df[
        trial_df["fall_onset_frame"].notna() & trial_df["fall_impact_frame"].notna()
    ]
    test_keys = set(
        zip(test_windows_df["dataset"], test_windows_df["subject_id"],
            test_windows_df["activity_code"], test_windows_df["trial_id"])
    )

    lead_times: list[float | None] = []
    cache: dict = {}
    for _, trial_row in fall_trials.iterrows():
        key = (trial_row["dataset"], trial_row["subject_id"],
               trial_row["activity_code"], trial_row["trial_id"])
        if key not in test_keys:
            continue  # only report on trials that actually landed in test

        trial_windows = test_windows_df[
            (test_windows_df["dataset"] == key[0]) & (test_windows_df["subject_id"] == key[1])
            & (test_windows_df["activity_code"] == key[2]) & (test_windows_df["trial_id"] == key[3])
        ]
        if len(trial_windows) == 0:
            continue

        windows = np.stack([
            load_augmented_window(row, windowing_config.window_length_samples, signal_cache=cache)
            for _, row in trial_windows.iterrows()
        ])
        x = torch.from_numpy(windows.transpose(0, 2, 1).astype(np.float32)).to(device)
        with torch.no_grad():
            preds = model(x).argmax(dim=1).cpu().numpy()

        lead_times.append(compute_lead_time_ms(
            start_frames=trial_windows["start_frame"].to_numpy(),
            predicted_label_ids=preds,
            impact_frame=int(trial_row["fall_impact_frame"]),
            sample_rate_hz=windowing_config.target_rate_hz,
        ))

    return lead_times


def train_one_architecture(
    architecture: str,
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    windowing_config: PredictionWindowingConfig,
    output_dir: Path,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    pre_impact_boost: float,
) -> dict:
    print(f"\n=== Training {architecture} ===")
    torch.manual_seed(42)

    window_length = windowing_config.window_length_samples
    rate = windowing_config.target_rate_hz
    train_ds = PredictionWindowDataset(train_df, window_length, rate)
    val_ds = PredictionWindowDataset(val_df, window_length, rate)
    test_ds = PredictionWindowDataset(test_df, window_length, rate)

    train_loader = DataLoader(
        train_ds, batch_sampler=TrialGroupedBatchSampler(train_df, batch_size, shuffle=True, seed=42)
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Alpha weights from the TRAIN split's OWN class balance only --
    # never the full dataset, which would leak held-out class-balance
    # information into the loss (see losses.default_alpha_weights's
    # own docstring on this point).
    alpha = default_alpha_weights(
        train_df["label"].value_counts().to_dict(), pre_impact_extra_boost=pre_impact_boost
    )
    print(f"  alpha weights (non_fall, pre_impact, fall) = {alpha.tolist()} "
          f"(pre_impact_extra_boost={pre_impact_boost})")
    loss_fn = FocalLoss(alpha=alpha)

    model = _ARCHITECTURES[architecture]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = _run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        print(f"  epoch {epoch:2d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)

    model_path = output_dir / f"{architecture}_model.pt"
    save_prediction_model(model, architecture, model_path)
    print(f"  saved best checkpoint (val_loss={best_val_loss:.4f}) to {model_path}")

    test_preds, test_labels = _predict_all(model, test_loader, device)
    report_text = classification_report(
        test_labels, test_preds, labels=[0, 1, 2],
        target_names=["non_fall", "pre_impact", "fall"], zero_division=0,
    )
    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1, 2])
    macro_f1 = f1_score(test_labels, test_preds, labels=[0, 1, 2], average="macro", zero_division=0)
    print(report_text)

    lead_times = _compute_test_lead_times(model, test_df, trial_df, windowing_config, device)
    lead_summary = summarize_lead_times(lead_times) if lead_times else None
    if lead_summary:
        print(f"  lead-time: detection_rate={lead_summary.detection_rate:.2f}  "
              f"mean={lead_summary.mean_lead_time_ms}ms  median={lead_summary.median_lead_time_ms}ms  "
              f"(n_trials={lead_summary.n_trials}, n_flagged={lead_summary.n_flagged})")
    else:
        print("  lead-time: no fall trials in the test split -- can't report (try a different --seed split)")

    return {
        "architecture": architecture,
        "pre_impact_extra_boost": pre_impact_boost,
        "best_val_loss": best_val_loss,
        "test_macro_f1": float(macro_f1),
        "test_confusion_matrix": cm.tolist(),
        "test_classification_report": report_text,
        "model_path": str(model_path),
        "lead_time": None if lead_summary is None else {
            "n_trials": lead_summary.n_trials,
            "n_flagged": lead_summary.n_flagged,
            "detection_rate": lead_summary.detection_rate,
            "mean_lead_time_ms": lead_summary.mean_lead_time_ms,
            "median_lead_time_ms": lead_summary.median_lead_time_ms,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=str(REPO_ROOT / "data" / "harmonized" / "manifest.parquet"),
    )
    parser.add_argument(
        "--output-dir", default=str(REPO_ROOT / "results" / "prediction_model"),
    )
    parser.add_argument(
        "--architecture", choices=["convlstm", "tiny_transformer", "both"], default="both",
    )
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--pre-impact-boost", type=float, default=2.0,
        help="Extra multiplier on pre_impact's inverse-frequency alpha weight, on top of the "
             "already-heavy inverse-frequency weighting itself (see prediction/losses.py's "
             "default_alpha_weights docstring). The default (2.0) was found to overcorrect on "
             "real KFall data -- push the model toward predicting pre_impact almost "
             "indiscriminately (~10%% precision on the first real training run, 2026-07-21). "
             "Try 1.0 (pure inverse-frequency, no extra boost) for a less aggressive weighting.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " (no CUDA GPU found -- training on CPU)"))

    print(f"Loading trial manifest from {args.manifest}...")
    trial_df = load_manifest(args.manifest)

    windowing_config = PredictionWindowingConfig()
    print(f"Building dense prediction windows (window={windowing_config.window_length_s}s, "
          f"stride={windowing_config.stride_s}s)...")
    windows_df = build_windows_manifest(trial_df, config=windowing_config)
    print(f"  {len(windows_df)} windows across {windows_df['global_subject_id'].nunique()} subjects "
          f"({windows_df['label'].value_counts().to_dict()})")

    print("Splitting train/val/test (subject-aware, single split -- see prediction/split.py)...")
    train_df, val_df, test_df = split_by_subject(windows_df, PredictionSplitConfig())
    print(f"  train={len(train_df)} ({train_df['global_subject_id'].nunique()} subjects)")
    print(f"  val=  {len(val_df)} ({val_df['global_subject_id'].nunique()} subjects)")
    print(f"  test= {len(test_df)} ({test_df['global_subject_id'].nunique()} subjects)")

    architectures = ["convlstm", "tiny_transformer"] if args.architecture == "both" else [args.architecture]

    reports = [
        train_one_architecture(
            architecture, train_df, val_df, test_df, trial_df, windowing_config,
            output_dir, args.max_epochs, args.patience, args.batch_size, device,
            args.pre_impact_boost,
        )
        for architecture in architectures
    ]

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "n_windows": len(windows_df),
            "train_size": len(train_df), "val_size": len(val_df), "test_size": len(test_df),
            "architectures": reports,
        }, f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
