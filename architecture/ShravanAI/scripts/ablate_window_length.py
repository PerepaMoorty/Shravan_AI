"""Window-length ablation for the prediction pipeline's ConvLSTM model.

Companion to `train_prediction_model.py` (see that script's docstring
for the full pipeline explanation -- this one reuses its
`train_one_architecture` function rather than duplicating the training
loop). The baseline prediction pipeline uses a 1.0s window / 0.1s
stride (`PredictionWindowingConfig` defaults). ConvLSTM's test-set
`pre_impact` precision on that baseline is poor (~0.57) despite
decent recall (~0.87) and strong performance on the other two classes
(non_fall ~0.98, fall ~0.84). This script asks: is window LENGTH a
material driver of that gap, or is it flat/noise-level across window
lengths?

Method: rebuild the dense windows manifest with `window_length_s` set
to the CLI value while holding `stride_s=0.1` fixed (so lead-time
temporal resolution doesn't confound the comparison -- window length
is the only thing that varies between runs), re-split subjects (same
`PredictionSplitConfig` defaults, same `split_by_subject` call as the
baseline -- window count per subject changes with window length, but
the split logic itself doesn't), and train ConvLSTM only (skipping
tiny_transformer -- one architecture per window length is enough to
answer the research question and keeps runtime bounded) with the same
hyperparameters the baseline script's known-good run used:
`--pre-impact-boost 1.0`, max_epochs=15, patience=5, batch_size=64.

Usage:
    python scripts/ablate_window_length.py --window-length-s 0.5
    python scripts/ablate_window_length.py --window-length-s 2.0

Writes `results/prediction_model/ablation_window_length_<Xs>.json`
(e.g. `ablation_window_length_0.5s.json`) -- a SEPARATE file from the
baseline's `evaluation_report.json`, which this script never touches.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from prediction.dataset import build_windows_manifest
from prediction.split import PredictionSplitConfig, split_by_subject
from prediction.windowing import PredictionWindowingConfig
from shared.manifest import load_manifest
from train_prediction_model import train_one_architecture  # noqa: E402  (script-relative import; see sys.path shim below)

REPO_ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=str(REPO_ROOT / "data" / "harmonized" / "manifest.parquet"),
    )
    parser.add_argument(
        "--output-dir", default=str(REPO_ROOT / "results" / "prediction_model"),
    )
    parser.add_argument(
        "--window-length-s", type=float, required=True,
        help="Window length in seconds. Stride is held fixed at 0.1s regardless of this value "
             "(see this script's module docstring for why).",
    )
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pre-impact-boost", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " (no CUDA GPU found -- training on CPU)"))

    t_start = time.time()

    print(f"Loading trial manifest from {args.manifest}...")
    trial_df = load_manifest(args.manifest)

    # This ablation's OWN windowing config -- window_length_s varies per
    # run, stride_s and target_rate_hz stay fixed at the baseline's
    # values so only window length differs between ablation points.
    windowing_config = PredictionWindowingConfig(
        window_length_s=args.window_length_s, stride_s=0.1, target_rate_hz=100.0,
    )
    print(f"Building dense prediction windows (window={windowing_config.window_length_s}s, "
          f"stride={windowing_config.stride_s}s)...")
    windows_df = build_windows_manifest(trial_df, config=windowing_config)
    label_counts = windows_df["label"].value_counts().to_dict()
    print(f"  {len(windows_df)} windows across {windows_df['global_subject_id'].nunique()} subjects "
          f"({label_counts})")

    print("Splitting train/val/test (subject-aware, single split -- see prediction/split.py)...")
    train_df, val_df, test_df = split_by_subject(windows_df, PredictionSplitConfig())
    print(f"  train={len(train_df)} ({train_df['global_subject_id'].nunique()} subjects)")
    print(f"  val=  {len(val_df)} ({val_df['global_subject_id'].nunique()} subjects)")
    print(f"  test= {len(test_df)} ({test_df['global_subject_id'].nunique()} subjects)")

    # train_one_architecture() saves its checkpoint to
    # `<output_dir>/convlstm_model.pt` unconditionally -- passing the
    # shared `results/prediction_model/` dir here would silently
    # overwrite the baseline checkpoint of the same name (this bit us
    # once already: the first run of this script clobbered the
    # baseline's convlstm_model.pt before this fix). Route each
    # ablation point's checkpoint into its own subdirectory instead;
    # the JSON summary below still lands in the shared results dir.
    tag = f"{args.window_length_s:g}s"
    checkpoint_dir = output_dir / f"ablation_{tag}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    report = train_one_architecture(
        "convlstm", train_df, val_df, test_df, trial_df, windowing_config,
        checkpoint_dir, args.max_epochs, args.patience, args.batch_size, device,
        args.pre_impact_boost,
    )

    wall_clock_s = time.time() - t_start
    print(f"\nTotal wall-clock time: {wall_clock_s:.1f}s ({wall_clock_s / 60:.1f} min)")

    result = {
        "window_length_s": windowing_config.window_length_s,
        "stride_s": windowing_config.stride_s,
        "target_rate_hz": windowing_config.target_rate_hz,
        "window_length_samples": windowing_config.window_length_samples,
        "n_windows": len(windows_df),
        "window_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "train_size": len(train_df), "val_size": len(val_df), "test_size": len(test_df),
        "train_subjects": int(train_df["global_subject_id"].nunique()),
        "val_subjects": int(val_df["global_subject_id"].nunique()),
        "test_subjects": int(test_df["global_subject_id"].nunique()),
        "wall_clock_s": wall_clock_s,
        "architecture_report": report,
    }

    report_path = output_dir / f"ablation_window_length_{tag}.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Ablation report saved to {report_path}")


if __name__ == "__main__":
    main()
