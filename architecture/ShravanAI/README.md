# Fall Detection & Fall Prediction — Model Code

Two independent pipelines sharing one harmonization/data layer:

- **Detection** (fall vs. ADL): 2.0s windows, 54 handcrafted features, XGBoost,
  trained on KFall (32 subjects) + SisFall (38 subjects). Held-out test set:
  accuracy 0.895, ROC-AUC 0.961.
- **Prediction** (non_fall / pre_impact / fall, early warning): dense windows on
  KFall, ConvLSTM and TinyTransformer with focal loss, evaluated on lead time.
  Held-out accuracy 0.91, ~1.4s mean lead time on the curated demo trials at
  the 0.80 event threshold.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -e .
```

Parquet I/O uses `fastparquet`. Do not install `pyarrow`: on Windows a broken
`pyarrow` install crashes pandas' parquet engine fallback. Uninstall it if
another package pulls it in. For GPU training, install the CUDA build of
`torch` that matches your driver.

Dataset paths are set in `configs/datasets/{kfall,sisfall}.yaml` and point at
the repo-level `datasets/` folder.

## Run

```bash
python scripts/harmonize_dataset.py --dataset kfall      # and --dataset sisfall
python scripts/train_detection_model.py
python scripts/train_prediction_model.py --architecture both --pre-impact-boost 1.0
python scripts/export_detection_to_unity.py              # after data_pipeline's export_for_unity.py
python scripts/export_prediction_to_unity.py             # either order with the line above
```

Trained checkpoints and evaluation reports go to `results/` (gitignored).

## Tests

```bash
pytest tests/
python scripts/verify_setup.py
```

Both run against the synthetic fixtures in `tests/fixtures/`, not real data.

## Layout

```
configs/      YAML configs, composed via `defaults:` (see shared/config.py)
shared/       Readers, harmonization, windowing, manifest, tracking (used by both pipelines)
detection/    Windowing, features, split, XGBoost model, inference
prediction/   Windowing, labelers, features, LOSO folds, models, losses, lead time
scripts/      Thin CLI entry points
notebooks/    Visual QA script for harmonized signals
tests/        pytest suite + synthetic fixtures
data/         Raw/harmonized data (gitignored)
results/      Checkpoints and reports (gitignored)
```
