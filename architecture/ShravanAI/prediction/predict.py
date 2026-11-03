"""Inference utilities: apply a trained prediction model to get
3-class (non_fall / pre_impact / fall) predictions on new or held-out
data.

Mirrors `detection/predict.py`'s two-entry-point shape, adapted for
this pipeline's real differences: a torch model (not XGBoost) and a
3-class softmax output (not a single fall-probability scalar). This
module didn't exist before -- the prediction pipeline had model
architectures, a loss, and a lead-time metric, but no code path that
actually loads a trained checkpoint and produces predictions on new
data. Building it is what makes the prediction stream runnable at all,
independent of whether a specific checkpoint has been trained yet.

Two entry points, same shape as detection's:
- `predict_from_manifest`: batch predictions over every window in a
  trial manifest (e.g. a held-out test set, or a brand-new trial run
  through harmonization later).
- `predict_single_window`: predict on one raw (n_samples, 6) array
  directly, for ad hoc / single-sample use (e.g. one Unity trial's
  windows, fed one at a time).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from prediction.dataset import build_windows_manifest
from prediction.features import augment_window, load_augmented_window
from prediction.labelers import LABEL_TO_INT
from prediction.models.convlstm import ConvLSTM
from prediction.models.tiny_transformer import TinyTransformer
from prediction.windowing import PredictionWindowingConfig
from shared.manifest import load_manifest

INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

# Registry mirroring shared/harmonize/units.py's converter-registry
# pattern: one name -> constructor mapping, so a checkpoint only needs
# to record which name it used, not the class itself (torch.save on a
# bare nn.Module pickles the class definition by reference, which
# breaks the moment this module's import path changes -- saving a
# state_dict + an architecture name string is the standard, more
# robust alternative).
_ARCHITECTURES = {
    "convlstm": ConvLSTM,
    "tiny_transformer": TinyTransformer,
}


def save_prediction_model(model: torch.nn.Module, architecture: str, path: Path) -> None:
    """Save a trained model's weights + which architecture they belong
    to, so `load_prediction_model` can reconstruct the right class
    without the caller needing to already know it.
    """
    if architecture not in _ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}; known: {sorted(_ARCHITECTURES)}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": architecture, "state_dict": model.state_dict()}, path)


def load_prediction_model(path: Path) -> torch.nn.Module:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    architecture = checkpoint["architecture"]
    if architecture not in _ARCHITECTURES:
        raise ValueError(
            f"Checkpoint at {path} names unknown architecture {architecture!r}; "
            f"known: {sorted(_ARCHITECTURES)}"
        )
    model = _ARCHITECTURES[architecture]()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_from_manifest(
    model_path: Path,
    trial_manifest_path: Path,
    windowing_config: Optional[PredictionWindowingConfig] = None,
) -> pd.DataFrame:
    """End-to-end: trial manifest -> dense windows -> augmented signal
    -> 3-class predictions.

    Returns one row per window with identifying columns (dataset,
    global_subject_id, activity_code, trial_id, window_index,
    start_frame, end_frame), the TRUE label (present whenever the
    source manifest has one), predicted label, and per-class
    probabilities (`proba_non_fall`, `proba_pre_impact`, `proba_fall`).
    """
    windowing_config = windowing_config or PredictionWindowingConfig()
    model = load_prediction_model(model_path)

    trial_df = load_manifest(trial_manifest_path)
    windows_df = build_windows_manifest(trial_df, config=windowing_config)

    if len(windows_df) == 0:
        return windows_df.assign(
            predicted_label=[], proba_non_fall=[], proba_pre_impact=[], proba_fall=[]
        )

    cache: dict = {}
    window_length = windowing_config.window_length_samples
    windows = np.stack([
        load_augmented_window(row, window_length, signal_cache=cache)
        for _, row in windows_df.iterrows()
    ])  # (n_windows, window_length, 9)

    x = torch.from_numpy(windows.transpose(0, 2, 1).astype(np.float32))  # (n_windows, 9, window_length)
    with torch.no_grad():
        logits = model(x)
        proba = F.softmax(logits, dim=1).numpy()

    result = windows_df[
        ["dataset", "global_subject_id", "activity_code", "trial_id",
         "window_index", "start_frame", "end_frame", "label"]
    ].copy()
    result["predicted_label"] = [INT_TO_LABEL[i] for i in proba.argmax(axis=1)]
    result["proba_non_fall"] = proba[:, LABEL_TO_INT["non_fall"]]
    result["proba_pre_impact"] = proba[:, LABEL_TO_INT["pre_impact"]]
    result["proba_fall"] = proba[:, LABEL_TO_INT["fall"]]
    return result


def predict_single_window(
    model_path: Path, window: np.ndarray, sample_rate_hz: float = 100.0
) -> dict:
    """Predict on one raw window array (shape (n_samples, 6), CHANNELS
    order: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z).

    Loads the model fresh on every call -- fine for occasional/ad hoc
    use (e.g. the digital twin's per-trial model-log generator, which
    loads a model once per trial, not per window -- see
    `predict_windows_batch` for the loop-efficient version of this).
    """
    model = load_prediction_model(model_path)
    return _predict_augmented(model, augment_window(window, sample_rate_hz=sample_rate_hz))


def predict_windows_batch(model: torch.nn.Module, windows: np.ndarray) -> list[dict]:
    """Predict on a batch of already-augmented windows (shape
    (n_windows, n_samples, 9)) using an already-loaded model -- the
    loop-efficient sibling of `predict_single_window`, for callers
    (like the digital twin's Unity export step) that need many
    predictions from one model without reloading it each time.
    """
    x = torch.from_numpy(windows.transpose(0, 2, 1).astype(np.float32))
    with torch.no_grad():
        logits = model(x)
        proba = F.softmax(logits, dim=1).numpy()

    return [
        {
            "predicted_label": INT_TO_LABEL[row.argmax()],
            "proba_non_fall": float(row[LABEL_TO_INT["non_fall"]]),
            "proba_pre_impact": float(row[LABEL_TO_INT["pre_impact"]]),
            "proba_fall": float(row[LABEL_TO_INT["fall"]]),
        }
        for row in proba
    ]


def _predict_augmented(model: torch.nn.Module, augmented_window: np.ndarray) -> dict:
    x = torch.from_numpy(augmented_window.T[np.newaxis].astype(np.float32))  # (1, 9, n_samples)
    with torch.no_grad():
        logits = model(x)
        proba = F.softmax(logits, dim=1).numpy()[0]

    return {
        "predicted_label": INT_TO_LABEL[int(proba.argmax())],
        "proba_non_fall": float(proba[LABEL_TO_INT["non_fall"]]),
        "proba_pre_impact": float(proba[LABEL_TO_INT["pre_impact"]]),
        "proba_fall": float(proba[LABEL_TO_INT["fall"]]),
    }
