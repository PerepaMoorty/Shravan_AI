"""Tests for prediction/predict.py."""
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from prediction.dataset import CHANNELS
from prediction.models.convlstm import ConvLSTM
from prediction.predict import (
    load_prediction_model,
    predict_from_manifest,
    predict_single_window,
    predict_windows_batch,
    save_prediction_model,
)
from prediction.windowing import PredictionWindowingConfig
from shared.manifest import ManifestRow, write_manifest


def _save_untrained_model(tmp_path, architecture="convlstm"):
    torch.manual_seed(0)
    model = ConvLSTM()
    path = tmp_path / "model.pt"
    save_prediction_model(model, architecture, path)
    return path


def _write_trial_parquet(path, n_rows):
    rng = np.random.default_rng(1)
    acc = rng.normal(0, 0.05, (n_rows, 3)).astype(np.float32)
    acc[:, 2] += 1.0
    gyro = rng.normal(0, 2.0, (n_rows, 3)).astype(np.float32)
    df = pd.DataFrame({
        "time_s": np.arange(n_rows) / 100.0,
        "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2],
        "gyro_x": gyro[:, 0], "gyro_y": gyro[:, 1], "gyro_z": gyro[:, 2],
    })
    df.to_parquet(path, index=False)
    return len(df)


def _trial_row(**overrides) -> dict:
    defaults = dict(
        dataset="kfall", subject_id="SA06", activity_code="T01", trial_id="R01",
        label="adl", duration_s=2.0, sample_rate_hz=100.0, accepted=True,
        calibration_source="T01", harmonized_path="/fake.parquet",
        fall_onset_frame=None, fall_impact_frame=None,
    )
    defaults.update(overrides)
    return asdict(ManifestRow(**defaults))


def test_save_and_load_round_trip_preserves_predictions(tmp_path):
    model_path = _save_untrained_model(tmp_path)
    loaded = load_prediction_model(model_path)

    window = np.zeros((100, len(CHANNELS)), dtype=np.float32)
    window[:, 2] = 1.0

    a = predict_single_window(model_path, window)
    b = predict_single_window(model_path, window)
    assert a == b  # same checkpoint, same input -> identical output
    assert loaded is not None  # loads without error


def test_load_prediction_model_rejects_unknown_architecture(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"architecture": "not_a_real_one", "state_dict": {}}, path)
    with pytest.raises(ValueError, match="Unknown architecture|unknown architecture"):
        load_prediction_model(path)


def test_predict_single_window_returns_valid_distribution(tmp_path):
    model_path = _save_untrained_model(tmp_path)
    window = np.zeros((100, len(CHANNELS)), dtype=np.float32)
    window[:, 2] = 1.0

    result = predict_single_window(model_path, window)

    assert result["predicted_label"] in ("non_fall", "pre_impact", "fall")
    probs = [result["proba_non_fall"], result["proba_pre_impact"], result["proba_fall"]]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-4


def test_predict_windows_batch_matches_single_window(tmp_path):
    model_path = _save_untrained_model(tmp_path)
    model = load_prediction_model(model_path)

    window = np.zeros((100, len(CHANNELS)), dtype=np.float32)
    window[:, 2] = 1.0
    from prediction.features import augment_window
    augmented = augment_window(window)

    single = predict_single_window(model_path, window)
    batch = predict_windows_batch(model, augmented[np.newaxis])[0]

    assert single["predicted_label"] == batch["predicted_label"]
    assert single["proba_pre_impact"] == pytest.approx(batch["proba_pre_impact"], abs=1e-5)


def test_predict_from_manifest_returns_expected_columns(tmp_path):
    model_path = _save_untrained_model(tmp_path)
    trial_path = tmp_path / "trial.parquet"
    n_rows = _write_trial_parquet(trial_path, n_rows=200)

    manifest_path = tmp_path / "manifest.parquet"
    write_manifest(
        [ManifestRow(**_trial_row(
            duration_s=n_rows / 100.0, harmonized_path=str(trial_path), label="adl",
        ))],
        manifest_path,
    )

    result = predict_from_manifest(model_path, manifest_path, windowing_config=PredictionWindowingConfig())

    assert len(result) > 0
    for col in ["dataset", "global_subject_id", "predicted_label",
                "proba_non_fall", "proba_pre_impact", "proba_fall"]:
        assert col in result.columns
    assert (result["proba_non_fall"] + result["proba_pre_impact"] + result["proba_fall"]).between(0.99, 1.01).all()


def test_predict_from_manifest_empty_manifest_returns_empty_with_columns(tmp_path):
    model_path = _save_untrained_model(tmp_path)
    manifest_path = tmp_path / "manifest.parquet"
    write_manifest(
        [ManifestRow(**_trial_row(accepted=False))],  # excluded by query_prediction_trials
        manifest_path,
    )

    result = predict_from_manifest(model_path, manifest_path, windowing_config=PredictionWindowingConfig())

    assert len(result) == 0
    assert "predicted_label" in result.columns
