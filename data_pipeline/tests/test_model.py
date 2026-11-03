import numpy as np
import pandas as pd

from shravan_ai.model import (
    EVENT_FALL_DETECTED,
    RAW_INPUT_COLUMNS,
    ThresholdModelConfig,
    _ground_truth_free,
    run_threshold_model,
)


def _timeline_with_impact_spike(n_frames=60, spike_frame=40):
    acc_z = np.ones(n_frames)
    acc_z[spike_frame:spike_frame + 3] = 4.0
    return pd.DataFrame(
        {
            "frame_idx": np.arange(n_frames),
            "timestamp_s": np.arange(n_frames) / 10.0,
            "acc_x": np.zeros(n_frames),
            "acc_y": np.zeros(n_frames),
            "acc_z": acc_z,
            "gyr_x": np.zeros(n_frames),
            "gyr_y": np.zeros(n_frames),
            "gyr_z": np.zeros(n_frames),
            "phase": ["Normal"] * n_frames,
            "fall_direction": ["fall_forward"] * n_frames,
        }
    )


def test_model_never_reads_ground_truth_columns():
    timeline = _timeline_with_impact_spike()
    stripped = _ground_truth_free(timeline)
    assert "phase" not in stripped.columns
    assert "fall_direction" not in stripped.columns
    assert set(stripped.columns) == set(RAW_INPUT_COLUMNS)


def test_fall_detected_fires_at_acceleration_spike():
    timeline = _timeline_with_impact_spike(spike_frame=40)
    log = run_threshold_model(timeline, ThresholdModelConfig(acc_impact_threshold_g=2.5))
    detections = log[log["event"] == EVENT_FALL_DETECTED]
    assert len(detections) >= 1
    assert detections.iloc[0]["frame_idx"] == 40


def test_no_spike_never_fires():
    n = 50
    timeline = pd.DataFrame(
        {
            "frame_idx": np.arange(n),
            "timestamp_s": np.arange(n) / 10.0,
            "acc_x": np.zeros(n), "acc_y": np.zeros(n), "acc_z": np.ones(n),
            "gyr_x": np.zeros(n), "gyr_y": np.zeros(n), "gyr_z": np.zeros(n),
        }
    )
    log = run_threshold_model(timeline)
    assert log.empty
