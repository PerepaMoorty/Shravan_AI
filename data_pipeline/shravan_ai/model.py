"""Threshold-based stub fall detector (rolling gyro std / acceleration
magnitude). Keeps the pipeline runnable end to end without the trained
models in architecture/ShravanAI, which overwrite its model logs.

This module only ever reads raw sensor
columns. It must never read `phase` or `fall_direction` — those are ground
truth and belong to the avatar's track only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RAW_INPUT_COLUMNS = [
    "frame_idx", "timestamp_s",
    "acc_x", "acc_y", "acc_z",
    "gyr_x", "gyr_y", "gyr_z",
]

EVENT_PRE_IMPACT_PREDICTED = "pre_impact_predicted"
EVENT_FALL_DETECTED = "fall_detected"


@dataclass
class ThresholdModelConfig:
    gyro_window_frames: int = 5
    gyro_std_threshold: float = 40.0  # deg/s, rolling std that flags instability
    acc_impact_threshold_g: float = 2.5  # acceleration magnitude spike at impact
    refractory_frames: int = 10  # ignore repeat firings this close together


def _ground_truth_free(timeline: pd.DataFrame) -> pd.DataFrame:
    """Strip anything the model isn't allowed to see, so a caller can't
    accidentally leak ground truth into detection logic."""
    return timeline[[c for c in RAW_INPUT_COLUMNS if c in timeline.columns]].copy()


def run_threshold_model(timeline: pd.DataFrame, cfg: ThresholdModelConfig | None = None) -> pd.DataFrame:
    """Run the stub detector over one trial's raw sensor stream.

    Returns a DataFrame of frame_idx, timestamp_s, event, confidence — the
    model's independent opinion, on the same shared clock as the timeline.
    """
    cfg = cfg or ThresholdModelConfig()
    data = _ground_truth_free(timeline)

    gyro_mag = np.sqrt(data["gyr_x"] ** 2 + data["gyr_y"] ** 2 + data["gyr_z"] ** 2)
    acc_mag_g = np.sqrt(data["acc_x"] ** 2 + data["acc_y"] ** 2 + data["acc_z"] ** 2)

    gyro_std = gyro_mag.rolling(window=cfg.gyro_window_frames, min_periods=1).std().fillna(0.0)

    events = []
    last_fire_frame = {EVENT_PRE_IMPACT_PREDICTED: -np.inf, EVENT_FALL_DETECTED: -np.inf}

    for i in range(len(data)):
        frame_idx = int(data["frame_idx"].iloc[i])
        timestamp_s = float(data["timestamp_s"].iloc[i])

        if gyro_std.iloc[i] > cfg.gyro_std_threshold and (
            frame_idx - last_fire_frame[EVENT_PRE_IMPACT_PREDICTED] > cfg.refractory_frames
        ):
            confidence = float(np.clip(gyro_std.iloc[i] / (cfg.gyro_std_threshold * 2), 0.0, 1.0))
            events.append((frame_idx, timestamp_s, EVENT_PRE_IMPACT_PREDICTED, confidence))
            last_fire_frame[EVENT_PRE_IMPACT_PREDICTED] = frame_idx

        if acc_mag_g.iloc[i] > cfg.acc_impact_threshold_g and (
            frame_idx - last_fire_frame[EVENT_FALL_DETECTED] > cfg.refractory_frames
        ):
            confidence = float(np.clip(acc_mag_g.iloc[i] / (cfg.acc_impact_threshold_g * 2), 0.0, 1.0))
            events.append((frame_idx, timestamp_s, EVENT_FALL_DETECTED, confidence))
            last_fire_frame[EVENT_FALL_DETECTED] = frame_idx

    return pd.DataFrame(events, columns=["frame_idx", "timestamp_s", "event", "confidence"])
