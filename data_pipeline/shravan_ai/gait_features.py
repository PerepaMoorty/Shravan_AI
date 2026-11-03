"""Step-timing gait features (cadence, stride time, stride-time variability)
for the Unity dashboard. Descriptive only: not a model input and not
validated against step annotations (neither KFall nor SisFall has any).

A single waist IMU can't tell left from right foot strikes, so a "step" is
an acceleration-magnitude peak at least `min_step_interval_s` after the
previous one, and stride time is approximated as 2x the mean step interval
(Bugane et al. 2012, Soaz et al. 2016).

Steps are only detected during the ground-truth "Normal" phase; values
freeze at the last estimate once the fall begins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MINIMUM_STEPS_FOR_CADENCE = 2  # one interval
MINIMUM_STEPS_FOR_VARIABILITY = 3  # two intervals, so std is non-trivial


@dataclass
class GaitFeatureConfig:
    smoothing_window_frames: int = 7
    baseline_window_frames: int = 50  # rolling window the adaptive threshold is computed over
    threshold_k: float = 1.5  # a peak must exceed rolling_mean + k * rolling_std to count as a step
    min_step_interval_s: float = 0.35  # refractory period; caps max detectable cadence at ~171 spm
    variability_window_steps: int = 5  # how many of the most recent step intervals the CV is computed over


def _detect_step_frames(acc_mag: pd.Series, timestamp_s: pd.Series, cfg: GaitFeatureConfig) -> list[int]:
    """Adaptive-threshold local-maxima peak detection on acceleration
    magnitude, refractory-gated by `min_step_interval_s`."""
    smoothed = acc_mag.rolling(window=cfg.smoothing_window_frames, min_periods=1, center=True).mean()
    rolling_mean = smoothed.rolling(window=cfg.baseline_window_frames, min_periods=1).mean()
    rolling_std = smoothed.rolling(window=cfg.baseline_window_frames, min_periods=1).std().fillna(0.0)
    threshold = rolling_mean + cfg.threshold_k * rolling_std

    values = smoothed.to_numpy()
    thresh = threshold.to_numpy()
    times = timestamp_s.to_numpy()

    steps: list[int] = []
    last_step_time = -np.inf
    for i in range(1, len(values) - 1):
        is_local_max = values[i] > values[i - 1] and values[i] >= values[i + 1]
        if is_local_max and values[i] > thresh[i] and (times[i] - last_step_time) >= cfg.min_step_interval_s:
            steps.append(i)
            last_step_time = times[i]
    return steps


def compute_gait_features(timeline: pd.DataFrame, cfg: GaitFeatureConfig | None = None) -> pd.DataFrame:
    """Per-frame gait features aligned to one trial's timeline, each value
    held from the most recently detected step (0 until enough steps exist).
    """
    cfg = cfg or GaitFeatureConfig()
    n = len(timeline)

    normal_mask = (timeline["phase"] == "Normal").to_numpy()
    normal_idx = np.flatnonzero(normal_mask)

    step_count = np.zeros(n, dtype=int)
    cadence_spm = np.zeros(n, dtype=float)
    step_interval_s = np.zeros(n, dtype=float)
    stride_time_s = np.zeros(n, dtype=float)
    stride_time_cv = np.zeros(n, dtype=float)

    if len(normal_idx) == 0:
        return _assemble(timeline, step_count, cadence_spm, step_interval_s, stride_time_s, stride_time_cv)

    normal = timeline.iloc[normal_idx]
    acc_mag = np.sqrt(normal["acc_x"] ** 2 + normal["acc_y"] ** 2 + normal["acc_z"] ** 2)
    local_step_positions = _detect_step_frames(acc_mag, normal["timestamp_s"], cfg)
    if not local_step_positions:
        return _assemble(timeline, step_count, cadence_spm, step_interval_s, stride_time_s, stride_time_cv)

    step_frames = normal_idx[local_step_positions]  # map back to full-timeline frame indices
    step_times = timeline["timestamp_s"].to_numpy()[step_frames]

    intervals: list[float] = []
    last_normal_end = normal_idx[-1] + 1  # values freeze from here through the rest of the trial

    for idx, frame in enumerate(step_frames):
        if idx > 0:
            intervals.append(step_times[idx] - step_times[idx - 1])

        recent = intervals[-cfg.variability_window_steps:]
        mean_interval = float(np.mean(recent)) if recent else 0.0

        fill_from = frame
        fill_to = step_frames[idx + 1] if idx + 1 < len(step_frames) else last_normal_end
        step_count[fill_from:fill_to] = idx + 1

        if idx + 1 >= MINIMUM_STEPS_FOR_CADENCE and mean_interval > 0:
            cadence_spm[fill_from:fill_to] = 60.0 / mean_interval
            step_interval_s[fill_from:fill_to] = mean_interval
            stride_time_s[fill_from:fill_to] = 2.0 * mean_interval
        if idx + 1 >= MINIMUM_STEPS_FOR_VARIABILITY and len(recent) >= 2:
            stride_time_cv[fill_from:fill_to] = float(np.std(recent) / mean_interval) if mean_interval > 0 else 0.0

    # Hold the last estimate from the end of the Normal phase onward.
    if last_normal_end < n:
        step_count[last_normal_end:] = step_count[last_normal_end - 1]
        cadence_spm[last_normal_end:] = cadence_spm[last_normal_end - 1]
        step_interval_s[last_normal_end:] = step_interval_s[last_normal_end - 1]
        stride_time_s[last_normal_end:] = stride_time_s[last_normal_end - 1]
        stride_time_cv[last_normal_end:] = stride_time_cv[last_normal_end - 1]

    return _assemble(timeline, step_count, cadence_spm, step_interval_s, stride_time_s, stride_time_cv)


def _assemble(timeline, step_count, cadence_spm, step_interval_s, stride_time_s, stride_time_cv) -> pd.DataFrame:
    return pd.DataFrame({
        "frame_idx": timeline["frame_idx"].to_numpy(),
        "timestamp_s": timeline["timestamp_s"].to_numpy(),
        "step_count": step_count,
        "cadence_spm": cadence_spm,
        "step_interval_s": step_interval_s,
        "stride_time_s": stride_time_s,
        "stride_time_cv": stride_time_cv,
    })
