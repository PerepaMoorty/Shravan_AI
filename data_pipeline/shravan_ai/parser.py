"""Parse one KFall sensor CSV + its label row into a single
"trial timeline" — one row per frame carrying raw sensor values, the
ground-truth phase, and the trial's fall direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema
from .direction import DirectionResult, classify_direction

PHASE_NORMAL = "Normal"
PHASE_PRE_IMPACT = "PreImpact"
PHASE_POST_IMPACT = "PostImpact"


@dataclass
class TrialLabel:
    subject_id: str
    trial_id: str
    task_code: str
    description: str
    fall_onset_frame: float | None  # NaN/None for ADL (non-fall) trials
    fall_impact_frame: float | None


def _phase_for_frame(frame_idx: int, onset: float | None, impact: float | None) -> str:
    if onset is None or np.isnan(onset):
        return PHASE_NORMAL
    if frame_idx < onset:
        return PHASE_NORMAL
    if impact is not None and not np.isnan(impact) and frame_idx >= impact:
        return PHASE_POST_IMPACT
    return PHASE_PRE_IMPACT


def parse_trial(sensor_df: pd.DataFrame, label: TrialLabel) -> pd.DataFrame:
    """Build a trial timeline DataFrame from raw sensor rows + one label.

    Raises ValueError if fall_impact_frame precedes fall_onset_frame, or if
    either frame index falls outside the sensor recording (a bad label).
    """
    sensor_df = schema.normalize_sensor_frame(sensor_df).reset_index(drop=True)
    n_frames = len(sensor_df)

    onset, impact = label.fall_onset_frame, label.fall_impact_frame
    if onset is not None and not np.isnan(onset):
        if impact is not None and not np.isnan(impact) and impact < onset:
            raise ValueError(
                f"{label.subject_id}/{label.trial_id}: fall_impact_frame ({impact}) "
                f"precedes fall_onset_frame ({onset})"
            )
        if onset >= n_frames or (impact is not None and not np.isnan(impact) and impact >= n_frames):
            raise ValueError(
                f"{label.subject_id}/{label.trial_id}: onset/impact frame out of range "
                f"(recording has {n_frames} frames, onset={onset}, impact={impact})"
            )

    direction_result: DirectionResult = classify_direction(label.task_code, label.description)

    frame_idx = np.arange(n_frames)
    phases = [_phase_for_frame(i, onset, impact) for i in frame_idx]

    timeline = pd.DataFrame(
        {
            "frame_idx": frame_idx,
            "timestamp_s": sensor_df["TimeStamp(s)"].astype(float),
            "acc_x": sensor_df["AccX"].astype(float),
            "acc_y": sensor_df["AccY"].astype(float),
            "acc_z": sensor_df["AccZ"].astype(float),
            "gyr_x": sensor_df["GyrX"].astype(float),
            "gyr_y": sensor_df["GyrY"].astype(float),
            "gyr_z": sensor_df["GyrZ"].astype(float),
            "euler_x": sensor_df["EulerX"].astype(float),
            "euler_y": sensor_df["EulerY"].astype(float),
            "euler_z": sensor_df["EulerZ"].astype(float),
            "phase": phases,
        }
    )
    timeline["fall_direction"] = direction_result.direction
    timeline["direction_source"] = direction_result.source
    timeline["task_code"] = label.task_code
    timeline["trial_id"] = label.trial_id
    timeline["subject_id"] = label.subject_id
    return timeline
