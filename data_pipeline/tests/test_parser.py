import numpy as np
import pandas as pd
import pytest

from shravan_ai.parser import PHASE_NORMAL, PHASE_POST_IMPACT, PHASE_PRE_IMPACT, TrialLabel, parse_trial


def _sensor_df(n_frames=20):
    return pd.DataFrame(
        {
            "TimeStamp(s)": np.arange(n_frames) / 10.0,
            "FrameCounter": np.arange(n_frames),
            "AccX": np.zeros(n_frames),
            "AccY": np.zeros(n_frames),
            "AccZ": np.ones(n_frames),
            "GyrX": np.zeros(n_frames),
            "GyrY": np.zeros(n_frames),
            "GyrZ": np.zeros(n_frames),
            "EulerX": np.zeros(n_frames),
            "EulerY": np.zeros(n_frames),
            "EulerZ": np.zeros(n_frames),
        }
    )


def test_fall_trial_phases_split_at_onset_and_impact():
    label = TrialLabel(
        subject_id="SA06", trial_id="T1", task_code="F09",
        description="Forward fall while walking caused by a trip",
        fall_onset_frame=10, fall_impact_frame=14,
    )
    timeline = parse_trial(_sensor_df(20), label)

    assert list(timeline.loc[timeline["frame_idx"] < 10, "phase"].unique()) == [PHASE_NORMAL]
    assert list(timeline.loc[(timeline["frame_idx"] >= 10) & (timeline["frame_idx"] < 14), "phase"].unique()) == [PHASE_PRE_IMPACT]
    assert list(timeline.loc[timeline["frame_idx"] >= 14, "phase"].unique()) == [PHASE_POST_IMPACT]
    assert (timeline["fall_direction"] == "fall_forward").all()


def test_adl_trial_is_normal_for_entire_recording():
    label = TrialLabel(
        subject_id="SA06", trial_id="T2", task_code="D01",
        description="Walking slowly", fall_onset_frame=None, fall_impact_frame=None,
    )
    timeline = parse_trial(_sensor_df(20), label)
    assert (timeline["phase"] == PHASE_NORMAL).all()
    assert (timeline["fall_direction"] == "normal_only").all()


def test_impact_before_onset_raises():
    label = TrialLabel(
        subject_id="SA06", trial_id="T3", task_code="F09", description="Forward fall",
        fall_onset_frame=14, fall_impact_frame=10,
    )
    with pytest.raises(ValueError, match="precedes"):
        parse_trial(_sensor_df(20), label)


def test_out_of_range_frame_raises():
    label = TrialLabel(
        subject_id="SA06", trial_id="T4", task_code="F09", description="Forward fall",
        fall_onset_frame=10, fall_impact_frame=100,
    )
    with pytest.raises(ValueError, match="out of range"):
        parse_trial(_sensor_df(20), label)


def test_column_aliases_are_normalized():
    df = _sensor_df(5).rename(columns={"TimeStamp(s)": "time", "AccX": "accx", "GyrZ": "GyrZ "})
    label = TrialLabel(
        subject_id="SA06", trial_id="T5", task_code="D01", description="Walking",
        fall_onset_frame=None, fall_impact_frame=None,
    )
    timeline = parse_trial(df, label)
    assert len(timeline) == 5
