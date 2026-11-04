"""Tests for prediction/events.py."""
import numpy as np
import pandas as pd
import pytest

from prediction.events import (
    TrialPredictions,
    evaluate_event_level,
    first_fire_frame,
    first_pre_impact_event,
    sweep_thresholds,
)


def _df(start_frames, probas):
    return pd.DataFrame({"start_frame": start_frames, "proba_pre_impact": probas})


def _trial(start_frames, probas, impact_frame, key="t"):
    return TrialPredictions(
        trial_key=(key,), subject_id="s",
        start_frames=np.asarray(start_frames), proba_pre_impact=np.asarray(probas, dtype=float),
        impact_frame=impact_frame,
    )


# --- first_pre_impact_event (DataFrame version used by the Unity export) ---

def test_fires_at_first_window_clearing_threshold():
    events = first_pre_impact_event(_df([0, 10, 20, 30], [0.1, 0.85, 0.9, 0.95]), 100, 100.0, 0.8)
    assert events == [(10, 0.10, "pre_impact_predicted", 0.85)]


def test_threshold_is_inclusive():
    assert first_pre_impact_event(_df([0, 10], [0.5, 0.8]), None, 100.0, 0.8)[0][0] == 10


def test_post_impact_windows_are_ineligible_for_fall_trials():
    assert first_pre_impact_event(_df([0, 100, 110], [0.1, 0.9, 0.99]), 100, 100.0, 0.8) == []


def test_adl_trial_any_window_can_fire():
    events = first_pre_impact_event(_df([0, 500], [0.1, 0.9]), None, 100.0, 0.8)
    assert events[0][0] == 500


def test_unsorted_input_still_returns_earliest_window():
    events = first_pre_impact_event(_df([30, 10, 20], [0.9, 0.9, 0.1]), None, 100.0, 0.8)
    assert events[0][0] == 10


# --- array version must agree with the DataFrame version ---

@pytest.mark.parametrize("impact", [None, 25, 100])
@pytest.mark.parametrize("threshold", [0.3, 0.8, 0.99])
def test_first_fire_frame_matches_dataframe_version(impact, threshold):
    rng = np.random.default_rng(0)
    frames = np.arange(0, 60, 5)
    probas = rng.random(len(frames))
    df_events = first_pre_impact_event(_df(frames, probas), impact, 100.0, threshold)
    expected = df_events[0][0] if df_events else None
    assert first_fire_frame(_trial(frames, probas, impact), threshold) == expected


# --- evaluate_event_level ---

def test_event_level_metrics():
    trials = [
        _trial([0, 10, 20], [0.1, 0.9, 0.9], impact_frame=100),   # warned, lead (100-10)/100 s = 900ms
        _trial([0, 10, 20], [0.1, 0.1, 0.85], impact_frame=50),   # warned, lead 300ms
        _trial([0, 10, 20], [0.1, 0.2, 0.3], impact_frame=100),   # missed
        _trial([0, 10, 20], [0.1, 0.95, 0.1], impact_frame=None),  # ADL false alarm
        _trial([0, 10, 20], [0.1, 0.2, 0.1], impact_frame=None),   # ADL silent
    ]
    r = evaluate_event_level(trials, threshold=0.8)
    assert (r.n_fall_trials, r.n_fall_warned) == (3, 2)
    assert r.detection_rate == pytest.approx(2 / 3)
    assert (r.n_adl_trials, r.n_adl_false_alarm) == (2, 1)
    assert r.false_alarm_rate == pytest.approx(0.5)
    assert r.mean_lead_time_ms == pytest.approx(600.0)
    assert r.median_lead_time_ms == pytest.approx(600.0)


def test_no_fall_trials_gives_none_detection_rate():
    r = evaluate_event_level([_trial([0], [0.9], None)], threshold=0.5)
    assert r.detection_rate is None and r.false_alarm_rate == 1.0


def test_sweep_is_monotone_in_threshold():
    rng = np.random.default_rng(1)
    trials = [_trial(np.arange(0, 100, 10), rng.random(10), 80 if i % 2 else None, key=i)
              for i in range(40)]
    results = sweep_thresholds(trials, np.linspace(0.05, 0.99, 20))
    det = [r.detection_rate for r in results]
    far = [r.false_alarm_rate for r in results]
    assert all(a >= b for a, b in zip(det, det[1:]))
    assert all(a >= b for a, b in zip(far, far[1:]))
