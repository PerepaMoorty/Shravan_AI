"""Event-level (one fire-or-not decision per trial) pre-impact alarm logic.

A caregiver never sees individual dense windows -- they see at most one
alarm per trial. This module holds the single shared definition of when
that alarm fires and how event-level metrics are computed from it, so
the Unity export (`scripts/export_prediction_to_unity.py`) and the
threshold-selection / evaluation scripts can't drift apart.

Firing rule: the alarm fires at the FIRST window (by `start_frame`) whose
`proba_pre_impact` is >= the threshold. For a fall trial only windows with
`start_frame < impact_frame` are eligible (a post-impact flag is not an
advance warning). For an ADL trial (no impact frame) any firing counts,
and is a false alarm.

This is deliberately a probability threshold, not the model's raw argmax
label: raw argmax (implicit threshold ~0.34) fires a false alarm on
62.6% of held-out ADL trials (see docs/COMPREHENSIVE_STATISTICAL_EVALUATION.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from prediction.lead_time import summarize_lead_times


def first_pre_impact_event(
    predictions_df: pd.DataFrame,
    impact_frame: Optional[int],
    target_rate_hz: float,
    pre_impact_threshold: float,
) -> list[tuple[int, float, str, float]]:
    """First eligible window whose `proba_pre_impact` clears
    `pre_impact_threshold`, as a one-element list of
    `(frame_idx, timestamp_s, "pre_impact_predicted", confidence)`.
    Empty list if the alarm never fires.

    `predictions_df` needs `start_frame` and `proba_pre_impact` columns.
    `timestamp_s` uses `start_frame` (not the window's end) so it is the
    true first-warning moment.
    """
    eligible = predictions_df[predictions_df["proba_pre_impact"] >= pre_impact_threshold]
    if impact_frame is not None:
        eligible = eligible[eligible["start_frame"] < impact_frame]
    if len(eligible) == 0:
        return []

    first = eligible.sort_values("start_frame").iloc[0]
    frame_idx = int(first["start_frame"])
    timestamp_s = frame_idx / target_rate_hz
    return [(frame_idx, timestamp_s, "pre_impact_predicted", float(first["proba_pre_impact"]))]


@dataclass(frozen=True)
class TrialPredictions:
    """One trial's dense-window model output, in the minimal form the
    event-level metrics need. `impact_frame is None` marks an ADL trial."""
    trial_key: tuple
    subject_id: str
    start_frames: np.ndarray        # shape (n_windows,)
    proba_pre_impact: np.ndarray    # shape (n_windows,), aligned with start_frames
    impact_frame: Optional[int]

    @property
    def is_fall(self) -> bool:
        return self.impact_frame is not None


def first_fire_frame(trial: TrialPredictions, threshold: float) -> Optional[int]:
    """Array version of `first_pre_impact_event`'s rule (same semantics,
    no DataFrame), for fast threshold sweeps. Returns the firing
    `start_frame`, or None if the alarm never fires."""
    mask = trial.proba_pre_impact >= threshold
    if trial.impact_frame is not None:
        mask &= trial.start_frames < trial.impact_frame
    if not mask.any():
        return None
    return int(trial.start_frames[mask].min())


@dataclass(frozen=True)
class EventLevelResult:
    threshold: float
    n_fall_trials: int
    n_fall_warned: int
    detection_rate: Optional[float]      # n_fall_warned / n_fall_trials
    n_adl_trials: int
    n_adl_false_alarm: int
    false_alarm_rate: Optional[float]    # n_adl_false_alarm / n_adl_trials
    mean_lead_time_ms: Optional[float]   # over warned falls only
    median_lead_time_ms: Optional[float]


def evaluate_event_level(
    trials: Sequence[TrialPredictions], threshold: float, target_rate_hz: float = 100.0
) -> EventLevelResult:
    """Event-level detection rate, false-alarm rate and lead time at one
    threshold. Lead time is reported over warned falls only; the
    detection rate is the separate number for how often a warning
    happened at all (same convention as `prediction.lead_time`)."""
    lead_times: list[Optional[float]] = []
    n_adl, n_adl_fired = 0, 0

    for trial in trials:
        fired = first_fire_frame(trial, threshold)
        if trial.is_fall:
            lead_times.append(
                None if fired is None
                else (trial.impact_frame - fired) / target_rate_hz * 1000.0
            )
        else:
            n_adl += 1
            n_adl_fired += fired is not None

    if lead_times:
        summary = summarize_lead_times(lead_times)
        n_fall, n_warned = summary.n_trials, summary.n_flagged
        detection_rate = summary.detection_rate
        mean_lt, median_lt = summary.mean_lead_time_ms, summary.median_lead_time_ms
    else:
        n_fall, n_warned, detection_rate, mean_lt, median_lt = 0, 0, None, None, None

    return EventLevelResult(
        threshold=float(threshold),
        n_fall_trials=n_fall,
        n_fall_warned=n_warned,
        detection_rate=detection_rate,
        n_adl_trials=n_adl,
        n_adl_false_alarm=n_adl_fired,
        false_alarm_rate=(n_adl_fired / n_adl) if n_adl else None,
        mean_lead_time_ms=mean_lt,
        median_lead_time_ms=median_lt,
    )


def sweep_thresholds(
    trials: Sequence[TrialPredictions], thresholds: Sequence[float], target_rate_hz: float = 100.0
) -> list[EventLevelResult]:
    return [evaluate_event_level(trials, t, target_rate_hz) for t in thresholds]
