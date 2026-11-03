"""Compute the model's lead/lag relative to ground truth, in
milliseconds, for the dashboard's timing strip readout.

Positive lead_ms means the model fired before the ground-truth frame
(good — it predicted early). Negative means it lagged behind.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .model import EVENT_FALL_DETECTED, EVENT_PRE_IMPACT_PREDICTED
from .parser import PHASE_POST_IMPACT, PHASE_PRE_IMPACT


@dataclass
class LeadLagResult:
    onset_timestamp_s: float | None
    impact_timestamp_s: float | None
    first_pre_impact_prediction_s: float | None
    first_fall_detected_s: float | None
    onset_lead_ms: float | None  # + = model predicted before true onset
    impact_lead_ms: float | None  # + = model detected before true impact


def _first_timestamp_where(timeline: pd.DataFrame, phase: str) -> float | None:
    matches = timeline.loc[timeline["phase"] == phase, "timestamp_s"]
    return float(matches.iloc[0]) if len(matches) else None


def compute_lead_lag(timeline: pd.DataFrame, model_log: pd.DataFrame) -> LeadLagResult:
    onset_ts = _first_timestamp_where(timeline, PHASE_PRE_IMPACT)
    impact_ts = _first_timestamp_where(timeline, PHASE_POST_IMPACT)

    pre_impact_events = model_log.loc[model_log["event"] == EVENT_PRE_IMPACT_PREDICTED, "timestamp_s"]
    fall_detected_events = model_log.loc[model_log["event"] == EVENT_FALL_DETECTED, "timestamp_s"]

    first_pred = float(pre_impact_events.iloc[0]) if len(pre_impact_events) else None
    first_detect = float(fall_detected_events.iloc[0]) if len(fall_detected_events) else None

    onset_lead_ms = (onset_ts - first_pred) * 1000 if onset_ts is not None and first_pred is not None else None
    impact_lead_ms = (
        (impact_ts - first_detect) * 1000 if impact_ts is not None and first_detect is not None else None
    )

    return LeadLagResult(
        onset_timestamp_s=onset_ts,
        impact_timestamp_s=impact_ts,
        first_pre_impact_prediction_s=first_pred,
        first_fall_detected_s=first_detect,
        onset_lead_ms=onset_lead_ms,
        impact_lead_ms=impact_lead_ms,
    )


def format_lead_lag(result: LeadLagResult) -> str:
    """Human-readable summary, e.g. 'Model predicted 340ms before onset'."""
    parts = []
    if result.onset_lead_ms is not None:
        if result.onset_lead_ms >= 0:
            parts.append(f"Model predicted {result.onset_lead_ms:.0f}ms before onset")
        else:
            parts.append(f"Model lagged onset by {-result.onset_lead_ms:.0f}ms")
    else:
        parts.append("Model never fired a pre-impact prediction")

    if result.impact_lead_ms is not None:
        if result.impact_lead_ms >= 0:
            parts.append(f"detected {result.impact_lead_ms:.0f}ms before impact")
        else:
            parts.append(f"detected {-result.impact_lead_ms:.0f}ms after impact")
    else:
        parts.append("never confirmed the fall")

    return "; ".join(parts)
