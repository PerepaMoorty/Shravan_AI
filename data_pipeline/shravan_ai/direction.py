"""Translate a KFall trial's task code / description into the
direction label Unity's Animator uses (fall_forward / fall_backward /
fall_lateral / normal_only).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config

ADL_TASK_PREFIX = "D"

DIRECTION_FORWARD = "fall_forward"
DIRECTION_BACKWARD = "fall_backward"
DIRECTION_LATERAL = "fall_lateral"
DIRECTION_NORMAL = "normal_only"
DIRECTION_UNKNOWN = "unknown_direction"

_KEYWORDS = [
    (DIRECTION_LATERAL, ("lateral", "sideway", "side ", " left", " right")),
    (DIRECTION_FORWARD, ("forward",)),
    (DIRECTION_BACKWARD, ("backward", "back ")),
]


@dataclass
class DirectionResult:
    direction: str
    source: str  # "adl" | "override" | "description" | "unresolved"


def _load_override_map() -> dict[str, str]:
    if not config.TASK_CODE_MAP_PATH.exists():
        return {}
    df = pd.read_csv(config.TASK_CODE_MAP_PATH, comment="#", skip_blank_lines=True)
    df = df.dropna(subset=["task_code", "direction_override"])
    return dict(zip(df["task_code"].astype(str).str.strip(), df["direction_override"].str.strip()))


def classify_direction(task_code: str, description: str) -> DirectionResult:
    """ADL codes are normal_only; otherwise task_code_map.csv overrides win,
    then a keyword match on the description."""
    code = (task_code or "").strip().upper()

    if code.startswith(ADL_TASK_PREFIX):
        return DirectionResult(DIRECTION_NORMAL, "adl")

    overrides = _load_override_map()
    if code in overrides:
        return DirectionResult(overrides[code], "override")

    desc = f" {(description or '').strip().lower()} "
    for direction, keywords in _KEYWORDS:
        if any(kw in desc for kw in keywords):
            return DirectionResult(direction, "description")

    return DirectionResult(DIRECTION_UNKNOWN, "unresolved")
