"""Column names for KFall sensor CSVs and label files.

Real-world exports vary a bit in header spelling/casing across KFall
releases, so lookups go through `find_column`, which matches case- and
punctuation-insensitively instead of hard-requiring one exact spelling.
"""

from __future__ import annotations

import re

import pandas as pd

SENSOR_COLUMNS = [
    "TimeStamp(s)",
    "FrameCounter",
    "AccX",
    "AccY",
    "AccZ",
    "GyrX",
    "GyrY",
    "GyrZ",
    "EulerX",
    "EulerY",
    "EulerZ",
]

# Aliases seen across KFall CSV export variants -> canonical column name.
SENSOR_COLUMN_ALIASES = {
    "timestamps": "TimeStamp(s)",
    "time": "TimeStamp(s)",
    "framecounter": "FrameCounter",
    "frame": "FrameCounter",
    "accx": "AccX",
    "accy": "AccY",
    "accz": "AccZ",
    "gyrx": "GyrX",
    "gyry": "GyrY",
    "gyrz": "GyrZ",
    "eulerx": "EulerX",
    "eulery": "EulerY",
    "eulerz": "EulerZ",
}

LABEL_COLUMN_ALIASES = {
    "taskcode": "Task Code",
    "task_code": "Task Code",
    "taskcodetaskid": "Task Code",  # real KFall header: "Task Code (Task ID)"
    "description": "Description",
    "trialid": "Trial ID",
    "trial_id": "Trial ID",
    "fallonsetframe": "Fall_onset_frame",
    "fall_onset_frame": "Fall_onset_frame",
    "fallimpactframe": "Fall_impact_frame",
    "fall_impact_frame": "Fall_impact_frame",
}


_ALL_ALIASES = {**SENSOR_COLUMN_ALIASES, **LABEL_COLUMN_ALIASES}


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _canonical_norm(name: str) -> str:
    """Normalize a column name, then resolve it through the alias table to
    its canonical form's normalized spelling — so both a canonical name and
    any of its aliases collapse to the same key for comparison."""
    norm = _normalize(name)
    alias = _ALL_ALIASES.get(norm)
    return _normalize(alias) if alias else norm


def find_column(columns, canonical_or_alias: str) -> str:
    """Return the actual column name in `columns` matching a canonical name
    or any known alias, regardless of case/punctuation. Raises KeyError with
    the available columns listed if nothing matches, so bad headers fail
    loudly at parse time instead of silently producing NaNs.
    """
    target = _canonical_norm(canonical_or_alias)
    for col in columns:
        if _canonical_norm(col) == target:
            return col
    raise KeyError(
        f"Column matching '{canonical_or_alias}' not found. Available columns: {list(columns)}"
    )


def normalize_sensor_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename an arbitrary KFall sensor CSV's columns onto SENSOR_COLUMNS."""
    rename = {}
    for canonical in SENSOR_COLUMNS:
        actual = find_column(df.columns, canonical)
        rename[actual] = canonical
    return df.rename(columns=rename)[SENSOR_COLUMNS]
