"""Paths and layout conventions for the KFall dataset on disk.

Expected layout (matches the public KFall release):

    <DATA_ROOT>/
        sensor_data/SA06/S06T01R01.csv ...
        label_data/SA06_label.xlsx (or .csv) ...

Sensor filenames encode <Subject><TaskID><Repetition>, e.g. S06T01R01.csv.

DATA_ROOT resolves to <repo root>/datasets/kfall_dataset when present,
otherwise to the synthetic sample_data/ set. SHRAVAN_KFALL_ROOT overrides
both. See docs/KFALL_TASK_CODE_MAPPING.md for how label sheets are parsed.
"""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # data_pipeline/
_REPO_ROOT = _PACKAGE_ROOT.parent

DEFAULT_REAL_DATA_ROOT = _REPO_ROOT / "datasets" / "kfall_dataset"
DEFAULT_SYNTHETIC_DATA_ROOT = _PACKAGE_ROOT / "sample_data"


def _default_data_root() -> Path:
    if DEFAULT_REAL_DATA_ROOT.exists():
        return DEFAULT_REAL_DATA_ROOT
    return DEFAULT_SYNTHETIC_DATA_ROOT


DATA_ROOT = Path(os.environ.get("SHRAVAN_KFALL_ROOT", str(_default_data_root())))


def describe_data_root() -> str:
    """Label for whichever DATA_ROOT resolved, printed by every script that
    reads raw data so real vs. synthetic is never ambiguous."""
    resolved = DATA_ROOT.resolve() if DATA_ROOT.exists() else DATA_ROOT
    if resolved == DEFAULT_REAL_DATA_ROOT.resolve():
        kind = "real KFall dataset (default)"
    elif DEFAULT_SYNTHETIC_DATA_ROOT.exists() and resolved == DEFAULT_SYNTHETIC_DATA_ROOT.resolve():
        kind = "SYNTHETIC sample dataset"
    elif "SHRAVAN_KFALL_ROOT" in os.environ:
        kind = "custom path via SHRAVAN_KFALL_ROOT"
    else:
        kind = "unrecognized default"
    return f"{DATA_ROOT}  [{kind}]"


TASK_CODE_MAP_PATH = Path(__file__).resolve().parent / "task_code_map.csv"

OUTPUT_DIR = Path(os.environ.get("SHRAVAN_OUTPUT_DIR", _PACKAGE_ROOT / "output"))
TRIAL_TIMELINE_DIR = OUTPUT_DIR / "trial_timelines"
MODEL_LOG_DIR = OUTPUT_DIR / "model_logs"
SANITY_PLOT_DIR = OUTPUT_DIR / "sanity_plots"
