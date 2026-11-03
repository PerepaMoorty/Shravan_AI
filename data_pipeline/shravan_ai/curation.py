"""Configurable selection of which processed trials get exported to Unity.

This is the "modularity" layer: which real KFall subjects the digital twin
actually shows is controlled by editing unity_export_config.json, not by
touching code. Re-run `scripts/export_for_unity.py` after editing it to
swap the demo to a different subject range.

Selection is direction-aware and asymmetric on purpose:
  - Fall trials are preferred when the model's log actually fired on them
    (showcases detection working, not a trial the stub model missed).
  - ADL (normal_only) trials are preferred when the model log is EMPTY
    (showcases the no-false-positive guarantee, which is the point of
    keeping an ADL trial in the demo at all).
Trials with unresolved direction (unknown_direction) are always excluded —
there's nothing coherent to demo about a trial whose label text couldn't
be classified.

Selection round-robins across subjects within each direction so a
single subject can't fill every slot.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from .direction import DIRECTION_NORMAL, DIRECTION_UNKNOWN

log = logging.getLogger("shravan_ai.curation")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "unity_export_config.json"

# Matches "SA06-SA10" or "S06-S10" -> expands to every subject ID in that
# inclusive numeric range that actually exists in the processed output.
SUBJECT_RANGE_RE = re.compile(r"^(?P<prefix>[A-Za-z]*)(?P<start>\d+)\s*-\s*(?:[A-Za-z]*)(?P<end>\d+)$")


def expand_subject_spec(spec: str, known_subjects: set[str]) -> list[str]:
    """Expand one config entry: a literal subject ID ("SA06") or an
    inclusive numeric range ("SA06-SA10"). Unknown/out-of-range subjects
    are dropped silently here; the caller logs the aggregate diff."""
    spec = spec.strip()
    range_match = SUBJECT_RANGE_RE.match(spec)
    if not range_match:
        return [spec] if spec in known_subjects else []

    prefix = range_match.group("prefix")
    width = len(range_match.group("start"))
    start, end = int(range_match.group("start")), int(range_match.group("end"))

    expanded = []
    for n in range(start, end + 1):
        candidate = f"{prefix}{n:0{width}d}"
        if candidate in known_subjects:
            expanded.append(candidate)
    return expanded


def load_selection_config(path: Path | None = None) -> dict:
    path = path or DEFAULT_CONFIG_PATH
    return json.loads(path.read_text())


def _read_trial_head(path: Path) -> tuple[str, str] | None:
    df = pd.read_csv(path, nrows=1)
    if df.empty:
        return None
    return str(df["subject_id"].iloc[0]), str(df["fall_direction"].iloc[0])


def _has_model_events(model_log_path: Path) -> bool:
    if not model_log_path.exists():
        return False
    # Header-only CSV (no fired events) is a handful of bytes; anything
    # meaningfully larger has at least one event row.
    return model_log_path.stat().st_size > len("frame_idx,timestamp_s,event,confidence\n") + 2


def select_trials_for_unity(timeline_dir: Path, model_log_dir: Path, cfg: dict) -> tuple[list[str], dict]:
    """Return (selected trial keys, stats) for export, per cfg's subject
    range and per-direction caps.
    """
    timeline_paths = sorted(p for p in timeline_dir.glob("*.csv") if not p.name.startswith("_"))

    by_subject_direction: dict[tuple[str, str], list[dict]] = {}
    known_subjects: set[str] = set()

    for path in timeline_paths:
        head = _read_trial_head(path)
        if head is None:
            continue
        subject_id, direction = head
        known_subjects.add(subject_id)
        if direction == DIRECTION_UNKNOWN:
            continue

        trial_key = path.stem
        model_log_path = model_log_dir / f"{trial_key}_model_log.csv"
        by_subject_direction.setdefault((subject_id, direction), []).append(
            {"trial_key": trial_key, "has_model_events": _has_model_events(model_log_path)}
        )

    subjects_cfg = cfg.get("subjects") or []
    range_cfg = cfg.get("subject_range")
    requested: list[str] = []
    if subjects_cfg:
        for spec in subjects_cfg:
            requested.extend(expand_subject_spec(spec, known_subjects))
    elif range_cfg:
        requested.extend(expand_subject_spec(range_cfg, known_subjects))

    requested_set = set(requested) if (subjects_cfg or range_cfg) else known_subjects
    configured_but_missing = set(subjects_cfg) - known_subjects if subjects_cfg else set()
    if configured_but_missing:
        log.warning("Configured subjects not found in processed output, skipping: %s", sorted(configured_but_missing))

    max_per_fall_direction = int(cfg.get("max_trials_per_direction", 3))
    max_adl = int(cfg.get("max_adl_trials", max_per_fall_direction))
    prefer_showcase = bool(cfg.get("prefer_trials_that_showcase_the_model", True))

    ordered_subjects = sorted(requested_set & known_subjects)
    selected: list[str] = []
    stats = {"by_direction": {}, "subjects_used": ordered_subjects}

    directions_present = sorted({d for (_, d) in by_subject_direction})
    for direction in directions_present:
        cap = max_adl if direction == DIRECTION_NORMAL else max_per_fall_direction

        # Rank each subject's candidates, then take one per subject per round.
        per_subject_ranked: list[list[dict]] = []
        for subject_id in ordered_subjects:
            trials = by_subject_direction.get((subject_id, direction))
            if not trials:
                continue
            if prefer_showcase and direction == DIRECTION_NORMAL:
                trials = sorted(trials, key=lambda t: (t["has_model_events"], t["trial_key"]))  # prefer NO events
            elif prefer_showcase:
                trials = sorted(trials, key=lambda t: (not t["has_model_events"], t["trial_key"]))  # prefer events
            else:
                trials = sorted(trials, key=lambda t: t["trial_key"])
            per_subject_ranked.append(trials)

        chosen: list[str] = []
        round_idx = 0
        while len(chosen) < cap and any(round_idx < len(pool) for pool in per_subject_ranked):
            for pool in per_subject_ranked:
                if len(chosen) >= cap:
                    break
                if round_idx < len(pool):
                    chosen.append(pool[round_idx]["trial_key"])
            round_idx += 1

        selected.extend(chosen)
        stats["by_direction"][direction] = chosen

    return selected, stats
