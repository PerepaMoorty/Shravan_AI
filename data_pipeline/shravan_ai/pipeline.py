"""Batch runner: walk DATA_ROOT, pair every sensor CSV with
its label row, and write one trial-timeline CSV per trial to
OUTPUT_DIR/trial_timelines/. Trials whose direction couldn't be resolved are
collected into _needs_review.csv instead of being silently guessed.

Real KFall label sheets (.xlsx) only document fall trials (F01-F15) — ADL
recordings have no label row at all, and the fall-trial rows use Excel
merged cells (task code/description only populated on each block's first
row) plus a combined "Task Code (Task ID)" field like "F01 (20)", where the
number in parentheses — not the visible Trial ID column — is what the
sensor filename's T-number actually refers to. All of that is handled here
so callers just get a clean row per trial regardless of source format.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from . import config, schema
from .direction import DIRECTION_UNKNOWN
from .parser import TrialLabel, parse_trial

log = logging.getLogger("shravan_ai.pipeline")

SUBJECT_LABEL_RE = re.compile(r"(?P<subject>[A-Za-z]*\d+)_label\.(csv|xlsx)$", re.IGNORECASE)
TASK_CODE_RE = re.compile(r"^\s*(?P<code>[A-Za-z]+\d+)\s*(?:\(\s*(?P<task_id>\d+)\s*\))?")

ADL_SYNTHESIZED_TASK_CODE = "D00"


def _iter_label_files(label_dir: Path):
    paths = sorted(list(label_dir.glob("*.csv")) + list(label_dir.glob("*.xlsx")))
    for path in paths:
        match = SUBJECT_LABEL_RE.search(path.name)
        if not match:
            log.warning("Skipping label file with unrecognized name: %s", path)
            continue
        yield match.group("subject"), path


def _parse_task_code(raw: str) -> tuple[str, int | None]:
    """Split "F01 (20)" into ("F01", 20); plain "F01" into ("F01", None)."""
    match = TASK_CODE_RE.match(raw)
    if not match:
        return raw.strip(), None
    task_id = match.group("task_id")
    return match.group("code"), (int(task_id) if task_id else None)


def _subject_digits(subject_id: str) -> str | None:
    match = re.search(r"\d+", subject_id)
    return match.group() if match else None


def _load_label_rows(label_path: Path) -> list[dict]:
    if label_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(label_path)
    else:
        df = pd.read_csv(label_path)
    df.columns = [str(c).strip() for c in df.columns]

    task_col = schema.find_column(df.columns, "Task Code")
    df[task_col] = df[task_col].ffill()  # undo Excel's merged-cell blanking
    desc_col = _safe_find(df.columns, "Description")
    if desc_col:
        df[desc_col] = df[desc_col].ffill()

    rows = []
    for _, row in df.iterrows():
        code, task_id = _parse_task_code(str(row[task_col]))
        rows.append(
            {
                "task_code": code,
                "task_id": task_id,
                "description": str(row.get(desc_col, "")) if desc_col else "",
                "trial_id": str(row[schema.find_column(df.columns, "Trial ID")]).strip(),
                "fall_onset_frame": _to_float_or_none(
                    row.get(_safe_find(df.columns, "Fall_onset_frame"))
                ),
                "fall_impact_frame": _to_float_or_none(
                    row.get(_safe_find(df.columns, "Fall_impact_frame"))
                ),
            }
        )
    return rows


def _safe_find(columns, name: str):
    try:
        return schema.find_column(columns, name)
    except KeyError:
        return None


def _to_float_or_none(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_sensor_csv(
    subject_dir: Path, subject_digits: str | None, task_id: int | None, trial_id: str
) -> Path | None:
    if not subject_dir.exists():
        return None

    # Synthetic/sample data: trial_id is already the full filename stem.
    exact = subject_dir / f"{trial_id}.csv"
    if exact.exists():
        return exact

    # Real KFall data: trial_id is just the repetition number: reconstruct
    # the S<subject>T<task_id>R<rep> filename from the label row's own
    # parsed task ID rather than guessing from Trial ID alone.
    if task_id is not None and subject_digits is not None:
        try:
            rep_num = int(trial_id)
        except ValueError:
            rep_num = None
        if rep_num is not None:
            candidate = subject_dir / f"S{subject_digits}T{task_id:02d}R{rep_num:02d}.csv"
            if candidate.exists():
                return candidate

    # Last-resort fallback for anything that doesn't fit either convention.
    candidates = [p for p in subject_dir.glob("*.csv") if str(trial_id).lower() in p.stem.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        log.warning("Ambiguous sensor file match for trial '%s' in %s: %s", trial_id, subject_dir, candidates)
    return None


def _process_trial(
    subject_id: str, label: TrialLabel, sensor_csv: Path, timeline_dir: Path,
    processed: list, skipped: list, needs_review: list,
) -> None:
    try:
        sensor_df = pd.read_csv(sensor_csv)
        timeline = parse_trial(sensor_df, label)
    except (ValueError, KeyError) as exc:
        skipped.append(f"{subject_id}/{label.trial_id}: {exc}")
        return

    out_path = timeline_dir / f"{subject_id}_{label.trial_id}.csv"
    timeline.to_csv(out_path, index=False)
    processed.append(str(out_path))

    if (timeline["fall_direction"] == DIRECTION_UNKNOWN).any():
        needs_review.append(
            f"{subject_id}/{label.trial_id}: task_code={label.task_code} "
            f"description={label.description!r} -> could not classify direction"
        )


def run_pipeline(data_root: Path | None = None, output_dir: Path | None = None) -> dict:
    """Process every trial under data_root, writing one timeline CSV each.

    Returns a summary dict: {"processed": int, "skipped": [...], "needs_review": [...]}.
    """
    data_root = Path(data_root) if data_root else config.DATA_ROOT
    output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
    timeline_dir = output_dir / "trial_timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    sensor_dir = data_root / "sensor_data"
    label_dir = data_root / "label_data"
    if not sensor_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Expected {sensor_dir} and {label_dir} to exist. "
            f"Set SHRAVAN_KFALL_ROOT to the KFall dataset root."
        )

    processed, skipped, needs_review = [], [], []

    for subject_id, label_path in _iter_label_files(label_dir):
        subject_sensor_dir = sensor_dir / subject_id
        subject_digits = _subject_digits(subject_id)
        claimed_files: set[Path] = set()

        for row in _load_label_rows(label_path):
            task_id = row.pop("task_id")
            sensor_csv = _find_sensor_csv(subject_sensor_dir, subject_digits, task_id, row["trial_id"])
            if sensor_csv is None:
                skipped.append(
                    f"{subject_id}/{row['task_code']} rep {row['trial_id']}: no matching sensor CSV found"
                )
                continue
            claimed_files.add(sensor_csv.resolve())
            row["trial_id"] = sensor_csv.stem  # use the real filename stem downstream, not the bare rep number

            label = TrialLabel(subject_id=subject_id, **row)
            _process_trial(subject_id, label, sensor_csv, timeline_dir, processed, skipped, needs_review)

        # Real KFall label sheets only document fall trials. Any sensor
        # recording not claimed by a labeled row above has no fall to
        # report, so it's an ADL — synthesize a normal_only label for it
        # rather than silently dropping it.
        if subject_sensor_dir.exists():
            for sensor_csv in sorted(subject_sensor_dir.glob("*.csv")):
                if sensor_csv.resolve() in claimed_files:
                    continue
                label = TrialLabel(
                    subject_id=subject_id,
                    task_code=ADL_SYNTHESIZED_TASK_CODE,
                    description="ADL (no fall label row in source data)",
                    trial_id=sensor_csv.stem,
                    fall_onset_frame=None,
                    fall_impact_frame=None,
                )
                _process_trial(subject_id, label, sensor_csv, timeline_dir, processed, skipped, needs_review)

    if needs_review:
        review_path = timeline_dir / "_needs_review.csv"
        pd.DataFrame({"issue": needs_review}).to_csv(review_path, index=False)

    summary = {"processed": processed, "skipped": skipped, "needs_review": needs_review}
    log.info(
        "Processed %d trials, skipped %d, %d need manual direction review",
        len(processed), len(skipped), len(needs_review),
    )
    return summary
