"""Subject-aware train/val/test splitting for the prediction pipeline.

Same leakage concern as `detection/split.py` (a per-window random split
would put windows from the same subject's same trial -- which overlap
even more heavily here, at 0.1s stride -- into different splits) and
the same fix: split by SUBJECT (`global_subject_id`), never by window.

Deliberately its OWN module rather than importing `detection.split`,
per the blueprint's no-cross-import rule between the two pipelines
(already the reason `prediction/dataset.py` duplicates rather than
imports `detection/dataset.py`'s `load_window`). Simpler than
detection's version in one respect: prediction is KFall-only by
design (`shared.manifest.query_prediction_trials` already filters to
just KFall), so there's no per-dataset-then-combine step needed --
just one `GroupShuffleSplit` pass.

This single train/val/test split is a deliberately lighter-weight
alternative to `prediction/loso.py`'s full leave-one-subject-out folds
-- one trained model instead of one-per-subject, at the cost of a less
rigorous per-subject evaluation. Use this for getting a real (trained,
non-random) model quickly; use LOSO for a proper per-subject report
later.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class PredictionSplitConfig:
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42


def split_by_subject(
    df: pd.DataFrame, config: PredictionSplitConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` (one row per window, must have a `global_subject_id`
    column) into (train_df, val_df, test_df), by subject.

    No `global_subject_id` ever appears in more than one of the three
    returned frames -- verified as a hard post-condition before
    returning, not just intended (same pattern as
    `detection.split.split_by_subject`).

    Raises ValueError if there aren't enough distinct subjects to carve
    out non-empty val/test splits.
    """
    config = config or PredictionSplitConfig()

    n_subjects = df["global_subject_id"].nunique()
    if n_subjects < 3:
        raise ValueError(
            f"Need at least 3 distinct global_subject_id values (one per "
            f"split) to carve out a non-empty train/val/test split; got "
            f"{n_subjects}. This is expected on small fixture data; on "
            f"real data, check the input df actually has real subject "
            f"variety."
        )

    df = df.reset_index(drop=True)
    try:
        test_splitter = GroupShuffleSplit(
            n_splits=1, test_size=config.test_size, random_state=config.random_state
        )
        trainval_idx, test_idx = next(
            test_splitter.split(df, groups=df["global_subject_id"])
        )

        trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # val_size was specified as a fraction of the ORIGINAL total,
        # but we're now splitting only the trainval remainder -- rescale
        # so the actual val fraction of the original total still matches
        # what the caller asked for (same rationale as detection.split).
        relative_val_size = config.val_size / (1 - config.test_size)
        val_splitter = GroupShuffleSplit(
            n_splits=1, test_size=relative_val_size, random_state=config.random_state
        )
        train_idx, val_idx = next(
            val_splitter.split(trainval_df, groups=trainval_df["global_subject_id"])
        )
    except ValueError as exc:
        raise ValueError(
            f"Could not split {n_subjects} subjects into train/val/test "
            f"with val_size={config.val_size}, test_size={config.test_size}: {exc}"
        ) from exc

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    _assert_no_subject_leakage(train_df, val_df, test_df)
    return train_df, val_df, test_df


def _assert_no_subject_leakage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    train_subjects = set(train_df["global_subject_id"])
    val_subjects = set(val_df["global_subject_id"])
    test_subjects = set(test_df["global_subject_id"])

    overlaps = {
        "train/val": train_subjects & val_subjects,
        "train/test": train_subjects & test_subjects,
        "val/test": val_subjects & test_subjects,
    }
    leaking = {k: v for k, v in overlaps.items() if v}
    if leaking:
        raise AssertionError(
            f"Subject leakage across splits (should never happen -- this "
            f"is a bug in split_by_subject, not a data issue): {leaking}"
        )
