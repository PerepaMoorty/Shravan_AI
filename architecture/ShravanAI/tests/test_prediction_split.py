"""Tests for prediction/split.py."""
import pandas as pd
import pytest

from prediction.split import PredictionSplitConfig, split_by_subject


def _synthetic_df(n_subjects: int = 10, windows_per_subject: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n_subjects):
        subject_id = f"SA{i:02d}"
        global_subject_id = f"kfall_{subject_id}"
        for w in range(windows_per_subject):
            rows.append({
                "dataset": "kfall",
                "subject_id": subject_id,
                "global_subject_id": global_subject_id,
                "label_id": w % 3,
                "some_feature": float(w),
            })
    return pd.DataFrame(rows)


def test_split_produces_no_subject_leakage():
    df = _synthetic_df()
    train_df, val_df, test_df = split_by_subject(df, PredictionSplitConfig(random_state=0))

    train_subjects = set(train_df["global_subject_id"])
    val_subjects = set(val_df["global_subject_id"])
    test_subjects = set(test_df["global_subject_id"])

    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)


def test_split_covers_every_window_exactly_once():
    df = _synthetic_df()
    train_df, val_df, test_df = split_by_subject(df, PredictionSplitConfig(random_state=0))

    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_split_sizes_roughly_match_requested_fractions():
    df = _synthetic_df(n_subjects=20, windows_per_subject=5)
    train_df, val_df, test_df = split_by_subject(
        df, PredictionSplitConfig(val_size=0.2, test_size=0.2, random_state=0)
    )

    total = len(df)
    test_fraction = len(test_df) / total
    val_fraction = len(val_df) / total
    assert 0.05 < test_fraction < 0.4
    assert 0.05 < val_fraction < 0.4


def test_split_is_reproducible_with_same_random_state():
    df = _synthetic_df()
    train_a, _, test_a = split_by_subject(df, PredictionSplitConfig(random_state=42))
    train_b, _, test_b = split_by_subject(df, PredictionSplitConfig(random_state=42))

    assert set(train_a["global_subject_id"]) == set(train_b["global_subject_id"])
    assert set(test_a["global_subject_id"]) == set(test_b["global_subject_id"])


def test_split_differs_with_different_random_state():
    df = _synthetic_df(n_subjects=15)
    _, _, test_a = split_by_subject(df, PredictionSplitConfig(random_state=1))
    _, _, test_b = split_by_subject(df, PredictionSplitConfig(random_state=2))

    assert set(test_a["global_subject_id"]) != set(test_b["global_subject_id"])


def test_split_raises_clear_error_on_too_few_subjects():
    df = _synthetic_df(n_subjects=2)

    with pytest.raises(ValueError, match="distinct global_subject_id"):
        split_by_subject(df, PredictionSplitConfig(random_state=0))
