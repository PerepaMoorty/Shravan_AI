"""Tests for the k-fold helpers in prediction/loso.py and
PrecomputedWindowDataset in prediction/torch_dataset.py."""
import numpy as np
import pandas as pd
import pytest

from prediction.loso import make_subject_kfolds, pick_validation_subjects
from prediction.torch_dataset import PrecomputedWindowDataset

SUBJECTS = [f"kfall_SA{i:02d}" for i in range(6, 38)]  # 32 subjects


def test_kfolds_partition_all_subjects_disjointly():
    folds = make_subject_kfolds(SUBJECTS, 5)
    flat = [s for f in folds for s in f]
    assert sorted(flat) == sorted(SUBJECTS) and len(flat) == len(set(flat))
    assert sorted(len(f) for f in folds) == [6, 6, 6, 7, 7]


def test_kfolds_reproducible_and_order_independent():
    assert make_subject_kfolds(SUBJECTS, 5, seed=1) == make_subject_kfolds(SUBJECTS[::-1], 5, seed=1)
    assert make_subject_kfolds(SUBJECTS, 5, seed=1) != make_subject_kfolds(SUBJECTS, 5, seed=2)


def test_kfolds_rejects_bad_n_folds():
    with pytest.raises(ValueError):
        make_subject_kfolds(SUBJECTS[:3], 4)


def test_pick_validation_subjects():
    pool = SUBJECTS[:25]
    val = pick_validation_subjects(pool, 5, seed=3)
    assert len(val) == 5 and set(val) <= set(pool)
    assert val == pick_validation_subjects(pool[::-1], 5, seed=3)
    with pytest.raises(ValueError):
        pick_validation_subjects(pool[:2], 2, seed=0)


def test_precomputed_dataset_reads_rows_via_cache_row():
    windows = np.arange(5 * 9 * 4, dtype=np.float32).reshape(5, 9, 4)
    df = pd.DataFrame({"cache_row": [3, 0], "label_id": [1, 2]})
    ds = PrecomputedWindowDataset(df, windows)
    x, y = ds[0]
    assert len(ds) == 2 and np.array_equal(x.numpy(), windows[3]) and int(y) == 1
    assert int(ds[1][1]) == 2


def test_precomputed_dataset_requires_cache_row():
    with pytest.raises(ValueError):
        PrecomputedWindowDataset(pd.DataFrame({"label_id": [0]}), np.zeros((1, 9, 4)))
