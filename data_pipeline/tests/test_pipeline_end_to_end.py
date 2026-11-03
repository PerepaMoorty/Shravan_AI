import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_synthetic_kfall import TRIALS, main as generate_synthetic
from shravan_ai.direction import DIRECTION_UNKNOWN
from shravan_ai.leadlag import compute_lead_lag
from shravan_ai.model import run_threshold_model
from shravan_ai.pipeline import run_pipeline


def test_end_to_end_pipeline(tmp_path):
    data_root = generate_synthetic(tmp_path / "sample_data")

    summary = run_pipeline(data_root=data_root, output_dir=tmp_path / "output")

    assert len(summary["processed"]) == len(TRIALS)
    assert summary["skipped"] == []
    assert summary["needs_review"] == []

    for path in summary["processed"]:
        timeline = pd.read_csv(path)
        assert timeline["fall_direction"].iloc[0] != DIRECTION_UNKNOWN

        model_log = run_threshold_model(timeline)
        result = compute_lead_lag(timeline, model_log)

        if timeline["fall_direction"].iloc[0] == "normal_only":
            assert model_log.empty, "ADL trial must never fire a model event"
        else:
            assert result.onset_timestamp_s is not None
            assert result.impact_timestamp_s is not None
