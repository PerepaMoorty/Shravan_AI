import pandas as pd

from shravan_ai.curation import expand_subject_spec, select_trials_for_unity


def test_literal_subject_passthrough():
    known = {"SA06", "SA07", "SA12"}
    assert expand_subject_spec("SA06", known) == ["SA06"]


def test_literal_subject_not_known_is_dropped():
    known = {"SA06"}
    assert expand_subject_spec("SA99", known) == []


def test_range_expands_to_known_subjects_only():
    known = {"SA06", "SA07", "SA08", "SA10"}  # SA09 deliberately missing (real dataset has gaps)
    assert expand_subject_spec("SA06-SA10", known) == ["SA06", "SA07", "SA08", "SA10"]


def test_range_preserves_zero_padded_width():
    known = {"SA06", "SA07", "SA08", "SA09"}
    result = expand_subject_spec("SA06-SA09", known)
    assert result == ["SA06", "SA07", "SA08", "SA09"]


def test_range_outside_known_subjects_is_empty():
    known = {"SA06"}
    assert expand_subject_spec("SA20-SA25", known) == []


def _write_timeline(path, subject_id, direction):
    pd.DataFrame({"subject_id": [subject_id], "fall_direction": [direction]}).to_csv(path, index=False)


def _write_model_log(path, has_events):
    if has_events:
        pd.DataFrame(
            {"frame_idx": [10], "timestamp_s": [1.0], "event": ["fall_detected"], "confidence": [0.9]}
        ).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=["frame_idx", "timestamp_s", "event", "confidence"]).to_csv(path, index=False)


def test_selection_round_robins_across_subjects_instead_of_one_dominating(tmp_path):
    timeline_dir = tmp_path / "trial_timelines"
    model_log_dir = tmp_path / "model_logs"
    timeline_dir.mkdir()
    model_log_dir.mkdir()

    # SA06 alone has 5 qualifying trials; SA07 has 1. A flat top-N ranking
    # (sorted by trial_key, which starts with subject_id) would fill both
    # slots from SA06 since "SA06_..." always sorts before "SA07_...".
    for i in range(1, 6):
        key = f"SA06_T{i}"
        _write_timeline(timeline_dir / f"{key}.csv", "SA06", "fall_forward")
        _write_model_log(model_log_dir / f"{key}_model_log.csv", has_events=True)

    key = "SA07_T1"
    _write_timeline(timeline_dir / f"{key}.csv", "SA07", "fall_forward")
    _write_model_log(model_log_dir / f"{key}_model_log.csv", has_events=True)

    cfg = {"subject_range": "SA06-SA07", "max_trials_per_direction": 2, "max_adl_trials": 2}
    _, stats = select_trials_for_unity(timeline_dir, model_log_dir, cfg)

    subjects_in_selection = {key.split("_")[0] for key in stats["by_direction"]["fall_forward"]}
    assert subjects_in_selection == {"SA06", "SA07"}, f"expected both subjects, got {stats['by_direction']}"


def test_fall_trials_prefer_ones_with_model_events(tmp_path):
    timeline_dir = tmp_path / "trial_timelines"
    model_log_dir = tmp_path / "model_logs"
    timeline_dir.mkdir()
    model_log_dir.mkdir()

    _write_timeline(timeline_dir / "SA06_Missed.csv", "SA06", "fall_forward")
    _write_model_log(model_log_dir / "SA06_Missed_model_log.csv", has_events=False)
    _write_timeline(timeline_dir / "SA06_Caught.csv", "SA06", "fall_forward")
    _write_model_log(model_log_dir / "SA06_Caught_model_log.csv", has_events=True)

    cfg = {"subject_range": "SA06", "max_trials_per_direction": 1}
    _, stats = select_trials_for_unity(timeline_dir, model_log_dir, cfg)

    assert stats["by_direction"]["fall_forward"] == ["SA06_Caught"]


def test_adl_trials_prefer_ones_without_model_events(tmp_path):
    timeline_dir = tmp_path / "trial_timelines"
    model_log_dir = tmp_path / "model_logs"
    timeline_dir.mkdir()
    model_log_dir.mkdir()

    _write_timeline(timeline_dir / "SA06_FalsePositive.csv", "SA06", "normal_only")
    _write_model_log(model_log_dir / "SA06_FalsePositive_model_log.csv", has_events=True)
    _write_timeline(timeline_dir / "SA06_Clean.csv", "SA06", "normal_only")
    _write_model_log(model_log_dir / "SA06_Clean_model_log.csv", has_events=False)

    cfg = {"subject_range": "SA06", "max_adl_trials": 1}
    _, stats = select_trials_for_unity(timeline_dir, model_log_dir, cfg)

    assert stats["by_direction"]["normal_only"] == ["SA06_Clean"]
