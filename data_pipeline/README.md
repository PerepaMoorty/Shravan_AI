# ShravanAI data pipeline

Parses KFall sensor + label pairs into per-trial ground-truth timelines, tags
each trial's fall direction, runs a threshold stub detector, and exports a
curated trial set for the Unity digital twin.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

## Data

If `<repo root>/datasets/kfall_dataset` exists it is used; otherwise the
synthetic set in `sample_data/` is. `SHRAVAN_KFALL_ROOT` overrides both. Every
script prints which root it resolved to on its first line.

The root must contain `sensor_data/<subject>/*.csv` and
`label_data/<subject>_label.xlsx` (or `.csv`). Real label sheets only list fall
trials; any unlabelled sensor recording is treated as an ADL (`normal_only`)
trial. See `docs/KFALL_TASK_CODE_MAPPING.md` for how task codes map to
directions.

## Run

```bash
.venv/Scripts/python scripts/run_pipeline.py      # -> output/trial_timelines/*.csv
.venv/Scripts/python scripts/run_model.py         # stub detector -> output/model_logs/*.csv
.venv/Scripts/python scripts/run_sanity_check.py  # -> output/sanity_plots/*.png
.venv/Scripts/python scripts/export_for_unity.py  # -> ../unity/ShravanAI/Assets/StreamingAssets/Trials/
```

`scripts/generate_synthetic_kfall.py` regenerates `sample_data/`.

Against the full KFall dataset: 5,075 trials processed, 0 skipped, 152 flagged
for review (all the non-directional "fainting" fall type).

`export_for_unity.py` writes the stub model's events into `_model_log.csv`.
Afterwards, run `architecture/ShravanAI/scripts/export_detection_to_unity.py`
and `export_prediction_to_unity.py` (in either order) to replace them with
the trained models' output.

## Choosing which subjects Unity sees

`unity_export_config.json` (read by `shravan_ai/curation.py`) controls the
export:

```json
{
  "subject_range": "SA06-SA15",
  "max_trials_per_direction": 3,
  "max_adl_trials": 3
}
```

Edit `subject_range` (or give an explicit `subjects` list) and re-run
`export_for_unity.py`, then the two architecture export scripts. The previous
export is cleared first. Fall trials where the stub fired, and ADL trials
where it stayed silent, are preferred. Delete the config to export every
trial.

## Tests

```bash
.venv/Scripts/python -m pytest tests/
```

## Layout

| Module | Purpose |
|---|---|
| `shravan_ai/config.py` | Data/output paths |
| `shravan_ai/schema.py` | Sensor/label column names, alias-tolerant lookup |
| `shravan_ai/direction.py` | Fall-direction tagging |
| `shravan_ai/parser.py` | Per-trial timeline builder |
| `shravan_ai/pipeline.py` | Batch runner pairing sensor CSVs with label rows |
| `shravan_ai/model.py` | Threshold stub detector |
| `shravan_ai/leadlag.py` | Model lead/lag vs. ground truth |
| `shravan_ai/gait_features.py` | Cadence / stride-time estimates |
| `shravan_ai/curation.py` | Trial selection for Unity |
| `shravan_ai/sanity_check.py` | Label sanity plots |
