# ShravanAI — Digital Twin for Fall Prediction & Detection

A Unity digital twin replays real KFall trials with two overlapping avatars:
one driven by trained fall-prediction/detection models, one by ground truth.
They run on a shared, slowed-down clock, so you can see whether the model
caught the fall and how early.

## Layout

| Folder | Contents |
|---|---|
| `data_pipeline/` | KFall parsing, direction tagging, stub detector, Unity export ([README](data_pipeline/README.md)) |
| `architecture/ShravanAI/` | Detection (XGBoost) and prediction (ConvLSTM / TinyTransformer) models ([README](architecture/ShravanAI/README.md)) |
| `unity/ShravanAI/` | Unity scene and C# playback/dashboard ([README](unity/ShravanAI/README.md)) |
| `docs/` | Statistical evaluation write-up and KFall task-code mapping |

The KFall and SisFall datasets go in `datasets/` (gitignored, not redistributable).

## Results

| Model | Held-out test set | Curated demo trials |
|---|---|---|
| Detection | accuracy 0.895, ROC-AUC 0.961 | 9/9 falls detected, 0/3 false alarms |
| Prediction | accuracy 0.91 | 9/9 falls warned, ~1.4s mean lead time, 0/3 false alarms |

Full analysis: [docs/COMPREHENSIVE_STATISTICAL_EVALUATION.md](docs/COMPREHENSIVE_STATISTICAL_EVALUATION.md).

## How to run

Each Python project has its own `.venv` (see its README). Run in order:

1. **Ground truth + curated trials**, from `data_pipeline/`:
   `run_pipeline.py` → `run_model.py` → `export_for_unity.py`
2. **Train models** (only when retraining), from `architecture/ShravanAI/`:
   `train_detection_model.py`, `train_prediction_model.py --architecture both --pre-impact-boost 1.0`
3. **Real model output into Unity**, from `architecture/ShravanAI/`:
   `export_detection_to_unity.py` and `export_prediction_to_unity.py` (either order).
   Re-run these whenever step 1 is re-run, since step 1 overwrites the model logs with the stub's.
4. **Play** `unity/ShravanAI/Assets/Scenes/SampleScene.unity` in Unity 6000.4.7f1.

Tests: `python -m pytest tests/` inside each Python project.
