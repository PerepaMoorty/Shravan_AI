# ShravanAI — Unity Digital Twin

Two avatars replay one KFall trial on a shared, speed-adjustable clock:

- **ModelAvatar** (solid) is driven only by the models' event log (`ModelPhaseController`).
- **GhostAvatar** (translucent) is driven only by ground truth (`AvatarPhaseController`).

The dashboard (alert badges, timing strip, lead-time readout, mismatch label,
event history, gait readout) reads the same clock. Open the project with
**Unity 6000.4.7f1** and play `Assets/Scenes/SampleScene.unity`.

## Trial data

Everything the scene reads lives in `Assets/StreamingAssets/Trials/`:

| File | Written by | Read by |
|---|---|---|
| `_index.json` | `data_pipeline/scripts/export_for_unity.py` | `AutoPlaylistController` |
| `<trial>_timeline.csv`, `_meta.json`, `_gait_features.csv` | `data_pipeline/scripts/export_for_unity.py` | `TrialTimelineLoader` |
| `<trial>_model_log.csv` | stub from `export_for_unity.py`, then overwritten by `architecture/ShravanAI/scripts/export_{detection,prediction}_to_unity.py` | alerts, timing strip, model avatar, Console |
| `<trial>_{detection,prediction}_confidence_trace.csv` | `export_{detection,prediction}_to_unity.py` | `ConfidenceGaugeController` |
| `_model_stats.json` | `export_{detection,prediction}_to_unity.py` | `ModelConsoleLogger` |

Which subjects get exported is set in `data_pipeline/unity_export_config.json`.

## Scene layout

- **DashboardRoot**: `SharedClock` (`Time Scale` 0.01 = 100x slower than real
  time), `TrialTimelineLoader` (`trialKey` = an entry from `_index.json`) and the
  dashboard controllers.
- **ModelAvatar / GhostAvatar**: same position, same `ShravanAI.controller`.
  The ghost uses the `Avatar/*_ghost.mat` materials.
- **AutoPlaylist**: `AutoPlaylistController` cycles every trial in
  `_index.json`, holding the last frame for `Inter Trial Pause S`.
- **ModelConsoleLogger**: mirrors events, ground-truth phase changes, a
  heartbeat and the models' test-set stats to the Console.

## Animator Controller

`Assets/Animations/ShravanAI.controller` parameters must match
`Scripts/Playback/AnimatorParams.cs`:

| Parameter | Type | Values |
|---|---|---|
| `Phase` | Int | 0 = Normal, 1 = PreImpact, 2 = PostImpact |
| `FallDirection` | Int | 0 = None, 1 = Forward, 2 = Backward, 3 = Lateral (unused by transitions) |

States `Normal` (Walking), `Falling` (Tripping, not looped) and `Fallen`
(Fallen Idle) are entered via **Any State** transitions on `Phase == 0/1/2`,
with Has Exit Time and Can Transition To Self off and a 0.15–0.3s crossfade.
Any State is required because scrubbing can jump straight from 0 to 2.

Set `ModelPhaseController.Min Stumble Duration Real Seconds` to the Falling
clip's length so the stumble isn't cut off.

## Not yet wired: confidence gauge

`ConfidenceGaugeController` is written but not in the scene. To add it:

1. Add a UI `Image` with Image Type = Filled, plus an optional `TMP_Text`.
2. Add `ConfidenceGaugeController`, wire `Trial Loader`, `Clock`, `Fill Image`
   and `Value Text`.
3. Set `Source` to `PredictionProbaPreImpact` (threshold 0.80) or
   `DetectionProbaFall` (threshold 0.5). Add one instance per stream.

## Checking it works

Press Play and scrub through a fall trial:

- The ghost switches Normal → Falling → Fallen at the labelled onset/impact;
  the model avatar follows the model's events and may diverge.
- Badges and timing-strip markers come only from `_model_log.csv`.
- The mismatch label stays blank on a correctly handled trial.
- The Console prints one line per event, re-logging events after a backward scrub.

## Known limitations

- Clips play at natural speed, so the timing strip, not the avatar, is the
  timing reference.
- KFall subjects are young healthy adults, a proxy for elderly gait.
- Animation clips are third-party Mixamo assets, illustrative only.
- Gait figures are single-waist-IMU estimates. The curated sit-down falls
  (F01–F03) have only a few pre-onset steps, so their readings are noisy.
