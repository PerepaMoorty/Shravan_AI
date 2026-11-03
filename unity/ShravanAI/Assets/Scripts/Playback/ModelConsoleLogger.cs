using System.IO;
using UnityEngine;

namespace ShravanAI.Playback
{
    /// <summary>
    /// Logs model events, ground-truth phase changes, a periodic heartbeat
    /// (with gait readout) and the models' test-set stats to the Unity Console,
    /// so the model's output can be audited without the dashboard.
    /// </summary>
    public class ModelConsoleLogger : MonoBehaviour
    {
        public TrialTimelineLoader trialLoader;
        public SharedClock clock;

        [Tooltip("Optional. Its predictionAvailable flag decides whether the prediction stream " +
                 "is reported as active.")]
        public AlertPanelController alertPanel;

        [Header("Verbosity")]
        public bool logGroundTruthPhaseChanges = true;
        public bool logPredictionStatus = true;
        public bool logModelStats = true;
        public bool logHeartbeat = true;

        [Tooltip("Heartbeat interval in trial (dataset) seconds.")]
        public float heartbeatIntervalS = 0.5f;

        private int _nextEventIndex;
        private string _lastLoggedPhase;
        private float _lastHeartbeatS;
        private ModelStats _modelStats;

        private bool PredictionActive => alertPanel != null && alertPanel.predictionAvailable;

        private void Awake()
        {
            string path = Path.Combine(Application.streamingAssetsPath, "Trials", "_model_stats.json");
            _modelStats = ModelStats.LoadFromJson(path);
        }

        private void Start()
        {
            if (logModelStats) LogModelStats();
        }

        private void OnEnable()
        {
            if (clock != null) clock.OnTimeChanged += OnTimeChanged;
            if (trialLoader != null) trialLoader.OnTrialLoaded += OnTrialLoaded;
        }

        private void OnDisable()
        {
            if (clock != null) clock.OnTimeChanged -= OnTimeChanged;
            if (trialLoader != null) trialLoader.OnTrialLoaded -= OnTrialLoaded;
        }

        private void OnTrialLoaded()
        {
            _nextEventIndex = 0;
            _lastLoggedPhase = null;
            _lastHeartbeatS = 0f;
            LogTrialBanner();
        }

        private void OnTimeChanged(float timeS)
        {
            if (logGroundTruthPhaseChanges) CheckGroundTruthPhase(timeS);
            if (logHeartbeat) CheckHeartbeat(timeS);

            ModelFiringLog log = trialLoader.ModelLog;
            if (log == null) return;

            // Rewind the cursor when scrubbing backward.
            while (_nextEventIndex > 0 && log.Events[_nextEventIndex - 1].TimestampS > timeS)
                _nextEventIndex--;

            while (_nextEventIndex < log.Events.Count && log.Events[_nextEventIndex].TimestampS <= timeS)
            {
                LogEvent(log.Events[_nextEventIndex]);
                _nextEventIndex++;
            }
        }

        private void LogEvent(ModelEvent evt)
        {
            TrialMeta meta = trialLoader.Meta;
            string label = evt.EventType == "fall_detected" ? "FALL DETECTED" : "PRE-IMPACT PREDICTED";

            Debug.Log(
                $"[ShravanAI model] {meta.subject_id}/{meta.trial_id} ({meta.fall_direction})  " +
                $"t={evt.TimestampS * 1000f:0.0}ms  frame={evt.FrameIdx}  " +
                $"{label}  confidence={evt.Confidence:0.00}"
            );
        }

        private void LogModelStats()
        {
            if (_modelStats == null)
            {
                Debug.Log(
                    "[ShravanAI model] Detection model performance stats: _model_stats.json not found -- " +
                    "run architecture/ShravanAI/scripts/export_detection_to_unity.py to generate it."
                );
                return;
            }

            Debug.Log(
                "[ShravanAI model] ==== Detection model performance " +
                "(held-out test-set numbers -- describe the model in general, not this specific trial) ==== " +
                $"trained on {string.Join("+", _modelStats.datasets)}, " +
                $"{_modelStats.train_subjects}/{_modelStats.val_subjects}/{_modelStats.test_subjects} " +
                "train/val/test subjects"
            );
            Debug.Log(
                $"[ShravanAI model] accuracy={_modelStats.accuracy:0.000}  precision={_modelStats.precision:0.000}  " +
                $"recall(sensitivity)={_modelStats.recall:0.000}  specificity={_modelStats.specificity:0.000}  " +
                $"false_positive_rate={_modelStats.false_positive_rate:0.000}  f1={_modelStats.f1:0.000}  " +
                $"roc_auc={_modelStats.roc_auc:0.000}"
            );
            ConfusionMatrixCounts cm = _modelStats.confusion_matrix;
            Debug.Log(
                $"[ShravanAI model] confusion matrix (n={_modelStats.n_test_samples}, " +
                $"positives={_modelStats.n_test_positive}): TN={cm.tn} FP={cm.fp} FN={cm.fn} TP={cm.tp}  " +
                $"decision_threshold={_modelStats.decision_threshold:0.00}  " +
                "(per-event confidence below is this model's raw fall-probability for that one window; " +
                "an event only fires once confidence crosses decision_threshold)"
            );

            LogPredictionStats();
        }

        // prediction.available comes from the exported JSON, independent of
        // AlertPanelController.predictionAvailable (the UI switch).
        private void LogPredictionStats()
        {
            PredictionStats pred = _modelStats.prediction;
            if (pred == null || !pred.available)
            {
                Debug.Log(
                    "[ShravanAI model] Prediction model stats: not available -- run " +
                    "architecture/ShravanAI/scripts/export_prediction_to_unity.py."
                );
                return;
            }

            Debug.Log(
                $"[ShravanAI model] ==== Prediction model performance (architecture={pred.architecture}, " +
                "held-out test-set numbers) ==== " +
                $"test_macro_f1={pred.test_macro_f1:0.000}"
            );

            PredictionPerClass pc = pred.per_class;
            if (pc != null)
            {
                Debug.Log(
                    $"[ShravanAI model] per-class -- non_fall: P={pc.non_fall.precision:0.00} R={pc.non_fall.recall:0.00} F1={pc.non_fall.f1:0.00}  " +
                    $"pre_impact: P={pc.pre_impact.precision:0.00} R={pc.pre_impact.recall:0.00} F1={pc.pre_impact.f1:0.00}  " +
                    $"fall: P={pc.fall.precision:0.00} R={pc.fall.recall:0.00} F1={pc.fall.f1:0.00}"
                );
            }

            PredictionLeadTime lt = pred.lead_time;
            if (lt != null && lt.n_trials > 0)
            {
                string leadTimeText = lt.n_flagged > 0
                    ? $"mean={lt.mean_lead_time_ms:0}ms median={lt.median_lead_time_ms:0}ms"
                    : "no trial ever flagged in time -- mean/median not meaningful";
                Debug.Log(
                    $"[ShravanAI model] lead time: " +
                    $"detection_rate={lt.detection_rate:0.00} ({lt.n_flagged}/{lt.n_trials} falls flagged in time)  {leadTimeText}"
                );
            }
        }

        private void LogTrialBanner()
        {
            TrialMeta meta = trialLoader.Meta;
            Debug.Log(
                $"[ShravanAI model] ==== Trial loaded: {meta.subject_id}/{meta.trial_id} " +
                $"({meta.fall_direction}), duration={meta.duration_s:0.00}s ===="
            );

            if (logPredictionStatus)
            {
                string status = PredictionActive
                    ? "ACTIVE"
                    : "DISABLED (AlertPanelController.predictionAvailable is unchecked)";
                Debug.Log($"[ShravanAI model] Prediction stream: {status}");
            }
        }

        // Edge-triggered on the ground-truth value at timeS, so scrubbing works.
        private void CheckGroundTruthPhase(float timeS)
        {
            if (trialLoader.Timeline == null || trialLoader.Timeline.Frames.Count == 0) return;

            TrialFrame frame = trialLoader.Timeline.FrameAtTime(timeS);
            if (frame.Phase == _lastLoggedPhase) return;
            _lastLoggedPhase = frame.Phase;

            TrialMeta meta = trialLoader.Meta;
            Debug.Log(
                $"[ShravanAI ground-truth] {meta.subject_id}/{meta.trial_id}  " +
                $"t={timeS * 1000f:0.0}ms  frame={frame.FrameIdx}  phase -> {frame.Phase}"
            );
        }

        private void CheckHeartbeat(float timeS)
        {
            // Scrubbing backward re-arms the heartbeat immediately.
            if (timeS - _lastHeartbeatS < heartbeatIntervalS && timeS >= _lastHeartbeatS) return;
            _lastHeartbeatS = timeS;

            TrialMeta meta = trialLoader.Meta;
            string phase = trialLoader.Timeline != null && trialLoader.Timeline.Frames.Count > 0
                ? trialLoader.Timeline.FrameAtTime(timeS).Phase
                : "unknown";
            string prediction = PredictionActive ? "active" : "disabled";
            string gait = FormatGait(timeS);

            Debug.Log(
                $"[ShravanAI model] heartbeat  {meta.subject_id}/{meta.trial_id}  " +
                $"t={timeS * 1000f:0.0}ms  ground_truth_phase={phase}  prediction={prediction}  {gait}  " +
                "(watching, no event fired since last log)"
            );
        }

        // Needs 2+ steps for cadence and 3+ for variability.
        private string FormatGait(float timeS)
        {
            GaitFeatureTrack track = trialLoader.GaitFeatures;
            if (track == null || track.Frames.Count == 0) return "gait=n/a";

            GaitFrame frame = track.FrameAtTime(timeS);
            if (frame.StepCount < 2) return $"gait=insufficient-data(steps={frame.StepCount})";

            string cv = frame.StrideTimeCv > 0f ? frame.StrideTimeCv.ToString("0.00") : "n/a (<3 steps)";
            return $"gait(cadence={frame.CadenceSpm:0.0}spm stride~{frame.StrideTimeS:0.00}s cv={cv} steps={frame.StepCount})";
        }
    }
}
