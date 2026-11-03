using System;
using System.IO;
using UnityEngine;

namespace ShravanAI.Playback
{
    /// <summary>
    /// Loads one trial's exported files from StreamingAssets/Trials/ and sets
    /// the shared clock's duration. Every other component reads from this.
    /// </summary>
    public class TrialTimelineLoader : MonoBehaviour
    {
        [Tooltip("Matches a key in Assets/StreamingAssets/Trials/_index.json, e.g. SA06_S06T20R01 (real forward fall)")]
        public string trialKey = "SA06_S06T20R01";

        public SharedClock clock;

        public TrialTimeline Timeline { get; private set; }
        public ModelFiringLog ModelLog { get; private set; }
        public TrialMeta Meta { get; private set; }
        public GaitFeatureTrack GaitFeatures { get; private set; }
        public ConfidenceTrace DetectionConfidence { get; private set; }
        public ConfidenceTrace PredictionConfidence { get; private set; }

        public event Action OnTrialLoaded;

        private void Awake()
        {
            if (clock == null) clock = GetComponent<SharedClock>();
        }

        private void Start()
        {
            LoadTrial(trialKey);
        }

        public void LoadTrial(string key)
        {
            string root = Path.Combine(Application.streamingAssetsPath, "Trials");
            Timeline = TrialTimeline.LoadFromCsv(Path.Combine(root, $"{key}_timeline.csv"));
            ModelLog = ModelFiringLog.LoadFromCsv(Path.Combine(root, $"{key}_model_log.csv"));
            Meta = TrialMeta.LoadFromJson(Path.Combine(root, $"{key}_meta.json"));
            GaitFeatures = GaitFeatureTrack.LoadFromCsv(Path.Combine(root, $"{key}_gait_features.csv"));
            DetectionConfidence = ConfidenceTrace.LoadDetectionCsv(Path.Combine(root, $"{key}_detection_confidence_trace.csv"));
            PredictionConfidence = ConfidenceTrace.LoadPredictionCsv(Path.Combine(root, $"{key}_prediction_confidence_trace.csv"));

            clock.DurationS = Timeline.DurationS;
            clock.Reset();

            OnTrialLoaded?.Invoke();
        }
    }
}
