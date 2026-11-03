using UnityEngine;
using UnityEngine.UI;

namespace ShravanAI.Playback
{
    /// <summary>
    /// Timing strip: a ground-truth row (onset/impact), a model row (event
    /// markers) and a playhead that follows the shared clock.
    /// </summary>
    public class TimingStripController : MonoBehaviour
    {
        public TrialTimelineLoader trialLoader;
        public SharedClock clock;

        [Header("Strip geometry (assign in Editor)")]
        public RectTransform stripWidthReference; // any RectTransform whose width = full strip length
        public RectTransform groundTruthRow;
        public RectTransform modelRow;
        public RectTransform playhead;

        [Header("Marker prefabs (simple Image, anchored left-middle)")]
        public RectTransform onsetMarkerPrefab;
        public RectTransform impactMarkerPrefab;
        public RectTransform preImpactPredictionMarkerPrefab;
        public RectTransform fallDetectedMarkerPrefab;

        private void OnEnable()
        {
            if (clock != null) clock.OnTimeChanged += OnTimeChanged;
            if (trialLoader != null) trialLoader.OnTrialLoaded += BuildMarkers;
        }

        private void OnDisable()
        {
            if (clock != null) clock.OnTimeChanged -= OnTimeChanged;
            if (trialLoader != null) trialLoader.OnTrialLoaded -= BuildMarkers;
        }

        private float StripWidth => stripWidthReference != null ? stripWidthReference.rect.width : 0f;

        private void PlaceAt(RectTransform marker, RectTransform row, float fraction)
        {
            var instance = Instantiate(marker, row);
            var pos = instance.anchoredPosition;
            pos.x = Mathf.Clamp01(fraction) * StripWidth;
            instance.anchoredPosition = pos;
        }

        public void BuildMarkers()
        {
            foreach (Transform child in groundTruthRow) Destroy(child.gameObject);
            foreach (Transform child in modelRow) Destroy(child.gameObject);

            TrialMeta meta = trialLoader.Meta;
            float duration = trialLoader.Timeline.DurationS;
            if (duration <= 0f) return;

            if (meta.has_onset && onsetMarkerPrefab != null)
                PlaceAt(onsetMarkerPrefab, groundTruthRow, meta.onset_timestamp_s / duration);

            if (meta.has_impact && impactMarkerPrefab != null)
                PlaceAt(impactMarkerPrefab, groundTruthRow, meta.impact_timestamp_s / duration);

            foreach (var evt in trialLoader.ModelLog.Events)
            {
                RectTransform prefab = evt.EventType == "fall_detected"
                    ? fallDetectedMarkerPrefab
                    : preImpactPredictionMarkerPrefab;
                if (prefab != null) PlaceAt(prefab, modelRow, evt.TimestampS / duration);
            }
        }

        private void OnTimeChanged(float timeS)
        {
            if (playhead == null) return;
            float duration = trialLoader.Timeline != null ? trialLoader.Timeline.DurationS : 0f;
            if (duration <= 0f) return;

            var pos = playhead.anchoredPosition;
            pos.x = Mathf.Clamp01(timeS / duration) * StripWidth;
            playhead.anchoredPosition = pos;
        }
    }
}
