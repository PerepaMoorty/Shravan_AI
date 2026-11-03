using TMPro;
using UnityEngine;

namespace ShravanAI.Playback
{
    /// <summary>
    /// Running list of model events fired so far this trial. Rebuilt only when
    /// the visible event count changes.
    /// </summary>
    public class EventHistoryPanelController : MonoBehaviour
    {
        public TrialTimelineLoader trialLoader;
        public SharedClock clock;

        [Header("Row prefab (a single TMP_Text, anchored top-left) + its container")]
        public TMP_Text rowPrefab;
        public Transform rowContainer;

        [Tooltip("Shown in the container when no event has fired yet this trial.")]
        public string emptyStateText = "No events yet — watching…";

        private int _lastShownCount = -1;

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
            _lastShownCount = -1; // force a rebuild
        }

        private void OnTimeChanged(float timeS)
        {
            if (rowPrefab == null || rowContainer == null) return;

            ModelFiringLog log = trialLoader != null ? trialLoader.ModelLog : null;
            if (log == null) return;

            int visibleCount = 0;
            foreach (var _ in log.EventsUpTo(timeS)) visibleCount++;

            if (visibleCount == _lastShownCount) return;
            _lastShownCount = visibleCount;

            Rebuild(log, timeS, visibleCount);
        }

        private void Rebuild(ModelFiringLog log, float timeS, int visibleCount)
        {
            foreach (Transform child in rowContainer) Destroy(child.gameObject);

            if (visibleCount == 0)
            {
                TMP_Text emptyRow = Instantiate(rowPrefab, rowContainer);
                emptyRow.text = emptyStateText;
                return;
            }

            int shown = 0;
            foreach (var evt in log.EventsUpTo(timeS))
            {
                if (shown >= visibleCount) break;
                string label = evt.EventType == "fall_detected" ? "FALL DETECTED" : "PRE-IMPACT PREDICTED";
                TMP_Text row = Instantiate(rowPrefab, rowContainer);
                row.text = $"t={evt.TimestampS:0.00}s — {label} ({evt.Confidence * 100f:0}% confidence)";
                shown++;
            }
        }
    }
}
