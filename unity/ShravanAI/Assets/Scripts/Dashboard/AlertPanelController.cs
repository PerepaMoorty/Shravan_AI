using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace ShravanAI.Playback
{
    /// <summary>
    /// Pre-impact and fall-detected badges, lit purely from the model's event
    /// log, with optional confidence text next to each.
    /// </summary>
    public class AlertPanelController : MonoBehaviour
    {
        public TrialTimelineLoader trialLoader;
        public SharedClock clock;

        public Image preImpactBadge;
        public Image fallDetectedBadge;

        [Tooltip("Optional. Shows the firing event's confidence next to each badge.")]
        public TMP_Text preImpactConfidenceText;
        public TMP_Text fallDetectedConfidenceText;

        [Tooltip("When false, the pre-impact badge is shown in disabledColor and ignores " +
                 "the model log.")]
        public bool predictionAvailable = false;

        public Color offColor = new Color(0.3f, 0.3f, 0.3f);
        public Color disabledColor = new Color(0.18f, 0.18f, 0.18f);
        public Color preImpactOnColor = new Color(0.95f, 0.65f, 0.1f); // amber
        public Color fallDetectedOnColor = new Color(0.8f, 0.15f, 0.15f); // red

        private void OnEnable()
        {
            if (clock != null) clock.OnTimeChanged += OnTimeChanged;
        }

        private void OnDisable()
        {
            if (clock != null) clock.OnTimeChanged -= OnTimeChanged;
        }

        private void OnTimeChanged(float timeS)
        {
            ModelFiringLog log = trialLoader.ModelLog;
            if (log == null) return;

            bool preImpactFired = false;
            bool fallDetectedFired = false;
            float preImpactConfidence = 0f;
            float fallDetectedConfidence = 0f;

            foreach (var evt in log.EventsUpTo(timeS))
            {
                if (predictionAvailable && evt.EventType == "pre_impact_predicted")
                {
                    preImpactFired = true;
                    preImpactConfidence = evt.Confidence;
                }
                else if (evt.EventType == "fall_detected")
                {
                    fallDetectedFired = true;
                    fallDetectedConfidence = evt.Confidence;
                }
            }

            if (preImpactBadge != null)
            {
                preImpactBadge.color = !predictionAvailable ? disabledColor : (preImpactFired ? preImpactOnColor : offColor);
            }
            if (fallDetectedBadge != null) fallDetectedBadge.color = fallDetectedFired ? fallDetectedOnColor : offColor;

            if (preImpactConfidenceText != null)
            {
                preImpactConfidenceText.text = (predictionAvailable && preImpactFired)
                    ? $"{preImpactConfidence * 100f:0}%"
                    : "";
            }
            if (fallDetectedConfidenceText != null)
            {
                fallDetectedConfidenceText.text = fallDetectedFired
                    ? $"{fallDetectedConfidence * 100f:0}%"
                    : "";
            }
        }
    }
}
