using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace ShravanAI.Playback
{
    /// <summary>
    /// Live fill-bar gauge of a model's raw confidence trace (detection
    /// proba_fall or prediction proba_pre_impact). Holds the latest real value
    /// between windows. Add one instance per stream.
    /// </summary>
    public class ConfidenceGaugeController : MonoBehaviour
    {
        public enum Source
        {
            DetectionProbaFall,
            PredictionProbaPreImpact,
        }

        public TrialTimelineLoader trialLoader;
        public SharedClock clock;

        public Source source = Source.PredictionProbaPreImpact;

        [Tooltip("Image with Image Type = Filled (Horizontal/Radial) -- fillAmount driven 0..1.")]
        public Image fillImage;
        public TMP_Text valueText;

        [Tooltip("Visual only: 0.5 for detection, 0.80 for prediction.")]
        public float decisionThreshold = 0.80f;

        public Color belowThresholdColor = new Color(0.4f, 0.55f, 0.7f);
        public Color aboveThresholdColor = new Color(0.9f, 0.3f, 0.2f);

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
            if (trialLoader == null) return;

            ConfidenceTrace trace = source == Source.DetectionProbaFall
                ? trialLoader.DetectionConfidence
                : trialLoader.PredictionConfidence;

            if (trace == null || trace.Frames.Count == 0)
            {
                if (valueText != null) valueText.text = "n/a";
                return;
            }

            ConfidenceFrame frame = trace.FrameAtTime(timeS);
            float value = source == Source.DetectionProbaFall ? frame.ProbaFall : frame.ProbaPreImpact;

            if (fillImage != null)
            {
                fillImage.fillAmount = Mathf.Clamp01(value);
                fillImage.color = value >= decisionThreshold ? aboveThresholdColor : belowThresholdColor;
            }
            if (valueText != null) valueText.text = $"{value * 100f:0}%";
        }
    }
}
