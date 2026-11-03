using TMPro;
using UnityEngine;

namespace ShravanAI.Playback
{
    /// <summary>
    /// Live cadence / stride-time / variability readout from
    /// TrialTimelineLoader.GaitFeatures. Descriptive only, not a model input.
    /// </summary>
    public class GaitFeaturePanelController : MonoBehaviour
    {
        public TrialTimelineLoader trialLoader;
        public SharedClock clock;

        public TMP_Text cadenceText;
        public TMP_Text strideTimeText;
        public TMP_Text variabilityText;

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
            GaitFeatureTrack track = trialLoader != null ? trialLoader.GaitFeatures : null;
            if (track == null || track.Frames.Count == 0)
            {
                SetNoData("no gait data exported for this trial");
                return;
            }

            GaitFrame frame = track.FrameAtTime(timeS);
            if (frame.StepCount < 2)
            {
                SetNoData($"insufficient steps detected (steps={frame.StepCount})");
                return;
            }

            if (cadenceText != null) cadenceText.text = $"Cadence: {frame.CadenceSpm:0.0} steps/min";
            if (strideTimeText != null)
                strideTimeText.text = $"Stride time: ~{frame.StrideTimeS:0.00}s ({frame.StepIntervalS:0.00}s/step)";
            if (variabilityText != null)
            {
                variabilityText.text = frame.StepCount >= 3
                    ? $"Stride variability (CV): {frame.StrideTimeCv:0.00}"
                    : "Stride variability: -- (need 3+ steps)";
            }
        }

        private void SetNoData(string reason)
        {
            if (cadenceText != null) cadenceText.text = "Cadence: --";
            if (strideTimeText != null) strideTimeText.text = "Stride time: --";
            if (variabilityText != null) variabilityText.text = $"Stride variability: -- ({reason})";
        }
    }
}
