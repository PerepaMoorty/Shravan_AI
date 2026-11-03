using TMPro;
using UnityEngine;

namespace ShravanAI.Playback
{
    /// <summary>
    /// Live "warned X.XXs before the confirmed fall" readout, computed from the
    /// current ModelLog each tick: blank until an event fires, "watching" after
    /// a pre-impact warning, then the lead time (or "no advance warning").
    /// </summary>
    public class LeadTimeReadoutController : MonoBehaviour
    {
        public TrialTimelineLoader trialLoader;
        public SharedClock clock;

        public TMP_Text leadLagText;

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
            if (leadLagText == null) return;

            ModelFiringLog log = trialLoader != null ? trialLoader.ModelLog : null;
            if (log == null)
            {
                leadLagText.text = "";
                return;
            }

            float? preImpactTimeS = null;
            float? fallDetectedTimeS = null;
            foreach (var evt in log.EventsUpTo(timeS))
            {
                if (evt.EventType == "pre_impact_predicted" && preImpactTimeS == null) preImpactTimeS = evt.TimestampS;
                else if (evt.EventType == "fall_detected" && fallDetectedTimeS == null) fallDetectedTimeS = evt.TimestampS;
            }

            if (fallDetectedTimeS == null && preImpactTimeS == null)
            {
                leadLagText.text = "";
            }
            else if (fallDetectedTimeS == null)
            {
                leadLagText.text = "Early warning given — watching for confirmation…";
            }
            else if (preImpactTimeS == null)
            {
                leadLagText.text = "Fall detected — no advance warning was given";
            }
            else
            {
                float leadS = fallDetectedTimeS.Value - preImpactTimeS.Value;
                leadLagText.text = leadS >= 0f
                    ? $"Warned {leadS:0.00}s before the confirmed fall"
                    : $"Detected {-leadS:0.00}s before the early warning (out of order)";
            }
        }
    }
}
