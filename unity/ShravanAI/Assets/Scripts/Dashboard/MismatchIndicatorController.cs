using TMPro;
using UnityEngine;

namespace ShravanAI.Playback
{
    /// <summary>
    /// Shows text only when the model got a trial wrong: a missed fall, a false
    /// alarm, or a detected fall with no early warning. Evaluated once per trial
    /// from Meta.has_impact and the full model log.
    /// </summary>
    public class MismatchIndicatorController : MonoBehaviour
    {
        public TrialTimelineLoader trialLoader;

        public TMP_Text mismatchText;

        public Color missedFallColor = new Color(0.85f, 0.2f, 0.2f);
        public Color falseAlarmColor = new Color(0.9f, 0.55f, 0.1f);
        public Color partialColor = new Color(0.9f, 0.8f, 0.2f);

        private void OnEnable()
        {
            if (trialLoader != null) trialLoader.OnTrialLoaded += OnTrialLoaded;
        }

        private void OnDisable()
        {
            if (trialLoader != null) trialLoader.OnTrialLoaded -= OnTrialLoaded;
        }

        private void OnTrialLoaded()
        {
            if (mismatchText == null) return;

            TrialMeta meta = trialLoader.Meta;
            ModelFiringLog log = trialLoader.ModelLog;
            if (meta == null || log == null)
            {
                mismatchText.text = "";
                return;
            }

            bool isRealFall = meta.has_impact;
            bool detectionFiredEver = false;
            bool predictionFiredEver = false;
            foreach (var evt in log.Events)
            {
                if (evt.EventType == "fall_detected") detectionFiredEver = true;
                else if (evt.EventType == "pre_impact_predicted") predictionFiredEver = true;
            }

            if (isRealFall && !detectionFiredEver)
            {
                mismatchText.text = "⚠ MODEL MISSED THIS FALL";
                mismatchText.color = missedFallColor;
            }
            else if (!isRealFall && detectionFiredEver)
            {
                mismatchText.text = "⚠ FALSE ALARM — nothing actually happened";
                mismatchText.color = falseAlarmColor;
            }
            else if (isRealFall && detectionFiredEver && !predictionFiredEver)
            {
                mismatchText.text = "⚠ Fall detected, but no early warning was given";
                mismatchText.color = partialColor;
            }
            else
            {
                mismatchText.text = "";
            }
        }
    }
}
