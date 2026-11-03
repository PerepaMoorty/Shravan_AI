using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace ShravanAI.Playback
{
    /// <summary>
    /// Play/Pause/Scrub controls for the shared clock, plus the trial label.
    /// The lead-time text is owned by LeadTimeReadoutController.
    /// </summary>
    public class DashboardController : MonoBehaviour
    {
        public TrialTimelineLoader trialLoader;
        public SharedClock clock;

        public Button playPauseButton;
        public TMP_Text playPauseLabel;
        public Slider scrubSlider;
        public TMP_Text trialLabelText;

        private void OnEnable()
        {
            if (clock != null) clock.OnTimeChanged += OnTimeChanged;
            if (trialLoader != null) trialLoader.OnTrialLoaded += OnTrialLoaded;
            if (playPauseButton != null) playPauseButton.onClick.AddListener(TogglePlayPause);
            if (scrubSlider != null) scrubSlider.onValueChanged.AddListener(OnSliderChanged);
        }

        private void OnDisable()
        {
            if (clock != null) clock.OnTimeChanged -= OnTimeChanged;
            if (trialLoader != null) trialLoader.OnTrialLoaded -= OnTrialLoaded;
            if (playPauseButton != null) playPauseButton.onClick.RemoveListener(TogglePlayPause);
            if (scrubSlider != null) scrubSlider.onValueChanged.RemoveListener(OnSliderChanged);
        }

        private void TogglePlayPause()
        {
            if (clock.IsPlaying) clock.Pause();
            else clock.Play();
            RefreshPlayPauseLabel();
        }

        private void OnSliderChanged(float normalized)
        {
            clock.ScrubTo(normalized * clock.DurationS);
        }

        private void OnTimeChanged(float timeS)
        {
            RefreshPlayPauseLabel();
            if (scrubSlider != null && clock.DurationS > 0f)
            {
                scrubSlider.SetValueWithoutNotify(timeS / clock.DurationS);
            }
        }

        private void OnTrialLoaded()
        {
            if (trialLabelText != null)
            {
                var meta = trialLoader.Meta;
                trialLabelText.text = $"{meta.subject_id}/{meta.trial_id}  ({meta.fall_direction})";
            }
            RefreshPlayPauseLabel();
        }

        private void RefreshPlayPauseLabel()
        {
            if (playPauseLabel != null) playPauseLabel.text = clock.IsPlaying ? "Pause" : "Play";
        }
    }
}
