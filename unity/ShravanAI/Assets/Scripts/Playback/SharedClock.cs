using System;
using UnityEngine;

namespace ShravanAI.Playback
{
    /// <summary>
    /// The single playback clock that both avatars and the dashboard read, so
    /// ground truth and model output can never drift apart.
    /// </summary>
    public class SharedClock : MonoBehaviour
    {
        [Tooltip("Dataset-seconds of trial time that elapse per one real-world second. " +
                 "Default 0.01 makes 10ms of recorded sensor time play out over 1 real " +
                 "second. Set to 1.0 for real time. Editable during Play mode.")]
        public float timeScale = 0.01f;

        public float DurationS { get; set; }
        public float CurrentTimeS { get; private set; }
        public bool IsPlaying { get; private set; }

        public event Action<float> OnTimeChanged;

        /// <summary>Fires once when playback runs to the end on its own (not on
        /// a manual scrub), so AutoPlaylistController can advance.</summary>
        public event Action OnPlaybackFinished;

        public void Play() => IsPlaying = true;
        public void Pause() => IsPlaying = false;

        public void ScrubTo(float timeS)
        {
            CurrentTimeS = Mathf.Clamp(timeS, 0f, DurationS);
            OnTimeChanged?.Invoke(CurrentTimeS);
        }

        public void Reset()
        {
            CurrentTimeS = 0f;
            IsPlaying = false;
            OnTimeChanged?.Invoke(CurrentTimeS);
        }

        private void Update()
        {
            if (!IsPlaying) return;

            CurrentTimeS += Time.deltaTime * Mathf.Max(0f, timeScale);
            bool justFinished = CurrentTimeS >= DurationS;
            if (justFinished)
            {
                CurrentTimeS = DurationS;
                IsPlaying = false;
            }
            OnTimeChanged?.Invoke(CurrentTimeS);
            if (justFinished) OnPlaybackFinished?.Invoke();
        }
    }
}
