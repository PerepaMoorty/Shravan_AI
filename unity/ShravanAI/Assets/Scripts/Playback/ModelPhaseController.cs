using UnityEngine;

namespace ShravanAI.Playback
{
    /// <summary>
    /// Drives the solid (model) avatar's Animator from the model's event log,
    /// never from ground truth: pre_impact_predicted starts Falling and
    /// fall_detected confirms Fallen. The two models are independent, so
    /// detection can fire first; the avatar then routes through Falling anyway.
    ///
    /// `minStumbleDurationRealSeconds` keeps Falling on screen for a minimum
    /// amount of real time before Fallen, because Animator clips play in real
    /// time while SharedClock.timeScale slows only the trial data. This only
    /// paces the avatar's pose; every other component reports raw event times.
    ///
    /// FallDirection stays None: neither model predicts direction.
    /// </summary>
    [RequireComponent(typeof(Animator))]
    public class ModelPhaseController : MonoBehaviour
    {
        public TrialTimelineLoader trialLoader;
        public SharedClock clock;

        [Tooltip("Minimum real-world seconds in the Falling animation before Fallen may show. " +
                 "Match the Falling clip's length.")]
        public float minStumbleDurationRealSeconds = 1.2f;

        private Animator _animator;
        private AnimatorParams.PhaseValue _lastPhase = AnimatorParams.PhaseValue.Normal;
        private float _fallingStartRealTime = -1f;

        public bool PreImpactFired { get; private set; }
        public bool FallDetectedFired { get; private set; }

        private void Awake()
        {
            _animator = GetComponent<Animator>();
        }

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
            // Clear the dwell timer so the previous trial can't satisfy it.
            _lastPhase = AnimatorParams.PhaseValue.Normal;
            _fallingStartRealTime = -1f;
        }

        private void OnTimeChanged(float timeS)
        {
            ModelFiringLog log = trialLoader != null ? trialLoader.ModelLog : null;
            if (log == null) return;

            // Full rescan so scrubbing backward un-fires events.
            bool preImpactFired = false;
            bool fallDetectedFired = false;
            foreach (var evt in log.EventsUpTo(timeS))
            {
                if (evt.EventType == "pre_impact_predicted") preImpactFired = true;
                else if (evt.EventType == "fall_detected") fallDetectedFired = true;
            }
            PreImpactFired = preImpactFired;
            FallDetectedFired = fallDetectedFired;

            AnimatorParams.PhaseValue desiredPhase = AnimatorParams.PhaseValue.Normal;
            if (fallDetectedFired) desiredPhase = AnimatorParams.PhaseValue.PostImpact;
            else if (preImpactFired) desiredPhase = AnimatorParams.PhaseValue.PreImpact;

            // While scrubbing, snap straight to the correct state.
            bool isScrubbing = clock != null && !clock.IsPlaying;

            AnimatorParams.PhaseValue phase = desiredPhase;
            if (!isScrubbing && desiredPhase == AnimatorParams.PhaseValue.PostImpact
                && _lastPhase != AnimatorParams.PhaseValue.PostImpact)
            {
                if (_lastPhase != AnimatorParams.PhaseValue.PreImpact)
                {
                    // Detection fired before any prediction: play Falling first.
                    phase = AnimatorParams.PhaseValue.PreImpact;
                }
                else
                {
                    float elapsed = Time.time - _fallingStartRealTime;
                    if (elapsed < minStumbleDurationRealSeconds) phase = AnimatorParams.PhaseValue.PreImpact;
                }
            }

            if (phase == _lastPhase) return;

            if (phase == AnimatorParams.PhaseValue.PreImpact && _lastPhase != AnimatorParams.PhaseValue.PreImpact)
            {
                _fallingStartRealTime = Time.time;
            }

            // All Phase transitions are Any State transitions, so any jump is safe.
            _animator.SetInteger(AnimatorParams.Phase, (int)phase);
            _animator.SetInteger(AnimatorParams.FallDirection, (int)AnimatorParams.FallDirectionValue.None);

            _lastPhase = phase;
        }
    }
}
