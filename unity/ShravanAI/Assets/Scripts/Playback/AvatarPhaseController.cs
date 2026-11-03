using UnityEngine;

namespace ShravanAI.Playback
{
    /// <summary>
    /// Drives the ghost avatar's Animator from ground truth. Edge-triggered so
    /// the Animator's crossfades aren't retriggered every tick.
    /// </summary>
    [RequireComponent(typeof(Animator))]
    public class AvatarPhaseController : MonoBehaviour
    {
        public TrialTimelineLoader trialLoader;
        public SharedClock clock;

        private Animator _animator;
        private string _lastPhase;
        private string _lastDirection;

        private void Awake()
        {
            _animator = GetComponent<Animator>();
        }

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
            if (trialLoader.Timeline == null || trialLoader.Timeline.Frames.Count == 0) return;

            TrialFrame frame = trialLoader.Timeline.FrameAtTime(timeS);
            if (frame.Phase == _lastPhase && frame.FallDirection == _lastDirection) return;

            _animator.SetInteger(AnimatorParams.Phase, (int)AnimatorParams.ParsePhase(frame.Phase));
            _animator.SetInteger(AnimatorParams.FallDirection, (int)AnimatorParams.ParseFallDirection(frame.FallDirection));

            _lastPhase = frame.Phase;
            _lastDirection = frame.FallDirection;
        }
    }
}
