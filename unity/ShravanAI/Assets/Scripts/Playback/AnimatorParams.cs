namespace ShravanAI.Playback
{
    /// <summary>
    /// Animator Controller parameter names and values. These must match the
    /// parameters defined in Assets/Animations/ShravanAI.controller.
    /// </summary>
    public static class AnimatorParams
    {
        public const string Phase = "Phase";
        public const string FallDirection = "FallDirection";

        public enum PhaseValue
        {
            Normal = 0,
            PreImpact = 1,
            PostImpact = 2,
        }

        public enum FallDirectionValue
        {
            None = 0,
            Forward = 1,
            Backward = 2,
            Lateral = 3,
        }

        public static FallDirectionValue ParseFallDirection(string label)
        {
            switch (label)
            {
                case "fall_forward": return FallDirectionValue.Forward;
                case "fall_backward": return FallDirectionValue.Backward;
                case "fall_lateral": return FallDirectionValue.Lateral;
                default: return FallDirectionValue.None; // normal_only / unknown_direction
            }
        }

        public static PhaseValue ParsePhase(string label)
        {
            switch (label)
            {
                case "PreImpact": return PhaseValue.PreImpact;
                case "PostImpact": return PhaseValue.PostImpact;
                default: return PhaseValue.Normal;
            }
        }
    }
}
