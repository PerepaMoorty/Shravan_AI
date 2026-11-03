using System.Collections.Generic;
using System.Globalization;

namespace ShravanAI.Playback
{
    public struct TrialFrame
    {
        public int FrameIdx;
        public float TimestampS;
        public string Phase;         // Normal / PreImpact / PostImpact
        public string FallDirection; // fall_forward / fall_backward / fall_lateral / normal_only
    }

    /// <summary>
    /// Ground-truth timeline for one trial, loaded from "&lt;trial&gt;_timeline.csv".
    /// The ghost avatar's animation state reads only this.
    /// </summary>
    public class TrialTimeline
    {
        public List<TrialFrame> Frames = new List<TrialFrame>();

        public float DurationS => Frames.Count > 0 ? Frames[Frames.Count - 1].TimestampS : 0f;

        public static TrialTimeline LoadFromCsv(string path)
        {
            var timeline = new TrialTimeline();
            foreach (var row in MiniCsv.ReadRows(path))
            {
                timeline.Frames.Add(new TrialFrame
                {
                    FrameIdx = int.Parse(row["frame_idx"], CultureInfo.InvariantCulture),
                    TimestampS = float.Parse(row["timestamp_s"], CultureInfo.InvariantCulture),
                    Phase = row["phase"],
                    FallDirection = row["fall_direction"],
                });
            }
            return timeline;
        }

        /// <summary>Frame active at a given playback time (linear scan).</summary>
        public TrialFrame FrameAtTime(float timeS)
        {
            TrialFrame result = Frames.Count > 0 ? Frames[0] : default;
            foreach (var frame in Frames)
            {
                if (frame.TimestampS > timeS) break;
                result = frame;
            }
            return result;
        }
    }
}
