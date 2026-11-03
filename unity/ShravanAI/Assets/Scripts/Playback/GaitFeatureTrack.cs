using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace ShravanAI.Playback
{
    public struct GaitFrame
    {
        public int FrameIdx;
        public float TimestampS;
        public int StepCount;
        public float CadenceSpm;
        public float StepIntervalS;
        public float StrideTimeS;
        public float StrideTimeCv;
    }

    /// <summary>
    /// Per-frame gait features from "&lt;trial&gt;_gait_features.csv" (see
    /// data_pipeline/shravan_ai/gait_features.py). Descriptive only; values freeze
    /// once the ground-truth Normal phase ends.
    /// </summary>
    public class GaitFeatureTrack
    {
        public List<GaitFrame> Frames = new List<GaitFrame>();

        public static GaitFeatureTrack LoadFromCsv(string path)
        {
            var track = new GaitFeatureTrack();
            if (!File.Exists(path)) return track;

            foreach (var row in MiniCsv.ReadRows(path))
            {
                track.Frames.Add(new GaitFrame
                {
                    FrameIdx = int.Parse(row["frame_idx"], CultureInfo.InvariantCulture),
                    TimestampS = float.Parse(row["timestamp_s"], CultureInfo.InvariantCulture),
                    StepCount = int.Parse(row["step_count"], CultureInfo.InvariantCulture),
                    CadenceSpm = float.Parse(row["cadence_spm"], CultureInfo.InvariantCulture),
                    StepIntervalS = float.Parse(row["step_interval_s"], CultureInfo.InvariantCulture),
                    StrideTimeS = float.Parse(row["stride_time_s"], CultureInfo.InvariantCulture),
                    StrideTimeCv = float.Parse(row["stride_time_cv"], CultureInfo.InvariantCulture),
                });
            }
            return track;
        }

        /// <summary>Frame active at a given playback time (linear scan, same
        /// approach as TrialTimeline.FrameAtTime -- trials are a few hundred
        /// to a few thousand frames, cheap enough every tick).</summary>
        public GaitFrame FrameAtTime(float timeS)
        {
            GaitFrame result = Frames.Count > 0 ? Frames[0] : default;
            foreach (var frame in Frames)
            {
                if (frame.TimestampS > timeS) break;
                result = frame;
            }
            return result;
        }
    }
}
