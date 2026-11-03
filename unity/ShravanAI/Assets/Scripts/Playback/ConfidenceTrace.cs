using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace ShravanAI.Playback
{
    public struct ConfidenceFrame
    {
        public int FrameIdx;
        public float TimestampS;
        public float ProbaPreImpact; // prediction trace only; 0 for a detection trace
        public float ProbaFall;
    }

    /// <summary>
    /// A model's raw per-window probability over one trial, from
    /// "&lt;trial&gt;_detection_confidence_trace.csv" or
    /// "&lt;trial&gt;_prediction_confidence_trace.csv" (written by
    /// architecture/ShravanAI's export scripts). Separate from ModelFiringLog,
    /// which only records threshold crossings and drives the alerts/avatar.
    /// </summary>
    public class ConfidenceTrace
    {
        public List<ConfidenceFrame> Frames = new List<ConfidenceFrame>();

        public static ConfidenceTrace LoadDetectionCsv(string path) => Load(path, hasPreImpact: false);

        public static ConfidenceTrace LoadPredictionCsv(string path) => Load(path, hasPreImpact: true);

        private static ConfidenceTrace Load(string path, bool hasPreImpact)
        {
            var trace = new ConfidenceTrace();
            if (!File.Exists(path)) return trace;

            foreach (var row in MiniCsv.ReadRows(path))
            {
                trace.Frames.Add(new ConfidenceFrame
                {
                    FrameIdx = int.Parse(row["frame_idx"], CultureInfo.InvariantCulture),
                    TimestampS = float.Parse(row["timestamp_s"], CultureInfo.InvariantCulture),
                    ProbaPreImpact = hasPreImpact ? float.Parse(row["proba_pre_impact"], CultureInfo.InvariantCulture) : 0f,
                    ProbaFall = float.Parse(row["proba_fall"], CultureInfo.InvariantCulture),
                });
            }
            return trace;
        }

        /// <summary>Latest real model output at or before a playback time;
        /// holds the value rather than interpolating between windows.</summary>
        public ConfidenceFrame FrameAtTime(float timeS)
        {
            ConfidenceFrame result = Frames.Count > 0 ? Frames[0] : default;
            foreach (var frame in Frames)
            {
                if (frame.TimestampS > timeS) break;
                result = frame;
            }
            return result;
        }
    }
}
