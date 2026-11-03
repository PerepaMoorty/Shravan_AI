using System.Collections.Generic;
using System.Globalization;

namespace ShravanAI.Playback
{
    public struct ModelEvent
    {
        public int FrameIdx;
        public float TimestampS;
        public string EventType; // pre_impact_predicted / fall_detected
        public float Confidence;
    }

    /// <summary>
    /// The model's firing log for one trial, from "&lt;trial&gt;_model_log.csv".
    /// Drives the alerts, timing strip and model avatar, never the ghost avatar.
    /// </summary>
    public class ModelFiringLog
    {
        public List<ModelEvent> Events = new List<ModelEvent>();

        public static ModelFiringLog LoadFromCsv(string path)
        {
            var log = new ModelFiringLog();
            foreach (var row in MiniCsv.ReadRows(path))
            {
                log.Events.Add(new ModelEvent
                {
                    FrameIdx = int.Parse(row["frame_idx"], CultureInfo.InvariantCulture),
                    TimestampS = float.Parse(row["timestamp_s"], CultureInfo.InvariantCulture),
                    EventType = row["event"],
                    Confidence = float.Parse(row["confidence"], CultureInfo.InvariantCulture),
                });
            }
            return log;
        }

        public IEnumerable<ModelEvent> EventsUpTo(float timeS)
        {
            foreach (var evt in Events)
            {
                if (evt.TimestampS <= timeS) yield return evt;
            }
        }
    }
}
