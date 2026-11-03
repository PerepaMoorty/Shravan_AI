using System;
using System.IO;
using UnityEngine;

namespace ShravanAI.Playback
{
    /// <summary>
    /// Per-trial ground-truth summary from "&lt;trial&gt;_meta.json", written by
    /// data_pipeline/scripts/export_for_unity.py.
    /// </summary>
    [Serializable]
    public class TrialMeta
    {
        public string subject_id;
        public string trial_id;
        public string fall_direction;
        public float duration_s;

        public bool has_onset;
        public float onset_timestamp_s;
        public bool has_impact;
        public float impact_timestamp_s;

        public static TrialMeta LoadFromJson(string path)
        {
            return JsonUtility.FromJson<TrialMeta>(File.ReadAllText(path));
        }
    }
}
