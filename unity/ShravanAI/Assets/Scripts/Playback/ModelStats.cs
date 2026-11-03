using System;
using System.IO;
using UnityEngine;

namespace ShravanAI.Playback
{
    [Serializable]
    public class ConfusionMatrixCounts
    {
        public int tn;
        public int fp;
        public int fn;
        public int tp;
    }

    [Serializable]
    public class ClassMetric
    {
        public float precision;
        public float recall;
        public float f1;
    }

    [Serializable]
    public class PredictionPerClass
    {
        public ClassMetric non_fall;
        public ClassMetric pre_impact;
        public ClassMetric fall;
    }

    /// <summary>
    /// Lead time (ms before impact the model first flagged pre_impact) over
    /// the curated demo trials. mean/median are only meaningful when
    /// n_flagged &gt; 0 (JsonUtility has no nullable floats, so unset reads as 0).
    /// </summary>
    [Serializable]
    public class PredictionLeadTime
    {
        public int n_trials;
        public int n_flagged;
        public float detection_rate;
        public float mean_lead_time_ms;
        public float median_lead_time_ms;
    }

    [Serializable]
    public class PredictionStats
    {
        public bool available;
        public string architecture;
        public float test_macro_f1;
        public PredictionPerClass per_class;
        public PredictionLeadTime lead_time;
    }

    /// <summary>
    /// Held-out test-set performance from "_model_stats.json", written by
    /// architecture/ShravanAI's export_detection_to_unity.py (detection) and
    /// export_prediction_to_unity.py (the nested `prediction` section).
    /// Describes the models in general, not any one trial.
    /// </summary>
    [Serializable]
    public class ModelStats
    {
        public string[] datasets;
        public int train_subjects;
        public int val_subjects;
        public int test_subjects;
        public int n_test_samples;
        public int n_test_positive;
        public float accuracy;
        public float precision;
        public float recall;
        public float specificity;
        public float false_positive_rate;
        public float f1;
        public float roc_auc;
        public float decision_threshold;
        public ConfusionMatrixCounts confusion_matrix;
        public PredictionStats prediction;

        /// <summary>Returns null if the file doesn't exist yet.</summary>
        public static ModelStats LoadFromJson(string path)
        {
            if (!File.Exists(path)) return null;
            return JsonUtility.FromJson<ModelStats>(File.ReadAllText(path));
        }
    }
}
