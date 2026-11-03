using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace ShravanAI.Playback
{
    [Serializable]
    internal class TrialIndex
    {
        public string[] trials;
    }

    /// <summary>
    /// Cycles playback through the trials in StreamingAssets/Trials/_index.json
    /// (or the Inspector's trialKeys list). The first load is deferred one frame
    /// so it runs after TrialTimelineLoader's own Start() load.
    /// </summary>
    public class AutoPlaylistController : MonoBehaviour
    {
        public TrialTimelineLoader trialLoader;
        public SharedClock clock;

        [Tooltip("If empty, plays every trial in StreamingAssets/Trials/_index.json.")]
        public List<string> trialKeys = new List<string>();

        public bool autoStart = true;
        public bool loopPlaylist = true;

        [Tooltip("Real-world seconds to hold the final frame before advancing.")]
        public float interTrialPauseS = 2.5f;

        public int CurrentIndex { get; private set; } = -1;

        private Coroutine _advanceRoutine;

        private void Awake()
        {
            if (trialKeys.Count == 0) trialKeys = LoadIndexTrialKeys();
        }

        private void OnEnable()
        {
            if (clock != null) clock.OnPlaybackFinished += OnPlaybackFinished;
        }

        private void OnDisable()
        {
            if (clock != null) clock.OnPlaybackFinished -= OnPlaybackFinished;
        }

        private void Start()
        {
            if (autoStart && trialKeys.Count > 0) StartCoroutine(BeginNextFrame());
        }

        private IEnumerator BeginNextFrame()
        {
            yield return null;
            PlayIndex(0);
        }

        private List<string> LoadIndexTrialKeys()
        {
            string path = Path.Combine(Application.streamingAssetsPath, "Trials", "_index.json");
            if (!File.Exists(path))
            {
                Debug.LogWarning($"[ShravanAI playlist] {path} not found -- nothing to auto-play. " +
                                  "Run data_pipeline/scripts/export_for_unity.py first.");
                return new List<string>();
            }

            TrialIndex index = JsonUtility.FromJson<TrialIndex>(File.ReadAllText(path));
            return index != null && index.trials != null ? new List<string>(index.trials) : new List<string>();
        }

        private void PlayIndex(int index)
        {
            if (trialKeys.Count == 0 || trialLoader == null || clock == null) return;

            CurrentIndex = ((index % trialKeys.Count) + trialKeys.Count) % trialKeys.Count;
            string key = trialKeys[CurrentIndex];
            Debug.Log($"[ShravanAI playlist] Playing {CurrentIndex + 1}/{trialKeys.Count}: {key}");

            trialLoader.LoadTrial(key);
            clock.Play();
        }

        private void OnPlaybackFinished()
        {
            if (_advanceRoutine != null) StopCoroutine(_advanceRoutine);
            _advanceRoutine = StartCoroutine(AdvanceAfterPause());
        }

        private IEnumerator AdvanceAfterPause()
        {
            bool isLastTrial = CurrentIndex >= trialKeys.Count - 1;
            if (isLastTrial && !loopPlaylist)
            {
                Debug.Log("[ShravanAI playlist] Reached the end of the playlist (loop disabled) -- stopping.");
                yield break;
            }

            Debug.Log($"[ShravanAI playlist] Trial finished -- advancing in {interTrialPauseS:0.0}s");
            yield return new WaitForSeconds(interTrialPauseS);
            PlayIndex(CurrentIndex + 1);
        }
    }
}
