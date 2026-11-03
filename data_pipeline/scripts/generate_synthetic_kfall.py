#!/usr/bin/env python3
"""Generate a tiny synthetic KFall-shaped dataset (three fall directions +
one ADL trial) for tests and runs without the real data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_RATE_HZ = 100.0

TRIALS = [
    # subject, trial_id, task_code, description, direction, has_fall
    ("SA06", "S06T01R01", "F09", "Forward fall while walking caused by a trip", "forward", True),
    ("SA06", "S06T05R01", "F05", "Backward fall while walking caused by a slip", "backward", True),
    ("SA06", "S06T08R01", "F08", "Lateral fall while walking caused by a slip", "lateral", True),
    ("SA06", "S06T20R01", "D01", "Walking slowly on a flat surface", "adl", False),
]

DURATION_S = 6.0


def _synthesize_sensor_frame(direction: str, has_fall: bool, onset_frame: int | None, impact_frame: int | None):
    n_frames = int(DURATION_S * SAMPLE_RATE_HZ)
    t = np.arange(n_frames) / SAMPLE_RATE_HZ
    rng = np.random.default_rng(hash((direction, has_fall)) % (2**32))

    acc_x = rng.normal(0, 0.05, n_frames)
    acc_y = rng.normal(0, 0.05, n_frames)
    acc_z = np.full(n_frames, 1.0) + rng.normal(0, 0.03, n_frames)  # resting on gravity
    gyr_x = rng.normal(0, 5, n_frames)
    gyr_y = rng.normal(0, 5, n_frames)
    gyr_z = rng.normal(0, 5, n_frames)
    euler_x = np.zeros(n_frames)
    euler_y = np.zeros(n_frames)
    euler_z = np.zeros(n_frames)

    if has_fall and onset_frame is not None and impact_frame is not None:
        # Instability ramps up from onset, spikes hard at impact, then settles
        # lying down (near-zero motion) — direction only changes which axis
        # carries the dominant component.
        axis_map = {
            "forward": (acc_x, gyr_y),
            "backward": (acc_x, gyr_y),
            "lateral": (acc_y, gyr_x),
        }
        primary_acc, primary_gyr = axis_map[direction]
        sign = -1.0 if direction == "backward" else 1.0

        for i in range(onset_frame, min(impact_frame, n_frames)):
            ramp = (i - onset_frame) / max(1, (impact_frame - onset_frame))
            primary_gyr[i] += sign * 150 * ramp

        impact_end = min(impact_frame + 5, n_frames)
        for i in range(impact_frame, impact_end):
            spike = 1.0 - (i - impact_frame) / 5
            primary_acc[i] += sign * 3.0 * spike
            acc_z[i] += 1.5 * spike

        # Lying down afterward: flat orientation, low motion
        for i in range(impact_end, n_frames):
            acc_z[i] = 0.2 + rng.normal(0, 0.02)
            primary_gyr[i] = rng.normal(0, 2)

    return pd.DataFrame(
        {
            "TimeStamp(s)": t,
            "FrameCounter": np.arange(n_frames),
            "AccX": acc_x,
            "AccY": acc_y,
            "AccZ": acc_z,
            "GyrX": gyr_x,
            "GyrY": gyr_y,
            "GyrZ": gyr_z,
            "EulerX": euler_x,
            "EulerY": euler_y,
            "EulerZ": euler_z,
        }
    )


def main(out_root: Path | None = None) -> Path:
    out_root = out_root or REPO_ROOT / "sample_data"
    sensor_dir = out_root / "sensor_data"
    label_dir = out_root / "label_data"

    labels_by_subject: dict[str, list[dict]] = {}

    for subject, trial_id, task_code, description, direction, has_fall in TRIALS:
        onset_frame = int(2.0 * SAMPLE_RATE_HZ) if has_fall else None
        impact_frame = int(2.8 * SAMPLE_RATE_HZ) if has_fall else None

        sensor_df = _synthesize_sensor_frame(direction, has_fall, onset_frame, impact_frame)
        subject_dir = sensor_dir / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        sensor_df.to_csv(subject_dir / f"{trial_id}.csv", index=False)

        labels_by_subject.setdefault(subject, []).append(
            {
                "Task Code": task_code,
                "Description": description,
                "Trial ID": trial_id,
                "Fall_onset_frame": onset_frame if onset_frame is not None else "",
                "Fall_impact_frame": impact_frame if impact_frame is not None else "",
            }
        )

    label_dir.mkdir(parents=True, exist_ok=True)
    for subject, rows in labels_by_subject.items():
        pd.DataFrame(rows).to_csv(label_dir / f"{subject}_label.csv", index=False)

    print(f"Wrote synthetic KFall-shaped dataset to {out_root}")
    return out_root


if __name__ == "__main__":
    main()
