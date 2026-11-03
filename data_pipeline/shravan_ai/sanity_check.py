"""Plot a trial timeline's acceleration magnitude with onset/impact
frames marked, so bad labels are visible before any Unity work starts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this writes PNGs, it doesn't open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .parser import PHASE_POST_IMPACT, PHASE_PRE_IMPACT


def plot_trial(timeline: pd.DataFrame, out_path: Path, title: str | None = None) -> Path:
    acc_mag = np.sqrt(timeline["acc_x"] ** 2 + timeline["acc_y"] ** 2 + timeline["acc_z"] ** 2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timeline["timestamp_s"], acc_mag, color="#3366cc", linewidth=1, label="Acceleration magnitude (g)")

    pre_impact = timeline.loc[timeline["phase"] == PHASE_PRE_IMPACT, "timestamp_s"]
    post_impact = timeline.loc[timeline["phase"] == PHASE_POST_IMPACT, "timestamp_s"]
    if len(pre_impact):
        ax.axvline(pre_impact.iloc[0], color="#e69138", linestyle="--", label="fall_onset_frame")
    if len(post_impact):
        ax.axvline(post_impact.iloc[0], color="#cc0000", linestyle="--", label="fall_impact_frame")

    direction = timeline["fall_direction"].iloc[0] if len(timeline) else "unknown"
    ax.set_title(title or f"{timeline['subject_id'].iloc[0]}/{timeline['trial_id'].iloc[0]} ({direction})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration magnitude (g)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
