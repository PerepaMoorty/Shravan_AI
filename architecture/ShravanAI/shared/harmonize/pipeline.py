"""Harmonization orchestrator.

Composes the four Stage 3 building blocks into one per-trial function,
in the frozen order: unit conversion -> resample -> axis alignment ->
filter.

IMPORTANT, non-obvious consequence of this order + the 0.5-20 Hz
band-pass filter: gravity is (by definition) a 0 Hz / DC signal. Axis
alignment correctly rotates it onto the canonical vertical axis, but
the subsequent band-pass filter then REMOVES that near-DC content --
same as it removes any slow postural drift, by design. So a harmonized trial's output will NOT show a
persistent ~1g bias on acc_z, even for a perfectly still trial. This is
intentional: the filter's job is to isolate movement dynamics, not
preserve absolute orientation. Don't be alarmed if you check a
harmonized T01 trial and acc_z isn't sitting near 1.0 -- check that the
MOVEMENT-frequency content lines up on the right axis instead (see
test_pipeline.py for how that's verified).

Channel restriction: the output only contains time_s + the six
acc_*/gyro_* columns, even if the input trial had extra channels (e.g.
KFall's Euler angles) -- this is what makes the harmonized output
directly comparable across KFall/SisFall/FallAllD later. Callers who
want KFall's Euler angles for a KFall-only experiment should read them
from the original trial.signal directly, before harmonization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from shared.harmonize.axis_alignment import CalibrationResult, apply_rotation
from shared.harmonize.filtering import apply_bandpass_filter
from shared.harmonize.resample import resample_signal
from shared.harmonize.units import ACCEL_COLUMNS, GYRO_COLUMNS, get_unit_converter

_CANONICAL_COLUMNS = ["time_s"] + ACCEL_COLUMNS + GYRO_COLUMNS + ["tilt_deg"]


@dataclass
class HarmonizationConfig:
    target_rate_hz: float = 100.0
    filter_low_hz: float = 0.5
    filter_high_hz: float = 20.0
    filter_order: int = 4


def harmonize_trial(
    trial: Any,
    calibration: CalibrationResult,
    config: HarmonizationConfig,
) -> pd.DataFrame:
    """Run one trial through the full harmonization pipeline.

    `trial` must have `.signal` (a DataFrame with time_s, acc_*, gyro_*
    columns) and `.metadata.dataset` (str, e.g. "kfall") and
    `.metadata.native_rate_hz` (float). Returns a new DataFrame with
    exactly `_CANONICAL_COLUMNS`, at `config.target_rate_hz`, aligned
    and filtered.
    """
    converter = get_unit_converter(trial.metadata.dataset)
    converted = converter.convert(trial.signal)

    resampled = resample_signal(
        converted,
        native_rate_hz=trial.metadata.native_rate_hz,
        target_rate_hz=config.target_rate_hz,
    )

    aligned = apply_rotation(resampled, calibration.rotation)

    # Tilt (angle between the acceleration vector and vertical) is
    # computed here, on the ALIGNED-BUT-NOT-YET-FILTERED signal --
    # deliberately, not on the final band-pass-filtered output below.
    # Gravity is a 0 Hz/DC signal, and the band-pass filter removes it
    # by design (see this module's own docstring) -- so on the FILTERED
    # signal, a genuinely still moment has almost no acceleration left
    # at all, and dividing that near-zero vector's z-component by its
    # own near-zero magnitude produces an angle dominated by sensor
    # noise, not real posture (observed on real data: a stationary
    # trial's tilt swinging between ~10 deg and ~170 deg from one
    # sample to the next). Computing it here instead, while gravity is
    # still present, means tilt stays close to 0 deg during genuinely
    # still moments and only moves meaningfully when the body actually
    # tips over -- the property the feature is supposed to capture.
    aligned_acc = aligned[ACCEL_COLUMNS].to_numpy()
    aligned_norm = np.linalg.norm(aligned_acc, axis=1)
    aligned_norm_safe = np.where(aligned_norm < 1e-6, 1e-6, aligned_norm)
    cos_theta = np.clip(aligned_acc[:, 2] / aligned_norm_safe, -1.0, 1.0)
    tilt_deg = np.degrees(np.arccos(cos_theta))

    filter_columns = ACCEL_COLUMNS + GYRO_COLUMNS
    filtered = apply_bandpass_filter(
        aligned,
        columns=filter_columns,
        sample_rate_hz=config.target_rate_hz,
        low_hz=config.filter_low_hz,
        high_hz=config.filter_high_hz,
        order=config.filter_order,
    )
    filtered = filtered.reset_index(drop=True)
    filtered["tilt_deg"] = tilt_deg

    return filtered[_CANONICAL_COLUMNS].reset_index(drop=True)
