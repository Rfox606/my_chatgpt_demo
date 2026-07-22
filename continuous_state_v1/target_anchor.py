from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ContinuousStateV1Config
from .data import assert_label_free


@dataclass(frozen=True)
class TargetAnchor:
    baseline_mask: np.ndarray
    awr_baseline_median: float
    anchor_method: str


def baseline_non_guard_mask(frame: pd.DataFrame, config: ContinuousStateV1Config) -> np.ndarray:
    assert_label_free(frame)
    mask = (
        (frame["end_cycle"].to_numpy(dtype=float) <= config.baseline_cycles)
        & (frame["is_restart_guard"].to_numpy(dtype=int) == 0)
    )
    if int(mask.sum()) >= 20:
        return mask
    usable = np.flatnonzero(frame["is_restart_guard"].to_numpy(dtype=int) == 0)
    fallback = np.zeros(len(frame), dtype=bool)
    fallback[usable[: min(100, len(usable))]] = True
    if not fallback.any():
        raise ValueError("No non-guard windows are available to establish a baseline")
    return fallback


def fit_target_anchor(
    frame: pd.DataFrame, awr_raw: np.ndarray, config: ContinuousStateV1Config
) -> TargetAnchor:
    assert_label_free(frame)
    baseline_mask = baseline_non_guard_mask(frame, config)
    method = "end_cycle_le_baseline_cycles" if baseline_mask.sum() >= 20 else "earliest_non_guard_fallback"
    return TargetAnchor(
        baseline_mask=baseline_mask,
        awr_baseline_median=float(np.median(np.asarray(awr_raw, dtype=float)[baseline_mask])),
        anchor_method=method,
    )


def score_awr(
    frame: pd.DataFrame,
    normalized_weight: np.ndarray,
    source_scale: float,
    config: ContinuousStateV1Config,
) -> tuple[pd.DataFrame, TargetAnchor]:
    assert_label_free(frame)
    result = frame.copy()
    raw = result.loc[:, list(config.stable_plus_features)].to_numpy(float) @ np.asarray(
        normalized_weight, dtype=float
    )
    anchor = fit_target_anchor(result, raw, config)
    result["AWR_raw"] = raw
    result["AWR_rel"] = raw - anchor.awr_baseline_median
    result["AWR_scaled"] = result["AWR_rel"] / max(float(source_scale), config.eps)
    return result, anchor


def source_awr_scale(
    source_frame: pd.DataFrame, normalized_weight: np.ndarray, config: ContinuousStateV1Config
) -> float:
    assert_label_free(source_frame)
    raw = source_frame.loc[:, list(config.stable_plus_features)].to_numpy(float) @ np.asarray(
        normalized_weight, dtype=float
    )
    baseline = baseline_non_guard_mask(source_frame, config)
    non_guard = source_frame["is_restart_guard"].to_numpy(dtype=int) == 0
    def iqr(values: np.ndarray) -> float:
        return float(np.percentile(values, 75) - np.percentile(values, 25)) if len(values) else 0.0
    return max(iqr(raw[baseline]), 0.25 * iqr(raw[non_guard]), 1e-6)
