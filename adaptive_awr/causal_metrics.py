from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import AdaptiveAWRConfig


def finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def robust_iqr(values: Iterable[float], eps: float = 1e-9) -> float:
    arr = finite(values)
    if arr.size == 0:
        return float(eps)
    return float(max(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25), eps))


def robust_mad(values: Iterable[float], eps: float = 1e-9) -> float:
    arr = finite(values)
    if arr.size == 0:
        return float(eps)
    median = float(np.nanmedian(arr))
    return float(max(np.nanmedian(np.abs(arr - median)) * 1.4826, eps))


def causal_rolling_median(values: Iterable[float], window: int) -> np.ndarray:
    return pd.Series(np.asarray(list(values), dtype=float)).rolling(window=window, min_periods=1).median().to_numpy(dtype=float)


def causal_rolling_mean(values: Iterable[float], window: int) -> np.ndarray:
    return pd.Series(np.asarray(list(values), dtype=float)).rolling(window=window, min_periods=1).mean().to_numpy(dtype=float)


def causal_rolling_robust_std(values: Iterable[float], window: int, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    result = np.zeros(len(arr), dtype=float)
    for pos in range(len(arr)):
        part = finite(arr[max(0, pos - window + 1) : pos + 1])
        if part.size < 3:
            result[pos] = 0.0
        else:
            result[pos] = robust_mad(part, eps)
    return result


def causal_rs(values: Sequence[float], horizon: int) -> float:
    """Median difference of adjacent trailing horizon windows, divided by horizon."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2 * horizon:
        return float("nan")
    recent = finite(arr[-horizon:])
    previous = finite(arr[-2 * horizon : -horizon])
    if recent.size == 0 or previous.size == 0:
        return float("nan")
    return float((np.nanmedian(recent) - np.nanmedian(previous)) / float(horizon))


def positive_robust_z(value: float, reference: Iterable[float], eps: float = 1e-9) -> float:
    ref = finite(reference)
    if ref.size == 0 or not np.isfinite(value):
        return 0.0
    return float(max((value - np.nanmedian(ref)) / robust_iqr(ref, eps), 0.0))


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0))))


@dataclass
class BaselineReferences:
    awr_vol: np.ndarray
    bd_jump: np.ndarray
    shape_jump: np.ndarray
    awr_p95: float
    bd_p95: float
    feature_mad: Mapping[str, float]


def build_baseline_references(
    calibration_awr: Sequence[float],
    calibration_bd: Sequence[float],
    calibration_shape: Sequence[float],
    calibration_feature_values: pd.DataFrame,
    config: AdaptiveAWRConfig,
) -> BaselineReferences:
    awr = np.asarray(calibration_awr, dtype=float)
    bd = np.asarray(calibration_bd, dtype=float)
    shape = np.asarray(calibration_shape, dtype=float)
    awr_vol = causal_rolling_robust_std(awr, config.reliability_window, config.eps)
    bd_jump = np.maximum(bd - causal_rolling_median(bd, config.reliability_window), 0.0)
    shape_jump = np.maximum(shape - causal_rolling_median(shape, config.reliability_window), 0.0)
    feature_mad = {
        feature: robust_mad(calibration_feature_values[feature].to_numpy(dtype=float), config.eps)
        for feature in calibration_feature_values.columns
    }
    return BaselineReferences(
        awr_vol=awr_vol,
        bd_jump=bd_jump,
        shape_jump=shape_jump,
        awr_p95=float(np.nanpercentile(finite(awr), 95)) if finite(awr).size else 0.0,
        bd_p95=float(np.nanpercentile(finite(bd), 95)) if finite(bd).size else 0.0,
        feature_mad=feature_mad,
    )


@dataclass
class CausalMetricTracker:
    """Stateful metric calculator that never accesses a later target window."""

    refs: BaselineReferences
    config: AdaptiveAWRConfig
    source_awr_high_threshold: float
    source_bd_high_threshold: float
    awr_history: list[float] = field(default_factory=list)
    bd_history: list[float] = field(default_factory=list)
    shape_history: list[float] = field(default_factory=list)
    high_pair_history: list[int] = field(default_factory=list)

    def step(self, awr: float, bd: float, shape: float) -> dict[str, float]:
        self.awr_history.append(float(awr))
        self.bd_history.append(float(bd))
        self.shape_history.append(float(shape))
        # The last value of a trailing robust rolling standard deviation is the
        # MAD of the current trailing segment; do not recompute its full history.
        awr_vol = robust_mad(self.awr_history[-self.config.reliability_window :], self.config.eps)
        bd_med = float(np.nanmedian(finite(self.bd_history[-self.config.reliability_window :]))) if finite(self.bd_history[-self.config.reliability_window :]).size else float(bd)
        shape_med = float(np.nanmedian(finite(self.shape_history[-self.config.reliability_window :]))) if finite(self.shape_history[-self.config.reliability_window :]).size else float(shape)
        bd_jump = max(float(bd) - bd_med, 0.0)
        shape_jump = max(float(shape) - shape_med, 0.0)
        tes = (
            0.4 * positive_robust_z(awr_vol, self.refs.awr_vol, self.config.eps)
            + 0.4 * positive_robust_z(bd_jump, self.refs.bd_jump, self.config.eps)
            + 0.2 * positive_robust_z(shape_jump, self.refs.shape_jump, self.config.eps)
        )
        high_pair = int(awr > self.source_awr_high_threshold and bd > self.source_bd_high_threshold)
        self.high_pair_history.append(high_pair)
        occupancy = float(np.mean(self.high_pair_history[-self.config.occupancy_window :]))
        rs20 = causal_rs(self.awr_history, 20)
        rs50 = causal_rs(self.awr_history, 50)
        rs100 = causal_rs(self.awr_history, 100)
        return {
            "RS20": rs20,
            "RS50": rs50,
            "RS100": rs100,
            "RS50_positive": float(max(rs50, 0.0)) if np.isfinite(rs50) else 0.0,
            "AWR_volatility": float(awr_vol),
            "BD_jump": float(bd_jump),
            "shape_jump": float(shape_jump),
            "TES": float(tes),
            "high_AWR_high_BD": int(high_pair),
            "high_AWR_high_BD_occupancy": occupancy,
        }
