from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import AdaptiveAWRV11Config


def finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def robust_iqr(values: Iterable[float], eps: float = 1e-9) -> float:
    arr = finite(values)
    return float(max(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25), eps)) if arr.size else float(eps)


def robust_mad(values: Iterable[float], eps: float = 1e-9) -> float:
    arr = finite(values)
    if arr.size == 0:
        return float(eps)
    return float(max(np.nanmedian(np.abs(arr - np.nanmedian(arr))) * 1.4826, eps))


def causal_median(values: Sequence[float], window: int) -> float:
    part = finite(values[-window:])
    return float(np.nanmedian(part)) if part.size else 0.0


def causal_rs(values: Sequence[float], horizon: int) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) < horizon * 2:
        return float("nan")
    now, prior = finite(arr[-horizon:]), finite(arr[-2 * horizon : -horizon])
    if now.size == 0 or prior.size == 0:
        return float("nan")
    return float((np.nanmedian(now) - np.nanmedian(prior)) / horizon)


def positive_robust_z(value: float, reference: Iterable[float], eps: float = 1e-9) -> float:
    ref = finite(reference)
    if not ref.size or not np.isfinite(value):
        return 0.0
    return float(max((value - np.nanmedian(ref)) / robust_iqr(ref, eps), 0.0))


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0))))


def event_risk_from_evidence(tes_z: float, bd_jump_z: float) -> tuple[float, float]:
    strength = max(1.5 * float(tes_z) + 0.5 * float(bd_jump_z), 0.0)
    return float(strength), float(1.0 - np.exp(-strength))


def boundary_guard_metadata(frame: pd.DataFrame, config: AdaptiveAWRV11Config) -> pd.DataFrame:
    """Identify windows at a known stop/restart boundary without using stage labels."""
    rows = []
    interval = config.known_stop_interval_cycles
    for row in frame.itertuples(index=False):
        start, end, center = float(row.start_cycle), float(row.end_cycle), float(row.center_cycle)
        lower = int(np.floor(start / interval))
        upper = int(np.floor(end / interval))
        crossed = lower != upper
        candidate = int(round(center / interval)) * interval
        candidates = [max(interval, lower * interval), max(interval, upper * interval), max(interval, candidate)]
        boundary = min(candidates, key=lambda value: abs(center - value))
        since = center - boundary
        guard = bool(crossed or (0.0 <= since <= config.restart_guard_cycles))
        rows.append(
            {
                "window_index": int(row.window_index),
                "is_restart_guard": int(guard),
                "nearest_stop_boundary": float(boundary),
                "cycles_since_stop_boundary": float(since),
                "crosses_stop_boundary": int(crossed),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class MetricReferences:
    awr_vol: np.ndarray
    bd_jump: np.ndarray
    shape_jump: np.ndarray
    awr_p95: float
    bd_p95: float
    feature_mad: Mapping[str, float]


def build_metric_references(
    calibration_awr: Sequence[float],
    calibration_bd: Sequence[float],
    calibration_shape: Sequence[float],
    calibration_features: pd.DataFrame,
    config: AdaptiveAWRV11Config,
) -> MetricReferences:
    awr, bd, shape = map(lambda value: np.asarray(value, dtype=float), (calibration_awr, calibration_bd, calibration_shape))
    awr_vol = np.asarray([robust_mad(awr[max(0, i - config.reliability_window + 1) : i + 1], config.eps) for i in range(len(awr))])
    bd_jump = np.asarray([max(bd[i] - causal_median(bd[: i + 1], config.reliability_window), 0.0) for i in range(len(bd))])
    shape_jump = np.asarray([max(shape[i] - causal_median(shape[: i + 1], config.reliability_window), 0.0) for i in range(len(shape))])
    feature_mad = {name: robust_mad(calibration_features[name].to_numpy(dtype=float), config.eps) for name in calibration_features.columns}
    return MetricReferences(
        awr_vol=awr_vol,
        bd_jump=bd_jump,
        shape_jump=shape_jump,
        awr_p95=float(np.nanpercentile(finite(awr), 95)) if finite(awr).size else 0.0,
        bd_p95=float(np.nanpercentile(finite(bd), 95)) if finite(bd).size else 0.0,
        feature_mad=feature_mad,
    )


@dataclass
class CausalMetricTrackerV11:
    refs: MetricReferences
    config: AdaptiveAWRV11Config
    source_awr_high: float
    source_bd_high: float
    awr_history: list[float] = field(default_factory=list)
    bd_history: list[float] = field(default_factory=list)
    shape_history: list[float] = field(default_factory=list)
    high_pair_history: list[int] = field(default_factory=list)

    def step(self, awr: float, bd: float, shape: float, is_restart_guard: bool) -> dict[str, float]:
        self.awr_history.append(float(awr))
        self.bd_history.append(float(bd))
        self.shape_history.append(float(shape))
        awr_vol = robust_mad(self.awr_history[-self.config.reliability_window :], self.config.eps)
        bd_jump = max(float(bd) - causal_median(self.bd_history, self.config.reliability_window), 0.0)
        shape_jump = max(float(shape) - causal_median(self.shape_history, self.config.reliability_window), 0.0)
        raw_tes = (
            0.4 * positive_robust_z(awr_vol, self.refs.awr_vol, self.config.eps)
            + 0.4 * positive_robust_z(bd_jump, self.refs.bd_jump, self.config.eps)
            + 0.2 * positive_robust_z(shape_jump, self.refs.shape_jump, self.config.eps)
        )
        clean_tes = 0.0 if is_restart_guard else raw_tes
        high_pair = int(awr > self.source_awr_high and bd > self.source_bd_high)
        self.high_pair_history.append(high_pair)
        return {
            "RS20": causal_rs(self.awr_history, 20),
            "RS50": causal_rs(self.awr_history, 50),
            "RS100": causal_rs(self.awr_history, 100),
            "RS50_positive": max(causal_rs(self.awr_history, 50), 0.0) if np.isfinite(causal_rs(self.awr_history, 50)) else 0.0,
            "AWR_volatility": float(awr_vol),
            "BD_jump": float(bd_jump),
            "shape_jump": float(shape_jump),
            "TES_raw": float(raw_tes),
            "TES_clean": float(clean_tes),
            "high_AWR_high_BD": int(high_pair),
            "high_AWR_high_BD_occupancy": float(np.mean(self.high_pair_history[-self.config.occupancy_window :])),
        }
