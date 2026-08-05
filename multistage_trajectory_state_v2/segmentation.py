from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FEATURE_CONFIGS, MultiStageTrajectoryConfig
from .data import robust_scale


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    start_cycle: float
    end_cycle: float


def _rolling_slope(cycles: np.ndarray, values: np.ndarray, horizon: float) -> np.ndarray:
    n = len(cycles); start = np.searchsorted(cycles, cycles - horizon, side="left")
    x = cycles; sx = np.r_[0.0, np.cumsum(x)]; sx2 = np.r_[0.0, np.cumsum(x * x)]
    sy = np.vstack((np.zeros(values.shape[1]), np.cumsum(values, axis=0)))
    sxy = np.vstack((np.zeros(values.shape[1]), np.cumsum(values * x[:, None], axis=0)))
    count = np.arange(n) - start + 1; sum_x = sx[1:] - sx[start]; sum_x2 = sx2[1:] - sx2[start]
    sum_y = sy[1:] - sy[start]; sum_xy = sxy[1:] - sxy[start]
    denominator = count * sum_x2 - sum_x ** 2
    return np.divide(count[:, None] * sum_xy - sum_x[:, None] * sum_y, denominator[:, None], out=np.zeros_like(sum_y), where=np.abs(denominator[:, None]) > 1e-12)


def _rolling_volatility(cycles: np.ndarray, values: np.ndarray, horizon: float) -> np.ndarray:
    start = np.searchsorted(cycles, cycles - horizon, side="left"); count = np.arange(len(cycles)) - start + 1
    s1 = np.vstack((np.zeros(values.shape[1]), np.cumsum(values, axis=0))); s2 = np.vstack((np.zeros(values.shape[1]), np.cumsum(values ** 2, axis=0)))
    mean = (s1[1:] - s1[start]) / count[:, None]; mean2 = (s2[1:] - s2[start]) / count[:, None]
    return np.sqrt(np.maximum(mean2 - mean ** 2, 0.0)).mean(axis=1)


def _rolling_correlation(cycles: np.ndarray, left: np.ndarray, right: np.ndarray, horizon: float) -> np.ndarray:
    start = np.searchsorted(cycles, cycles - horizon, side="left"); count = np.arange(len(cycles)) - start + 1
    def sums(values: np.ndarray) -> np.ndarray: return np.r_[0.0, np.cumsum(values)]
    sx, sy, sx2, sy2, sxy = sums(left), sums(right), sums(left ** 2), sums(right ** 2), sums(left * right)
    ax, ay = (sx[1:] - sx[start]) / count, (sy[1:] - sy[start]) / count
    covariance = (sxy[1:] - sxy[start]) / count - ax * ay
    vx = np.maximum((sx2[1:] - sx2[start]) / count - ax ** 2, 0.0); vy = np.maximum((sy2[1:] - sy2[start]) / count - ay ** 2, 0.0)
    return np.divide(covariance, np.sqrt(vx * vy), out=np.zeros_like(covariance), where=(vx * vy) > 1e-12)


def causal_descriptors(frame: pd.DataFrame, config: MultiStageTrajectoryConfig, reference: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray], tuple[str, ...]]:
    """Return descriptors whose every row is based only on current and prior raw windows."""
    features = FEATURE_CONFIGS[config.primary_feature_config]; rows: list[pd.DataFrame] = []
    reference_out: tuple[np.ndarray, np.ndarray] | None = reference
    descriptor_columns: tuple[str, ...] | None = None
    for _, group in frame.groupby("dataset", sort=True):
        group = group.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
        raw = group.loc[:, list(features)].to_numpy(float)
        if reference_out is None:
            reference_out = robust_scale(raw)
        z = (raw - reference_out[0]) / reference_out[1]; cycles = group.center_cycle.to_numpy(float)
        descriptors: dict[str, np.ndarray] = {f"level__{name}": z[:, index] for index, name in enumerate(features)}
        slopes: dict[int, np.ndarray] = {}
        for horizon in (100, 500, 1000):
            slopes[horizon] = _rolling_slope(cycles, z, float(horizon)).mean(axis=1)
            descriptors[f"slope_mean_{horizon}"] = slopes[horizon]
            descriptors[f"volatility_mean_{horizon}"] = _rolling_volatility(cycles, z, float(horizon))
        descriptors["slope_short_long_gap"] = slopes[100] - slopes[1000]
        descriptors["direction_flip"] = ((slopes[100] * slopes[1000]) < 0).astype(float)
        descriptors["rx_ry_correlation_500"] = _rolling_correlation(cycles, z[:, features.index("rx_mean")], z[:, features.index("ry_mean")], 500.0)
        descriptors["rs_relative_to_rx_ry"] = z[:, features.index("rs_mean")] - .5 * (z[:, features.index("rx_mean")] + z[:, features.index("ry_mean")])
        result = group.loc[:, ["dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle"]].copy()
        for name, values in descriptors.items(): result[name] = values
        result["causal_descriptor_only"] = True; rows.append(result)
        descriptor_columns = tuple(descriptors)
    assert reference_out is not None and descriptor_columns is not None
    return pd.concat(rows, ignore_index=True), reference_out, descriptor_columns


def segments_from_consensus(frame: pd.DataFrame, consensus: pd.DataFrame, config: MultiStageTrajectoryConfig) -> list[Segment]:
    ordered = frame.sort_values(["center_cycle", "window_index"]).reset_index(drop=True); span = float(ordered.center_cycle.iloc[-1] - ordered.center_cycle.iloc[0])
    candidates = consensus.loc[(consensus.dataset.eq(str(ordered.dataset.iloc[0]))) & consensus.passed.eq(1), "center_cycle"].to_numpy(float)
    minimum = span * config.regime_min_duration_fraction; cut_indices: list[int] = []
    for cycle in sorted(candidates):
        index = int(np.searchsorted(ordered.center_cycle.to_numpy(float), cycle));
        if 0 < index < len(ordered) and (not cut_indices or ordered.center_cycle.iloc[index] - ordered.center_cycle.iloc[cut_indices[-1]] >= minimum): cut_indices.append(index)
    boundaries = [0, *cut_indices, len(ordered)]; result: list[Segment] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right - left >= 3:
            result.append(Segment(left, right, float(ordered.center_cycle.iloc[left]), float(ordered.center_cycle.iloc[right - 1])))
    return result
