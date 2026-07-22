from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baseline_distance import DistanceBaseline, distance_values, fit_distance_baseline
from .common_axis import causal_rolling_median
from .config import ContinuousStateV2Config
from .data import assert_label_free, baseline_non_guard_mask


def _iqr(values: np.ndarray, eps: float = 1e-9) -> float:
    return float(max(np.quantile(values, .75) - np.quantile(values, .25), eps)) if len(values) else eps


def _rate(history: list[float], length: int) -> float:
    if len(history) < length:
        return 0.0
    half = length // 2
    return float((np.median(history[-half:]) - np.median(history[-length:-half])) / half)


@dataclass(frozen=True)
class StateSpace:
    features: tuple[str, ...]
    common_indices: tuple[int, ...]
    w_common: np.ndarray
    w_branch: np.ndarray
    p_anchor: float
    b_anchor: float
    distance: DistanceBaseline
    baseline_mask: np.ndarray

    def score(self, values: np.ndarray) -> tuple[float, float, float, float, str]:
        p_raw = float(values[list(self.common_indices)] @ self.w_common) if len(self.common_indices) else 0.0
        b_raw = float(values @ self.w_branch)
        bd, diag = distance_values(values.reshape(1, -1), self.distance)
        return p_raw - self.p_anchor, float(bd[0]), b_raw - self.b_anchor, float(diag[0]), self.distance.method


def make_state_space(frame: pd.DataFrame, features: tuple[str, ...], common_features: tuple[str, ...], w_common: np.ndarray, w_branch: np.ndarray, config: ContinuousStateV2Config) -> StateSpace:
    assert_label_free(frame)
    values = frame.loc[:, list(features)].to_numpy(float)
    mask = baseline_non_guard_mask(frame, config).to_numpy(bool)
    common_indices = tuple(features.index(feature) for feature in common_features)
    p_raw = values[:, list(common_indices)] @ w_common if common_indices else np.zeros(len(values))
    b_raw = values @ w_branch
    return StateSpace(features, common_indices, w_common, w_branch, float(np.median(p_raw[mask])), float(np.median(b_raw[mask])), fit_distance_baseline(frame, features, config), mask)


def baseline_tes_reference(frame: pd.DataFrame, space: StateSpace, config: ContinuousStateV2Config) -> dict[str, float]:
    """Reference derives exclusively from initial non-guard baseline values."""
    values = frame.loc[:, list(space.features)].to_numpy(float)
    if len(values) == len(space.baseline_mask):
        indices = np.flatnonzero(space.baseline_mask)
    else:
        # Prefix-causality checks intentionally reuse the frozen calibration space
        # on a shorter target prefix, so its stored full-frame mask is not aligned.
        indices = np.flatnonzero((frame.end_cycle <= config.baseline_cycles) & frame.is_restart_guard.eq(0))
    p, bd, b = [], [], []
    for index in indices:
        one = space.score(values[index]); p.append(one[0]); bd.append(one[1]); b.append(one[2])
    p = np.asarray(p); bd = np.asarray(bd); b = np.asarray(b)
    vol = np.asarray([np.median(np.abs(p[max(0, i - 19):i + 1] - np.median(p[max(0, i - 19):i + 1]))) for i in range(len(p))])
    jump_bd = np.asarray([max(bd[i] - np.median(bd[max(0, i - 19):i + 1]), 0.) for i in range(len(bd))])
    jump_b = np.asarray([abs(b[i] - np.median(b[max(0, i - 19):i + 1])) for i in range(len(b))])
    tes = .4 * np.maximum((vol - np.median(vol)) / _iqr(vol), 0) + .4 * np.maximum((jump_bd - np.median(jump_bd)) / _iqr(jump_bd), 0) + .2 * np.maximum((jump_b - np.median(jump_b)) / _iqr(jump_b), 0)
    return {"p_vol_median": float(np.median(vol)), "p_vol_iqr": _iqr(vol), "bd_jump_median": float(np.median(jump_bd)), "bd_jump_iqr": _iqr(jump_bd), "b_jump_median": float(np.median(jump_b)), "b_jump_iqr": _iqr(jump_b), "tes_p95": float(np.quantile(tes, config.tes_update_max_quantile))}


@dataclass
class CausalStateTracker:
    space: StateSpace
    tes_ref: dict[str, float]
    config: ContinuousStateV2Config
    p_history: list[float] = field(default_factory=list)
    bd_history: list[float] = field(default_factory=list)
    b_history: list[float] = field(default_factory=list)

    def predict(self, values: np.ndarray) -> dict[str, float | str]:
        p, bd, branch, diag, method = self.space.score(values)
        p_history, bd_history, b_history = [*self.p_history, p], [*self.bd_history, bd], [*self.b_history, branch]
        recent_p = np.asarray(p_history[-20:]); recent_bd = np.asarray(bd_history[-20:]); recent_b = np.asarray(b_history[-20:])
        vol = float(np.median(np.abs(recent_p - np.median(recent_p))))
        bd_jump = max(bd - float(np.median(recent_bd)), 0.)
        b_jump = abs(branch - float(np.median(recent_b)))
        tes = .4 * max((vol - self.tes_ref["p_vol_median"]) / self.tes_ref["p_vol_iqr"], 0.) + .4 * max((bd_jump - self.tes_ref["bd_jump_median"]) / self.tes_ref["bd_jump_iqr"], 0.) + .2 * max((b_jump - self.tes_ref["b_jump_median"]) / self.tes_ref["b_jump_iqr"], 0.)
        result: dict[str, float | str] = {"P_common": p, "P_smooth_5": float(np.median(p_history[-5:])), "P_smooth_20": float(np.median(p_history[-20:])), "BD": bd, "BD_diag": diag, "bd_method": method, "B_terminal": branch, "TES": tes, "P_short_volatility": vol}
        for length in self.config.rs_horizons_windows:
            result[f"P_RS{length}"] = _rate(p_history, length)
            result[f"BD_RS{length}"] = _rate(bd_history, length)
            result[f"B_RS{length}"] = _rate(b_history, length)
        return result

    def append_pre_update(self, output: dict[str, float | str]) -> None:
        self.p_history.append(float(output["P_common"])); self.bd_history.append(float(output["BD"])); self.b_history.append(float(output["B_terminal"]))


def frozen_state_scores(frame: pd.DataFrame, space: StateSpace, config: ContinuousStateV2Config) -> pd.DataFrame:
    tracker = CausalStateTracker(space, baseline_tes_reference(frame, space, config), config)
    rows = []
    for values in frame.loc[:, list(space.features)].to_numpy(float):
        row = tracker.predict(values); tracker.append_pre_update(row); rows.append(row)
    return pd.DataFrame(rows)
