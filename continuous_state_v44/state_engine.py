from __future__ import annotations

"""Causal, effective-cycle continuous state metrics with frozen self-baselines."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import ContinuousStateV44Config
from .data import assert_label_free, baseline_mask, robust_location_scale


@dataclass(frozen=True)
class GroupReference:
    features: tuple[str, ...]
    location: np.ndarray
    scale: np.ndarray
    precision: np.ndarray


@dataclass(frozen=True)
class BaselineReference:
    features: tuple[str, ...]
    groups: dict[str, GroupReference]
    baseline_count: int
    distance_form: str


def feature_subset(features: tuple[str, ...], excluded_group: str | None = None) -> tuple[str, ...]:
    if excluded_group is None: return features
    if excluded_group not in {"rx", "ry", "rs"}: raise ValueError(f"Unknown feature group: {excluded_group}")
    kept = tuple(name for name in features if not name.startswith(f"{excluded_group}_"))
    if not kept: raise ValueError("Feature-group ablation removed every feature")
    return kept


def _group_features(features: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    groups = {group: tuple(name for name in features if name.startswith(f"{group}_")) for group in ("rs", "rx", "ry")}
    return {group: members for group, members in groups.items() if members}


def _precision(values: np.ndarray, distance_form: str) -> np.ndarray:
    if distance_form == "diagonal": return np.eye(values.shape[1])
    if distance_form != "mahalanobis": raise ValueError(f"Unsupported distance form: {distance_form}")
    return LedoitWolf(assume_centered=False).fit(values).precision_


def _distance(values: np.ndarray, precision: np.ndarray, distance_form: str) -> np.ndarray:
    if distance_form == "diagonal": return np.sqrt(np.mean(values * values, axis=1))
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", values, precision, values), 0.0) / values.shape[1])


def _fused_distance(values: dict[str, np.ndarray], refs: dict[str, GroupReference], distance_form: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    per_group = {group: _distance(values[group], refs[group].precision, distance_form) for group in refs}
    return np.mean(np.vstack(list(per_group.values())), axis=0), per_group


def _starts(cycles: np.ndarray, periods: tuple[int, ...]) -> dict[int, np.ndarray]:
    return {period: np.searchsorted(cycles, cycles - float(period), side="left") for period in periods}


def _velocity(values: np.ndarray, cycles: np.ndarray, starts: np.ndarray, eps: float) -> np.ndarray:
    index = np.arange(len(values)); elapsed = np.maximum(cycles - cycles[starts], eps)
    result = (values - values[starts]) / elapsed[:, None] * 100.0
    result[starts == index] = 0.0
    return result


def _rolling_group_volatility(values: np.ndarray, starts: np.ndarray, ref: GroupReference, distance_form: str) -> np.ndarray:
    total = np.vstack([np.zeros(values.shape[1]), np.cumsum(values, axis=0)])
    squares = np.vstack([np.zeros(values.shape[1]), np.cumsum(values * values, axis=0)])
    right = np.arange(len(values)) + 1; count = (right - starts).astype(float)
    mean = (total[right] - total[starts]) / count[:, None]
    variance = np.maximum((squares[right] - squares[starts]) / count[:, None] - mean * mean, 0.0)
    if distance_form == "diagonal": return np.sqrt(np.mean(variance, axis=1))
    return np.sqrt(np.maximum(np.einsum("ij,jj->i", variance, ref.precision), 0.0) / values.shape[1])


def run_target_state(frame: pd.DataFrame, state_id: str, features: tuple[str, ...], config: ContinuousStateV44Config, *, include_velocity_details: bool = False) -> tuple[pd.DataFrame, BaselineReference]:
    """Freeze first effective-cycle baseline, then compute each row from its prefix only."""
    assert_label_free(frame)
    ordered = frame.sort_values(["center_cycle_effective", "window_index"]).reset_index(drop=True).copy()
    baseline = baseline_mask(ordered, config); raw = ordered.loc[:, list(features)].to_numpy(float)
    refs: dict[str, GroupReference] = {}; normalized: dict[str, np.ndarray] = {}
    for group, members in _group_features(features).items():
        positions = [features.index(name) for name in members]
        location, scale = robust_location_scale(raw[baseline][:, positions], config.eps)
        values = (raw[:, positions] - location) / scale
        refs[group] = GroupReference(members, location, scale, _precision(values[baseline], config.distance_form))
        normalized[group] = values
    cycles = ordered.center_cycle_effective.to_numpy(float)
    periods = tuple(set((*config.velocity_windows_cycles, config.volatility_window_cycles)))
    starts = _starts(cycles, periods)
    vectors = {period: {group: _velocity(values, cycles, starts[period], config.eps) for group, values in normalized.items()} for period in config.velocity_windows_cycles}
    d_state, d_groups = _fused_distance(normalized, refs, config.distance_form)
    v100, v100_groups = _fused_distance(vectors[100], refs, config.distance_form)
    v500, v500_groups = _fused_distance(vectors[500], refs, config.distance_form)
    v1000, v1000_groups = _fused_distance(vectors[1000], refs, config.distance_form)
    divergence, divergence_groups = _fused_distance({group: vectors[100][group] - vectors[1000][group] for group in refs}, refs, config.distance_form)
    volatility_groups = {group: _rolling_group_volatility(normalized[group], starts[config.volatility_window_cycles], refs[group], config.distance_form) for group in refs}
    volatility = np.mean(np.vstack(list(volatility_groups.values())), axis=0)
    monitor = ordered.start_cycle_effective.to_numpy(float) > float(config.baseline_cycles)
    selected = np.flatnonzero(monitor); monitored = ordered.loc[monitor].reset_index(drop=True)
    output: dict[str, object] = {
        "state_id": state_id, "dataset": monitored.dataset.to_numpy(), "window_id": monitored.window_id.to_numpy(), "window_index": monitored.window_index.to_numpy(),
        "start_cycle_effective": monitored.start_cycle_effective.to_numpy(float), "end_cycle_effective": monitored.end_cycle_effective.to_numpy(float), "center_cycle_effective": monitored.center_cycle_effective.to_numpy(float),
        "start_cycle_actual": monitored.start_cycle_actual.to_numpy(float), "end_cycle_actual": monitored.end_cycle_actual.to_numpy(float), "center_cycle_actual": monitored.center_cycle_actual.to_numpy(float),
        "cycle_effective": monitored.cycle_effective.to_numpy(float), "cycle_actual": monitored.cycle_actual.to_numpy(float),
        "baseline_cycles": np.full(len(monitored), config.baseline_cycles), "baseline_frozen": np.ones(len(monitored), dtype=int), "distance_form": np.full(len(monitored), config.distance_form),
        "feature_count": np.full(len(monitored), len(features)), "feature_names": np.full(len(monitored), ";".join(features)),
        "D_state": d_state[monitor], "V100_norm": v100[monitor], "V500_norm": v500[monitor], "V1000_norm": v1000[monitor],
        "A_state": divergence[monitor], "state_volatility": volatility[monitor],
    }
    if include_velocity_details:
        for group, value in d_groups.items(): output[f"D_{group}_subspace"] = value[monitor]
        for period, label, per_group in ((100, "velocity_v100", v100_groups), (500, "velocity_v500", v500_groups), (1000, "velocity_v1000", v1000_groups)):
            total = np.sum(np.vstack(list(per_group.values())), axis=0)
            for group, per_distance in per_group.items():
                output[f"{label}_{group}_subspace_distance"] = per_distance[monitor]
                output[f"{label}_{group}_contribution"] = np.divide(per_distance[monitor], total[monitor], out=np.zeros(len(selected)), where=total[monitor] > config.eps)
                for position, name in enumerate(refs[group].features): output[f"{label}_{name}"] = vectors[period][group][selected, position]
    states = pd.DataFrame(output); assert_label_free(states)
    return states, BaselineReference(features, refs, int(baseline.sum()), config.distance_form)


def state_columns() -> tuple[str, ...]:
    return ("D_state", "V500_norm", "V1000_norm", "A_state", "state_volatility")
