from __future__ import annotations

"""Frozen-baseline state vectors operating on v4.5 raw window features only."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import ContinuousStateV45Config


FORBIDDEN_COLUMNS = frozenset({"stage", "stage_label", "Stage1to5", "Sa", "Sq", "Sz", "Sku", "morphology"})


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
    baseline_cycles: int


def assert_label_free(frame: pd.DataFrame) -> None:
    leaked = sorted(FORBIDDEN_COLUMNS.intersection(frame.columns))
    if leaked:
        raise AssertionError(f"Forbidden labels or morphology reached v4.5 state calculation: {leaked}")


def feature_subset(features: tuple[str, ...], excluded_group: str | None) -> tuple[str, ...]:
    if excluded_group is None:
        return features
    if excluded_group not in {"rx", "ry", "rs"}:
        raise ValueError(f"Unknown feature group: {excluded_group}")
    result = tuple(feature for feature in features if not feature.startswith(f"{excluded_group}_"))
    if not result:
        raise ValueError("Feature ablation removed every group")
    return result


def _groups(features: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    grouped = {group: tuple(feature for feature in features if feature.startswith(f"{group}_")) for group in ("rx", "ry", "rs")}
    return {name: values for name, values in grouped.items() if values}


def robust_location_scale(values: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    location = np.median(values, axis=0)
    mad = np.median(np.abs(values - location), axis=0)
    iqr = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
    return location, np.maximum.reduce((1.4826 * mad, iqr / 1.349, np.full(values.shape[1], eps)))


def _precision(values: np.ndarray, distance_form: str) -> np.ndarray:
    if distance_form == "diagonal":
        return np.eye(values.shape[1])
    if distance_form != "mahalanobis":
        raise ValueError(f"Unsupported distance form {distance_form}")
    return LedoitWolf(assume_centered=False).fit(values).precision_


def _distance(values: np.ndarray, reference: GroupReference, distance_form: str) -> np.ndarray:
    if distance_form == "diagonal":
        return np.sqrt(np.mean(values * values, axis=1))
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", values, reference.precision, values), 0.0) / values.shape[1])


def _starts(cycles: np.ndarray, duration: int) -> np.ndarray:
    return np.searchsorted(cycles, cycles - float(duration), side="left")


def _velocity(values: np.ndarray, cycles: np.ndarray, starts: np.ndarray, eps: float) -> np.ndarray:
    index = np.arange(len(values)); elapsed = np.maximum(cycles - cycles[starts], eps)
    result = (values - values[starts]) / elapsed[:, None] * 100.0
    result[starts == index] = 0.0
    return result


def _rolling_volatility(values: np.ndarray, starts: np.ndarray, ref: GroupReference, distance_form: str) -> np.ndarray:
    total = np.vstack((np.zeros(values.shape[1]), np.cumsum(values, axis=0)))
    square = np.vstack((np.zeros(values.shape[1]), np.cumsum(values * values, axis=0)))
    right = np.arange(len(values)) + 1; count = (right - starts).astype(float)
    mean = (total[right] - total[starts]) / count[:, None]
    variance = np.maximum((square[right] - square[starts]) / count[:, None] - mean * mean, 0.0)
    if distance_form == "diagonal":
        return np.sqrt(np.mean(variance, axis=1))
    return np.sqrt(np.maximum(np.einsum("ij,jj->i", variance, ref.precision), 0.0) / values.shape[1])


def _fuse(values: dict[str, np.ndarray], refs: dict[str, GroupReference], distance_form: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    per_group = {name: _distance(value, refs[name], distance_form) for name, value in values.items()}
    return np.mean(np.vstack(list(per_group.values())), axis=0), per_group


def run_state(frame: pd.DataFrame, state_id: str, features: tuple[str, ...], config: ContinuousStateV45Config, *, include_details: bool = False) -> tuple[pd.DataFrame, BaselineReference]:
    """The only feature standardisation occurs here, from the configuration's frozen early baseline."""
    assert_label_free(frame)
    ordered = frame.sort_values(["center_cycle_effective", "window_index"]).reset_index(drop=True)
    baseline_mask = ordered.end_cycle_effective.to_numpy(float) <= float(config.baseline_cycles)
    if baseline_mask.sum() < 5:
        raise ValueError("Insufficient baseline windows")
    raw = ordered.loc[:, list(features)].to_numpy(float)
    if not np.isfinite(raw).all():
        raise ValueError("v4.5 raw state input must be finite without global future imputation")
    groups: dict[str, GroupReference] = {}; normalized: dict[str, np.ndarray] = {}
    for name, members in _groups(features).items():
        positions = [features.index(feature) for feature in members]
        location, scale = robust_location_scale(raw[baseline_mask][:, positions], config.eps)
        values = (raw[:, positions] - location) / scale
        groups[name] = GroupReference(members, location, scale, _precision(values[baseline_mask], config.distance_form))
        normalized[name] = values
    cycles = ordered.center_cycle_effective.to_numpy(float)
    start_short = _starts(cycles, config.velocity_short_cycles)
    start_long = _starts(cycles, config.velocity_long_cycles)
    start_volatility = _starts(cycles, config.volatility_window_cycles)
    speed_short = {name: _velocity(values, cycles, start_short, config.eps) for name, values in normalized.items()}
    speed_long = {name: _velocity(values, cycles, start_long, config.eps) for name, values in normalized.items()}
    d_state, d_groups = _fuse(normalized, groups, config.distance_form)
    v1000, v1000_groups = _fuse(speed_long, groups, config.distance_form)
    divergence, divergence_groups = _fuse({name: speed_short[name] - speed_long[name] for name in groups}, groups, config.distance_form)
    volatility_groups = {name: _rolling_volatility(values, start_volatility, groups[name], config.distance_form) for name, values in normalized.items()}
    volatility = np.mean(np.vstack(list(volatility_groups.values())), axis=0)
    monitor = ordered.start_cycle_effective.to_numpy(float) > float(config.baseline_cycles)
    selected = np.flatnonzero(monitor); monitored = ordered.loc[monitor].reset_index(drop=True)
    output: dict[str, object] = {
        "state_id": state_id, "dataset": monitored.dataset.to_numpy(), "window_id": monitored.window_id.to_numpy(), "window_index": monitored.window_index.to_numpy(),
        "start_cycle_effective": monitored.start_cycle_effective.to_numpy(float), "end_cycle_effective": monitored.end_cycle_effective.to_numpy(float), "center_cycle_effective": monitored.center_cycle_effective.to_numpy(float),
        "start_cycle_actual": monitored.start_cycle_actual.to_numpy(float), "end_cycle_actual": monitored.end_cycle_actual.to_numpy(float), "center_cycle_actual": monitored.center_cycle_actual.to_numpy(float),
        "cycle_effective": monitored.cycle_effective.to_numpy(float), "cycle_actual": monitored.cycle_actual.to_numpy(float),
        "baseline_cycles": np.full(len(monitored), config.baseline_cycles), "baseline_frozen": np.ones(len(monitored), dtype=int), "distance_form": np.full(len(monitored), config.distance_form),
        "feature_names": np.full(len(monitored), ";".join(features)), "feature_count": np.full(len(monitored), len(features)),
        "D_state": d_state[monitor], "V1000_norm": v1000[monitor], "A_state": divergence[monitor], "state_volatility": volatility[monitor],
    }
    if include_details:
        d_total = np.sum(np.vstack(list(d_groups.values())), axis=0)
        for group, values in d_groups.items():
            output[f"D_{group}_subspace"] = values[monitor]
            output[f"D_{group}_contribution"] = np.divide(values[monitor], d_total[monitor], out=np.zeros(len(selected)), where=d_total[monitor] > config.eps)
        for label, values_by_group in (("velocity_v1000", v1000_groups), ("rate_divergence", divergence_groups)):
            total = np.sum(np.vstack(list(values_by_group.values())), axis=0)
            for group, values in values_by_group.items():
                output[f"{label}_{group}_subspace_distance"] = values[monitor]
                output[f"{label}_{group}_contribution"] = np.divide(values[monitor], total[monitor], out=np.zeros(len(selected)), where=total[monitor] > config.eps)
    return pd.DataFrame(output), BaselineReference(features, groups, int(baseline_mask.sum()), config.distance_form, config.baseline_cycles)
