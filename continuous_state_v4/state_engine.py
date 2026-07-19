from __future__ import annotations

"""Causal, path-free state scores and independent algorithm-evidence tracks."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FEATURE_GROUPS, ContinuousStateV4Config
from .data import assert_label_free, baseline_mask, robust_location_scale


EVIDENCE_NAMES = (
    "low_activity_evidence",
    "directed_change_evidence",
    "acceleration_evidence",
    "abrupt_change_evidence",
)


@dataclass(frozen=True)
class BaselineReference:
    features: tuple[str, ...]
    location: np.ndarray
    scale: np.ndarray
    precision: np.ndarray
    distance_form: str
    baseline_count: int
    thresholds: dict[str, float]


def feature_subset(features: tuple[str, ...], excluded_group: str | None = None) -> tuple[str, ...]:
    if excluded_group is None:
        return features
    if excluded_group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {excluded_group}")
    blocked = set(FEATURE_GROUPS[excluded_group])
    kept = tuple(feature for feature in features if feature not in blocked)
    if not kept:
        raise ValueError("Feature-group ablation removed every candidate feature")
    return kept


def _precision(normalized_baseline: np.ndarray, distance_form: str, ridge: float) -> np.ndarray:
    width = normalized_baseline.shape[1]
    if distance_form == "diagonal":
        return np.eye(width)
    if distance_form != "mahalanobis":
        raise ValueError(f"Unsupported distance form: {distance_form}")
    covariance = np.cov(normalized_baseline, rowvar=False)
    covariance = np.atleast_2d(covariance)
    scale = float(np.trace(covariance) / max(width, 1))
    covariance += np.eye(width) * max(ridge, ridge * scale)
    return np.linalg.pinv(covariance, hermitian=True)


def _distance(values: np.ndarray, precision: np.ndarray, distance_form: str) -> np.ndarray:
    if distance_form == "diagonal":
        return np.sqrt(np.mean(values * values, axis=1))
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", values, precision, values), 0.0) / values.shape[1])


def _start_indices(cycles: np.ndarray, periods: tuple[int, ...]) -> dict[int, np.ndarray]:
    return {period: np.searchsorted(cycles, cycles - float(period), side="left") for period in periods}


def _velocity_vectors(values: np.ndarray, cycles: np.ndarray, starts: np.ndarray, eps: float) -> np.ndarray:
    index = np.arange(len(values))
    elapsed = np.maximum(cycles - cycles[starts], eps)
    velocity = (values - values[starts]) / elapsed[:, None] * 100.0
    velocity[starts == index] = 0.0
    return velocity


def _rolling_volatility(values: np.ndarray, starts: np.ndarray, distance_form: str, precision: np.ndarray) -> np.ndarray:
    """Causal feature-space variability inside a time-defined window."""
    total = np.vstack([np.zeros(values.shape[1]), np.cumsum(values, axis=0)])
    total_sq = np.vstack([np.zeros(values.shape[1]), np.cumsum(values * values, axis=0)])
    right = np.arange(len(values)) + 1
    count = (right - starts).astype(float)
    mean = (total[right] - total[starts]) / count[:, None]
    variance = np.maximum((total_sq[right] - total_sq[starts]) / count[:, None] - mean * mean, 0.0)
    if distance_form == "diagonal":
        return np.sqrt(np.mean(variance, axis=1))
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", variance, precision, np.ones_like(variance)), 0.0) / values.shape[1])


def _cosine(left: np.ndarray, right: np.ndarray, eps: float) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > eps)


def _channel_contributions(vector: np.ndarray, features: tuple[str, ...], suffix: str) -> dict[str, np.ndarray]:
    denominator = np.sum(vector * vector, axis=1)
    result: dict[str, np.ndarray] = {}
    for group, members in FEATURE_GROUPS.items():
        positions = [features.index(member) for member in members if member in features]
        numerator = np.sum(vector[:, positions] ** 2, axis=1) if positions else np.zeros(len(vector))
        result[f"{suffix}_{group}_contribution"] = np.divide(numerator, denominator, out=np.zeros(len(vector)), where=denominator > 1e-12)
    return result


def _frozen_thresholds(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    v20: np.ndarray,
    v50: np.ndarray,
    v100: np.ndarray,
    volatility: np.ndarray,
    direction: np.ndarray,
    acceleration: np.ndarray,
    abrupt_score: np.ndarray,
) -> dict[str, float]:
    usable = baseline & frame.is_restart_guard.eq(0).to_numpy(bool)
    if usable.sum() < 5:
        raise ValueError("Frozen baseline has fewer than five valid windows")
    def q(values: np.ndarray, level: float) -> float:
        return float(np.quantile(values[usable], level))
    return {
        "low_velocity_max": q(v20, .80),
        "low_volatility_max": q(volatility, .80),
        "directed_speed_min": q(v50, .75),
        "direction_consistency_min": max(.60, q(direction, .75)),
        "acceleration_min": q(acceleration, .95),
        "abrupt_score_min": q(abrupt_score, .99),
        "abrupt_baseline_center": float(np.median(abrupt_score[usable])),
        "abrupt_baseline_scale": max(float(np.median(np.abs(abrupt_score[usable] - np.median(abrupt_score[usable]))) * 1.4826), 1e-6),
    }


def _run_evidence_tracks(
    frame: pd.DataFrame,
    conditions: dict[str, np.ndarray],
    abrupt_score: np.ndarray,
    thresholds: dict[str, float],
    config: ContinuousStateV4Config,
    protocol_id: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
    n = len(frame)
    stride = float(np.median(np.diff(frame.center_cycle.to_numpy(float)))) if n > 1 else 1.0
    guard = frame.is_restart_guard.to_numpy(bool)
    values: dict[str, np.ndarray] = {"evidence_increment_cycles": np.where(guard, 0.0, stride)}
    for name in EVIDENCE_NAMES:
        values[f"{name}_run_cycles"] = np.zeros(n)
        values[f"{name}_false_cycles"] = np.zeros(n)
        values[name] = np.zeros(n, dtype=int)
        values[f"{name}_condition"] = conditions[name].astype(int)
    cusum = np.zeros(n)
    events: list[dict[str, object]] = []
    runs = {name: 0.0 for name in EVIDENCE_NAMES}
    false_runs = {name: 0.0 for name in EVIDENCE_NAMES}
    active = {name: False for name in EVIDENCE_NAMES}
    running_cusum = 0.0
    centers = frame.center_cycle.to_numpy(float)
    for index in range(n):
        if guard[index]:
            for name in EVIDENCE_NAMES:
                values[f"{name}_run_cycles"][index] = runs[name]
                values[f"{name}_false_cycles"][index] = false_runs[name]
                values[name][index] = int(active[name])
            cusum[index] = running_cusum
            continue
        z = (abrupt_score[index] - thresholds["abrupt_baseline_center"]) / thresholds["abrupt_baseline_scale"]
        running_cusum = max(0.0, running_cusum + z - 1.0)
        cusum[index] = running_cusum
        # The change-point track is online: only its historical accumulator is used.
        conditions["abrupt_change_evidence"][index] = bool(
            abrupt_score[index] >= thresholds["abrupt_score_min"] and running_cusum >= 5.0
        )
        values["abrupt_change_evidence_condition"][index] = int(conditions["abrupt_change_evidence"][index])
        for name in EVIDENCE_NAMES:
            if conditions[name][index]:
                runs[name] += stride
                false_runs[name] = 0.0
            else:
                false_runs[name] += stride
                if false_runs[name] >= config.evidence_reset_cycles:
                    runs[name] = 0.0
                    if active[name]:
                        active[name] = False
                        events.append({"protocol_id": protocol_id, "target_dataset": str(frame.dataset.iloc[0]),
                                       "event": "algorithm_evidence_offset", "evidence_type": name,
                                       "cycle": float(centers[index]), "run_cycles": 0.0})
            if not active[name] and runs[name] >= config.evidence_confirm_cycles:
                active[name] = True
                events.append({"protocol_id": protocol_id, "target_dataset": str(frame.dataset.iloc[0]),
                               "event": "algorithm_evidence_onset", "evidence_type": name,
                               "cycle": float(centers[index]), "run_cycles": float(runs[name])})
            values[f"{name}_run_cycles"][index] = runs[name]
            values[f"{name}_false_cycles"][index] = false_runs[name]
            values[name][index] = int(active[name])
    return values, pd.DataFrame(events, columns=["protocol_id", "target_dataset", "event", "evidence_type", "cycle", "run_cycles"]), cusum


def run_target_state(
    frame: pd.DataFrame,
    protocol_id: str,
    features: tuple[str, ...],
    config: ContinuousStateV4Config,
) -> tuple[pd.DataFrame, pd.DataFrame, BaselineReference]:
    """Score a target stream without observing target data beyond each row."""
    assert_label_free(frame)
    ordered = frame.sort_values(["center_cycle", "window_index"]).reset_index(drop=True).copy()
    baseline = baseline_mask(ordered, config)
    raw = ordered.loc[:, list(features)].to_numpy(float)
    location, scale = robust_location_scale(raw[baseline], config.eps)
    normalized = (raw - location) / scale
    precision = _precision(normalized[baseline], config.distance_form, config.covariance_ridge)
    cycles = ordered.center_cycle.to_numpy(float)
    starts = _start_indices(cycles, tuple(set((*config.velocity_windows_cycles, config.volatility_window_cycles))))
    v20 = _velocity_vectors(normalized, cycles, starts[20], config.eps)
    v50 = _velocity_vectors(normalized, cycles, starts[50], config.eps)
    v100 = _velocity_vectors(normalized, cycles, starts[100], config.eps)
    d_state = _distance(normalized, precision, config.distance_form)
    v20_norm = _distance(v20, precision, config.distance_form)
    v50_norm = _distance(v50, precision, config.distance_form)
    v100_norm = _distance(v100, precision, config.distance_form)
    direction = (_cosine(v20, v50, config.eps) + _cosine(v20, v100, config.eps) + _cosine(v50, v100, config.eps)) / 3.0
    acceleration = _distance(v20 - v100, precision, config.distance_form)
    volatility = _rolling_volatility(normalized, starts[config.volatility_window_cycles], config.distance_form, precision)
    weighted_oos = np.mean(np.abs(normalized) > 3.0, axis=1)
    abrupt_score = acceleration + np.maximum(v20_norm - v100_norm, 0.0)
    thresholds = _frozen_thresholds(ordered, baseline, v20_norm, v50_norm, v100_norm, volatility, direction, acceleration, abrupt_score)
    conditions = {
        "low_activity_evidence": (v20_norm <= thresholds["low_velocity_max"]) & (volatility <= thresholds["low_volatility_max"]),
        "directed_change_evidence": (direction >= thresholds["direction_consistency_min"]) & (v50_norm >= thresholds["directed_speed_min"]),
        "acceleration_evidence": (acceleration >= thresholds["acceleration_min"]) & (v20_norm > v100_norm),
        "abrupt_change_evidence": np.zeros(len(ordered), dtype=bool),
    }
    evidence, events, cusum = _run_evidence_tracks(ordered, conditions, abrupt_score, thresholds, config, protocol_id)
    stride = float(np.median(np.diff(cycles))) if len(cycles) > 1 else 1.0
    output: dict[str, object] = {
        "protocol_id": protocol_id, "dataset": ordered.dataset.to_numpy(), "window_id": ordered.window_id.to_numpy(),
        "window_index": ordered.window_index.to_numpy(), "center_cycle": cycles,
        "is_restart_guard": ordered.is_restart_guard.to_numpy(int), "crosses_stop_boundary": ordered.crosses_stop_boundary.to_numpy(int),
        "nominal_stride_cycles": np.full(len(ordered), stride), "baseline_frozen": baseline.astype(int),
        "baseline_cycles": np.full(len(ordered), config.baseline_cycles), "distance_form": np.full(len(ordered), config.distance_form),
        "feature_count": np.full(len(ordered), len(features)), "feature_names": np.full(len(ordered), ";".join(features)),
        "D_state": d_state, "V20_norm": v20_norm, "V50_norm": v50_norm, "V100_norm": v100_norm,
        "direction_consistency": direction, "A_state": acceleration, "state_volatility": volatility,
        "weighted_oos": weighted_oos, "abrupt_score": abrupt_score, "abrupt_cusum": cusum,
    }
    for label, vector in (("velocity_v20", v20), ("velocity_v50", v50), ("velocity_v100", v100)):
        output.update({f"{label}_{feature}": vector[:, index] for index, feature in enumerate(features)})
        output.update(_channel_contributions(vector, features, label))
    output.update(evidence)
    for key, value in thresholds.items():
        output[f"frozen_{key}"] = np.full(len(ordered), value)
    states = pd.DataFrame(output)
    assert_label_free(states)
    reference = BaselineReference(features, location, scale, precision, config.distance_form, int(baseline.sum()), thresholds)
    return states, events, reference


def main_state_columns() -> tuple[str, ...]:
    return (
        "D_state", "V20_norm", "V50_norm", "V100_norm", "direction_consistency", "A_state", "state_volatility",
        "weighted_oos", "abrupt_score", "abrupt_cusum", *EVIDENCE_NAMES,
        *(f"{name}_run_cycles" for name in EVIDENCE_NAMES),
    )
