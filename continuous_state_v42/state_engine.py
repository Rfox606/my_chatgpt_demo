from __future__ import annotations

"""Robust causal state scores with a calibration-only target baseline."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import FEATURE_GROUPS, ContinuousStateV42Config
from .data import assert_label_free, baseline_mask, robust_location_scale


EVIDENCE_NAMES = (
    "low_activity_evidence",
    "directed_change_evidence",
    "acceleration_evidence",
    "abrupt_change_evidence",
)


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
    distance_form: str
    baseline_count: int
    thresholds: dict[str, float]


@dataclass(frozen=True)
class SourceSupport:
    features: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray


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


def fit_source_support(source: pd.DataFrame, features: tuple[str, ...]) -> SourceSupport:
    """A source-only feature support envelope; labels are never read."""
    assert_label_free(source)
    values = source.loc[:, list(features)].to_numpy(float)
    return SourceSupport(features, np.quantile(values, .005, axis=0), np.quantile(values, .995, axis=0))


def _group_features(features: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    result = {group: tuple(feature for feature in members if feature in features) for group, members in FEATURE_GROUPS.items()}
    return {group: members for group, members in result.items() if members}


def _precision(values: np.ndarray, distance_form: str) -> np.ndarray:
    width = values.shape[1]
    if distance_form == "diagonal":
        return np.eye(width)
    if distance_form != "mahalanobis":
        raise ValueError(f"Unsupported distance form: {distance_form}")
    return LedoitWolf(assume_centered=False).fit(values).precision_


def _distance(values: np.ndarray, precision: np.ndarray, distance_form: str) -> np.ndarray:
    if distance_form == "diagonal":
        return np.sqrt(np.mean(values * values, axis=1))
    width = values.shape[1]
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", values, precision, values), 0.0) / width)


def _fused_distance(values: dict[str, np.ndarray], refs: dict[str, GroupReference], distance_form: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    per_group = {group: _distance(values[group], refs[group].precision, distance_form) for group in refs}
    fused = np.mean(np.vstack(list(per_group.values())), axis=0)
    return fused, per_group


def _starts(cycles: np.ndarray, periods: tuple[int, ...]) -> dict[int, np.ndarray]:
    return {period: np.searchsorted(cycles, cycles - float(period), side="left") for period in periods}


def _velocity(values: np.ndarray, cycles: np.ndarray, starts: np.ndarray, eps: float) -> np.ndarray:
    index = np.arange(len(values))
    elapsed = np.maximum(cycles - cycles[starts], eps)
    vector = (values - values[starts]) / elapsed[:, None] * 100.0
    vector[starts == index] = 0.0
    return vector


def _rolling_group_volatility(values: np.ndarray, starts: np.ndarray, ref: GroupReference, distance_form: str) -> np.ndarray:
    total = np.vstack([np.zeros(values.shape[1]), np.cumsum(values, axis=0)])
    squares = np.vstack([np.zeros(values.shape[1]), np.cumsum(values * values, axis=0)])
    right = np.arange(len(values)) + 1
    count = (right - starts).astype(float)
    mean = (total[right] - total[starts]) / count[:, None]
    variance = np.maximum((squares[right] - squares[starts]) / count[:, None] - mean * mean, 0.0)
    if distance_form == "diagonal":
        return np.sqrt(np.mean(variance, axis=1))
    diagonal = np.einsum("ij,jj->i", variance, ref.precision)
    return np.sqrt(np.maximum(diagonal, 0.0) / values.shape[1])


def _mean_group_cosine(left: dict[str, np.ndarray], right: dict[str, np.ndarray], eps: float) -> np.ndarray:
    scores: list[np.ndarray] = []
    for group in left:
        numerator = np.sum(left[group] * right[group], axis=1)
        denominator = np.linalg.norm(left[group], axis=1) * np.linalg.norm(right[group], axis=1)
        scores.append(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > eps))
    return np.mean(np.vstack(scores), axis=0)


def _online_residual_score(values: dict[str, np.ndarray], refs: dict[str, GroupReference], frame: pd.DataFrame, config: ContinuousStateV42Config) -> np.ndarray:
    """Residual is from a one-step causal EWMA, independent of acceleration."""
    n = len(frame)
    means = {group: np.zeros(values[group].shape[1]) for group in values}
    score = np.zeros(n)
    guard = frame.is_restart_guard.to_numpy(bool)
    for index in range(n):
        residual = {group: values[group][index:index + 1] - means[group] for group in values}
        score[index] = float(np.mean([_distance(residual[group], refs[group].precision, config.distance_form)[0] for group in residual]))
        if not guard[index]:
            for group in means:
                means[group] = (1.0 - config.residual_ewma_alpha) * means[group] + config.residual_ewma_alpha * values[group][index]
    return score


def _thresholds(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    v100: np.ndarray,
    v500: np.ndarray,
    volatility: np.ndarray,
    direction: np.ndarray,
    acceleration: np.ndarray,
    residual: np.ndarray,
) -> dict[str, float]:
    usable = baseline & frame.is_restart_guard.eq(0).to_numpy(bool)
    if usable.sum() < 5:
        raise ValueError("Frozen target baseline has fewer than five valid windows")
    def q(values: np.ndarray, level: float) -> float:
        return float(np.quantile(values[usable], level))
    center = float(np.median(residual[usable]))
    scale = max(float(np.median(np.abs(residual[usable] - center)) * 1.4826), 1e-6)
    return {
        "low_velocity_max": q(v100, .80), "low_volatility_max": q(volatility, .80),
        "directed_speed_min": q(v500, .75), "direction_consistency_min": max(.60, q(direction, .75)),
        "acceleration_min": q(acceleration, .95), "residual_score_min": q(residual, .99),
        "residual_center": center, "residual_scale": scale,
    }


def _run_evidence_tracks(
    frame: pd.DataFrame,
    conditions: dict[str, np.ndarray],
    residual_score: np.ndarray,
    thresholds: dict[str, float],
    continuous_scores: dict[str, np.ndarray],
    config: ContinuousStateV42Config,
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
        values[name.replace("_evidence", "_score")] = continuous_scores[name]
    cusum = np.zeros(n)
    runs = {name: 0.0 for name in EVIDENCE_NAMES}; false_runs = {name: 0.0 for name in EVIDENCE_NAMES}; active = {name: False for name in EVIDENCE_NAMES}
    running_cusum = 0.0; events: list[dict[str, object]] = []
    centers = frame.center_cycle.to_numpy(float)
    for index in range(n):
        if guard[index]:
            for name in EVIDENCE_NAMES:
                values[f"{name}_run_cycles"][index] = runs[name]; values[f"{name}_false_cycles"][index] = false_runs[name]; values[name][index] = int(active[name])
            cusum[index] = running_cusum
            continue
        z = (residual_score[index] - thresholds["residual_center"]) / thresholds["residual_scale"]
        running_cusum = min(100.0, max(0.0, running_cusum + z - 1.0))
        cusum[index] = running_cusum
        conditions["abrupt_change_evidence"][index] = bool(residual_score[index] >= thresholds["residual_score_min"] and running_cusum >= config.abrupt_cusum_threshold)
        values["abrupt_change_evidence_condition"][index] = int(conditions["abrupt_change_evidence"][index])
        values["abrupt_change_score"][index] = min(residual_score[index] / max(thresholds["residual_score_min"], config.eps), running_cusum / config.abrupt_cusum_threshold)
        for name in EVIDENCE_NAMES:
            confirm = config.low_activity_confirm_cycles if name == "low_activity_evidence" else config.evidence_confirm_cycles
            release = config.low_activity_release_cycles if name == "low_activity_evidence" else config.evidence_reset_cycles
            if conditions[name][index]:
                runs[name] += stride; false_runs[name] = 0.0
            else:
                false_runs[name] += stride
                if false_runs[name] >= release:
                    runs[name] = 0.0
                    if active[name]:
                        active[name] = False
                        events.append({"protocol_id": protocol_id, "target_dataset": str(frame.dataset.iloc[0]), "event": "algorithm_evidence_offset", "evidence_type": name, "cycle": float(centers[index]), "run_cycles": 0.0})
            if not active[name] and runs[name] >= confirm:
                active[name] = True
                events.append({"protocol_id": protocol_id, "target_dataset": str(frame.dataset.iloc[0]), "event": "algorithm_evidence_onset", "evidence_type": name, "cycle": float(centers[index]), "run_cycles": float(runs[name])})
            values[f"{name}_run_cycles"][index] = runs[name]; values[f"{name}_false_cycles"][index] = false_runs[name]; values[name][index] = int(active[name])
    return values, pd.DataFrame(events, columns=["protocol_id", "target_dataset", "event", "evidence_type", "cycle", "run_cycles"]), cusum


def run_target_state(
    frame: pd.DataFrame,
    protocol_id: str,
    features: tuple[str, ...],
    config: ContinuousStateV42Config,
    source_support: SourceSupport | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, BaselineReference]:
    """Freeze 0--baseline calibration, then emit only post-baseline causal rows."""
    assert_label_free(frame)
    ordered = frame.sort_values(["center_cycle", "window_index"]).reset_index(drop=True).copy()
    baseline = baseline_mask(ordered, config)
    raw = ordered.loc[:, list(features)].to_numpy(float)
    group_members = _group_features(features)
    refs: dict[str, GroupReference] = {}; normalized: dict[str, np.ndarray] = {}
    for group, members in group_members.items():
        positions = [features.index(feature) for feature in members]
        location, scale = robust_location_scale(raw[baseline][:, positions], config.eps)
        values = (raw[:, positions] - location) / scale
        refs[group] = GroupReference(members, location, scale, _precision(values[baseline], config.distance_form))
        normalized[group] = values
    cycles = ordered.center_cycle.to_numpy(float)
    starts = _starts(cycles, tuple(set((*config.velocity_windows_cycles, config.volatility_window_cycles))))
    vectors = {period: {group: _velocity(values, cycles, starts[period], config.eps) for group, values in normalized.items()} for period in config.velocity_windows_cycles}
    d_state, d_group = _fused_distance(normalized, refs, config.distance_form)
    v100, v100_group = _fused_distance(vectors[100], refs, config.distance_form)
    v500, v500_group = _fused_distance(vectors[500], refs, config.distance_form)
    v1000, v1000_group = _fused_distance(vectors[1000], refs, config.distance_form)
    direction = (_mean_group_cosine(vectors[100], vectors[500], config.eps) + _mean_group_cosine(vectors[100], vectors[1000], config.eps) + _mean_group_cosine(vectors[500], vectors[1000], config.eps)) / 3.0
    acceleration, acceleration_group = _fused_distance({group: vectors[100][group] - vectors[1000][group] for group in refs}, refs, config.distance_form)
    volatility_parts = {group: _rolling_group_volatility(normalized[group], starts[config.volatility_window_cycles], refs[group], config.distance_form) for group in refs}
    volatility = np.mean(np.vstack(list(volatility_parts.values())), axis=0)
    baseline_outlier = np.mean(np.hstack([np.abs(normalized[group]) > 3.0 for group in normalized]), axis=1)
    if source_support is None:
        source_oos = np.full(len(ordered), np.nan)
    else:
        source_oos = np.mean((raw < source_support.lower) | (raw > source_support.upper), axis=1)
    residual = _online_residual_score(normalized, refs, ordered, config)
    thresholds = _thresholds(ordered, baseline, v100, v500, volatility, direction, acceleration, residual)
    # Calibration-only rows are never emitted as state, evidence, or forecast output.
    monitor = ordered.start_cycle.to_numpy(float) > float(config.baseline_cycles)
    monitor_frame = ordered.loc[monitor].reset_index(drop=True)
    conditions = {
        "low_activity_evidence": ((v100 <= thresholds["low_velocity_max"]) & (volatility <= thresholds["low_volatility_max"]))[monitor],
        "directed_change_evidence": ((direction >= thresholds["direction_consistency_min"]) & (v500 >= thresholds["directed_speed_min"]))[monitor],
        "acceleration_evidence": ((acceleration >= thresholds["acceleration_min"]) & (v100 > v1000))[monitor],
        "abrupt_change_evidence": np.zeros(int(monitor.sum()), dtype=bool),
    }
    continuous = {
        "low_activity_evidence": (1.0 - np.maximum(v100 / max(thresholds["low_velocity_max"], config.eps), volatility / max(thresholds["low_volatility_max"], config.eps)))[monitor],
        "directed_change_evidence": np.minimum(direction / max(thresholds["direction_consistency_min"], config.eps), v500 / max(thresholds["directed_speed_min"], config.eps))[monitor],
        "acceleration_evidence": (acceleration / max(thresholds["acceleration_min"], config.eps) * v100 / np.maximum(v1000, config.eps))[monitor],
        "abrupt_change_evidence": np.zeros(int(monitor.sum())),
    }
    evidence, events, cusum = _run_evidence_tracks(monitor_frame, conditions, residual[monitor], thresholds, continuous, config, protocol_id)
    selected = np.flatnonzero(monitor)
    stride = float(np.median(np.diff(cycles))) if len(cycles) > 1 else 1.0
    output: dict[str, object] = {
        "protocol_id": protocol_id, "dataset": monitor_frame.dataset.to_numpy(), "window_id": monitor_frame.window_id.to_numpy(), "window_index": monitor_frame.window_index.to_numpy(),
        "start_cycle": monitor_frame.start_cycle.to_numpy(float), "end_cycle": monitor_frame.end_cycle.to_numpy(float), "center_cycle": monitor_frame.center_cycle.to_numpy(float),
        "is_restart_guard": monitor_frame.is_restart_guard.to_numpy(int), "crosses_stop_boundary": monitor_frame.crosses_stop_boundary.to_numpy(int),
        "nominal_stride_cycles": np.full(len(monitor_frame), stride), "baseline_frozen": np.ones(len(monitor_frame), dtype=int),
        "baseline_cycles": np.full(len(monitor_frame), config.baseline_cycles), "monitoring_start_cycle": np.full(len(monitor_frame), float(monitor_frame.start_cycle.min()) if len(monitor_frame) else np.nan),
        "distance_form": np.full(len(monitor_frame), config.distance_form), "feature_count": np.full(len(monitor_frame), len(features)), "feature_names": np.full(len(monitor_frame), ";".join(features)),
        "D_state": d_state[monitor], "V100_norm": v100[monitor], "V500_norm": v500[monitor], "V1000_norm": v1000[monitor],
        "direction_consistency": direction[monitor], "A_state": acceleration[monitor], "state_volatility": volatility[monitor],
        "baseline_outlier_fraction": baseline_outlier[monitor], "source_support_oos": source_oos[monitor], "residual_change_score": residual[monitor], "abrupt_cusum": cusum,
    }
    for group, values in d_group.items(): output[f"D_{group}_subspace"] = values[monitor]
    for period, label, per_group in ((100, "velocity_v100", v100_group), (500, "velocity_v500", v500_group), (1000, "velocity_v1000", v1000_group)):
        group_total = np.sum(np.vstack(list(per_group.values())), axis=0)
        for group, vector in vectors[period].items():
            for feature_index, feature in enumerate(refs[group].features): output[f"{label}_{feature}"] = vector[selected, feature_index]
            output[f"{label}_{group}_subspace_distance"] = per_group[group][monitor]
            output[f"{label}_{group}_contribution"] = np.divide(per_group[group][monitor], group_total[monitor], out=np.zeros(len(selected)), where=group_total[monitor] > config.eps)
    output.update(evidence)
    for key, value in thresholds.items(): output[f"frozen_{key}"] = np.full(len(monitor_frame), value)
    states = pd.DataFrame(output)
    assert_label_free(states)
    return states, events, BaselineReference(features, refs, config.distance_form, int(baseline.sum()), thresholds)


def main_state_columns() -> tuple[str, ...]:
    return ("D_state", "V100_norm", "V500_norm", "V1000_norm", "direction_consistency", "A_state", "state_volatility", "baseline_outlier_fraction", "source_support_oos", "residual_change_score", "abrupt_cusum", *(name.replace("_evidence", "_score") for name in EVIDENCE_NAMES), *EVIDENCE_NAMES, *(f"{name}_run_cycles" for name in EVIDENCE_NAMES))
