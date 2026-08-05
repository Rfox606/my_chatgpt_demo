from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ContinuousStateV43Config


CONTINUOUS_METRICS = (
    "D_state", "V100_norm", "V500_norm", "V1000_norm", "direction_consistency",
    "A_state", "state_volatility", "baseline_outlier_fraction", "source_support_oos",
    "residual_change_score", "abrupt_cusum", "directed_change_score", "acceleration_score", "abrupt_change_score",
)
EVIDENCE_CONDITIONS = {
    "directed": "directed_change_evidence_condition",
    "rate_divergence": "acceleration_evidence_condition",
    "abrupt": "abrupt_change_evidence_condition",
}


@dataclass(frozen=True)
class ConfigurationRecord:
    config_id: str
    baseline_cycles: int
    distance_form: str
    removed_feature_group: str
    states: pd.DataFrame


def _metric_summary(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(float)
    median = float(np.median(array))
    return {"q50": median, "q25": float(np.quantile(array, .25)), "q75": float(np.quantile(array, .75)),
            "mad": float(np.median(np.abs(array - median)))}


def consensus_trajectories(records: list[ConfigurationRecord], config: ContinuousStateV43Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate post-calibration configurations without treating physical metadata as features."""
    pieces: list[pd.DataFrame] = []
    for record in records:
        state = record.states.loc[record.states.start_cycle_effective > config.consensus_emit_start_cycles].copy()
        state["configuration_id"] = record.config_id
        state["configuration_baseline_cycles"] = record.baseline_cycles
        state["configuration_distance_form"] = record.distance_form
        state["configuration_removed_feature_group"] = record.removed_feature_group
        state["configuration_combined_change_score"] = state.loc[:, ["directed_change_score", "acceleration_score", "abrupt_change_score"]].max(axis=1)
        state["configuration_any_change"] = state.loc[:, list(EVIDENCE_CONDITIONS.values())].max(axis=1)
        pieces.append(state)
    long = pd.concat(pieces, ignore_index=True)
    group_columns = [
        "protocol_id", "dataset", "window_id", "window_index", "start_cycle_effective", "end_cycle_effective", "center_cycle_effective",
        "start_cycle_actual", "end_cycle_actual", "center_cycle_actual", "cycle_effective", "cycle_actual",
    ]
    rows: list[dict[str, object]] = []; support_rows: list[dict[str, object]] = []
    for key, group in long.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, key)); support = dict(zip(group_columns, key))
        row["effective_configuration_count"] = int(group.configuration_id.nunique())
        support["effective_configuration_count"] = int(group.configuration_id.nunique())
        for metric in CONTINUOUS_METRICS:
            summary = _metric_summary(group[metric])
            name = "multi_scale_rate_divergence" if metric == "A_state" else metric
            for statistic, value in summary.items():
                row[f"{name}_{statistic}"] = value
        for name, column in EVIDENCE_CONDITIONS.items():
            value = float(group[column].mean())
            row[f"{name}_configuration_support"] = value
            support[f"{name}_configuration_support"] = value
        row["combined_change_score_q50"] = _metric_summary(group.configuration_combined_change_score)["q50"]
        row["change_configuration_support"] = float(group.configuration_any_change.mean())
        row["guard_configuration_fraction"] = float(group.is_restart_guard.mean())
        row["stop_boundary_configuration_fraction"] = float(group.crosses_stop_boundary.mean())
        support.update({name: row[name] for name in (
            "combined_change_score_q50", "change_configuration_support", "guard_configuration_fraction", "stop_boundary_configuration_fraction",
        )})
        row["change_trigger"] = int(
            row["effective_configuration_count"] > 0
            and row["change_configuration_support"] >= config.consensus_support_min
            and row["combined_change_score_q50"] >= config.consensus_score_min
        )
        support["change_trigger"] = row["change_trigger"]
        rows.append(row); support_rows.append(support)
    return pd.DataFrame(rows), pd.DataFrame(support_rows), long


def _coord(frame: pd.DataFrame) -> pd.Series:
    return frame.center_cycle_actual


def _sustained_mask(values: np.ndarray, cycles: np.ndarray, required: float) -> np.ndarray:
    result = np.zeros(len(values), dtype=bool); start = 0
    while start < len(values):
        if not values[start]:
            start += 1; continue
        end = start + 1
        while end < len(values) and values[end]: end += 1
        if cycles[end - 1] - cycles[start] >= required:
            result[start:end] = True
        start = end
    return result


def _dominant_track(context: pd.DataFrame) -> np.ndarray:
    values = context.loc[:, [f"{name}_configuration_support" for name in EVIDENCE_CONDITIONS]].to_numpy(float)
    return np.asarray(tuple(EVIDENCE_CONDITIONS))[np.argmax(values, axis=1)]


def _split_points(context: pd.DataFrame, config: ContinuousStateV43Config) -> list[float]:
    """Fixed, pre-declared splits: a score valley, sustained support decline, or persistent evidence switch."""
    if len(context) < 3 or float(_coord(context).iloc[-1] - _coord(context).iloc[0]) < 2 * config.episode_split_min_actual_cycles:
        return []
    cycles = _coord(context).to_numpy(float); score = context.combined_change_score_q50.to_numpy(float)
    support = context.change_configuration_support.to_numpy(float); dominant = _dominant_track(context)
    decline = _sustained_mask(support < config.episode_split_support_decline, cycles, config.episode_split_persistence_cycles)
    candidates: list[tuple[float, float]] = []
    for index in range(1, len(context) - 1):
        left_span = cycles[index] - cycles[0]; right_span = cycles[-1] - cycles[index]
        if left_span < config.episode_split_min_actual_cycles or right_span < config.episode_split_min_actual_cycles:
            continue
        local_valley = score[index] <= score[max(0, index - 1):index + 2].min()
        valley_level = score[index] <= config.episode_split_valley_fraction * min(score[:index].max(), score[index + 1:].max())
        switch = dominant[index - 1] != dominant[index]
        left_persistent = _sustained_mask(dominant == dominant[index - 1], cycles, config.episode_split_persistence_cycles)[index - 1]
        right_persistent = _sustained_mask(dominant == dominant[index], cycles, config.episode_split_persistence_cycles)[index]
        if (local_valley and valley_level) or decline[index] or (switch and left_persistent and right_persistent):
            # Smaller score and lower support are preferred when multiple fixed-rule candidates exist.
            candidates.append((cycles[index], score[index] - (1.0 - support[index])))
    candidates.sort(key=lambda item: item[1])
    accepted: list[float] = []
    for cycle, _ in candidates:
        if all(abs(cycle - previous) >= config.episode_split_min_actual_cycles for previous in accepted):
            accepted.append(cycle)
    return sorted(accepted)


def _location_uncertainty(long: pd.DataFrame, start_actual: float, end_actual: float) -> tuple[float, int]:
    portion = long.loc[(long.center_cycle_actual >= start_actual) & (long.center_cycle_actual <= end_actual)]
    peaks = [float(group.loc[group.configuration_combined_change_score.idxmax(), "center_cycle_actual"])
             for _, group in portion.groupby("configuration_id") if not group.empty]
    return (float(np.quantile(peaks, .75) - np.quantile(peaks, .25)) if len(peaks) >= 2 else 0.0, len(peaks))


def _duration_weighted_mean(frame: pd.DataFrame, column: str) -> float:
    weights = np.maximum(frame.end_cycle_actual.to_numpy(float) - frame.start_cycle_actual.to_numpy(float), 1e-9)
    return float(np.average(frame[column].to_numpy(float), weights=weights))


def detect_change_episodes(consensus: pd.DataFrame, long: pd.DataFrame, config: ContinuousStateV43Config) -> pd.DataFrame:
    """Actual-coordinate, support-qualified episodes with fixed deconfounded split rules."""
    rows: list[dict[str, object]] = []; ordinal = 0
    for protocol, whole in consensus.sort_values("center_cycle_actual").groupby("protocol_id"):
        active = whole.loc[whole.change_trigger.eq(1)].copy()
        if active.empty: continue
        starts = [0]; cycles = active.center_cycle_actual.to_numpy(float)
        for index in range(1, len(active)):
            if cycles[index] - cycles[index - 1] > config.episode_merge_gap_cycles: starts.append(index)
        starts.append(len(active))
        for left, right in zip(starts[:-1], starts[1:]):
            base = active.iloc[left:right]
            context = whole.loc[(whole.center_cycle_actual >= base.center_cycle_actual.min()) & (whole.center_cycle_actual <= base.center_cycle_actual.max())].copy()
            cuts = [float(context.center_cycle_actual.min()), *_split_points(context, config), float(context.center_cycle_actual.max())]
            for start_actual, end_actual in zip(cuts[:-1], cuts[1:]):
                segment = context.loc[(context.center_cycle_actual >= start_actual) & (context.center_cycle_actual <= end_actual)].copy()
                if segment.empty or end_actual - start_actual < config.episode_min_cycles: continue
                peak = segment.loc[segment.combined_change_score_q50.idxmax()]
                means = {name: _duration_weighted_mean(segment, f"{name}_configuration_support") for name in EVIDENCE_CONDITIONS}
                total = sum(means.values())
                shares = {name: (means[name] / total if total > 0 else 0.0) for name in EVIDENCE_CONDITIONS}
                uncertainty, n_peaks = _location_uncertainty(long.loc[long.protocol_id.eq(protocol)], float(segment.start_cycle_actual.min()), float(segment.end_cycle_actual.max()))
                weight = np.maximum(segment.end_cycle_actual.to_numpy(float) - segment.start_cycle_actual.to_numpy(float), 1e-9)
                support = segment.change_configuration_support.to_numpy(float)
                ordinal += 1
                rows.append({
                    "episode_id": f"{protocol}:{ordinal:03d}", "protocol_id": protocol, "target_dataset": str(segment.dataset.iloc[0]),
                    "start_cycle_effective": float(segment.start_cycle_effective.min()), "end_cycle_effective": float(segment.end_cycle_effective.max()),
                    "peak_cycle_effective": float(peak.center_cycle_effective), "start_cycle_actual": float(segment.start_cycle_actual.min()),
                    "end_cycle_actual": float(segment.end_cycle_actual.max()), "peak_cycle_actual": float(peak.center_cycle_actual),
                    "peak_change_score": float(peak.combined_change_score_q50), "support_peak": float(support.max()),
                    "support_mean": float(np.average(support, weights=weight)), "support_median": float(np.median(support)), "support_min": float(support.min()),
                    "support_ge_080_duration_fraction": float(weight[support >= .80].sum() / weight.sum()),
                    "configuration_support": float(support.max()), "location_uncertainty_actual": uncertainty, "configuration_peak_count": n_peaks,
                    "directed_evidence_support_mean": means["directed"], "rate_divergence_evidence_support_mean": means["rate_divergence"], "abrupt_evidence_support_mean": means["abrupt"],
                    "directed_evidence_share": shares["directed"], "rate_divergence_evidence_share": shares["rate_divergence"], "abrupt_evidence_share": shares["abrupt"],
                    "evidence_composition": ";".join(f"{name}={shares[name]:.3f}" for name in ("directed", "rate_divergence", "abrupt")),
                    "dominant_evidence": max(shares, key=shares.get),
                    "covers_guard_or_stop_boundary": bool((segment.guard_configuration_fraction > 0).any() or (segment.stop_boundary_configuration_fraction > 0).any()),
                })
    columns = [
        "episode_id", "protocol_id", "target_dataset", "start_cycle_effective", "end_cycle_effective", "peak_cycle_effective", "start_cycle_actual", "end_cycle_actual", "peak_cycle_actual",
        "peak_change_score", "support_peak", "support_mean", "support_median", "support_min", "support_ge_080_duration_fraction", "configuration_support", "location_uncertainty_actual", "configuration_peak_count",
        "directed_evidence_support_mean", "rate_divergence_evidence_support_mean", "abrupt_evidence_support_mean", "directed_evidence_share", "rate_divergence_evidence_share", "abrupt_evidence_share",
        "evidence_composition", "dominant_evidence", "covers_guard_or_stop_boundary",
    ]
    return pd.DataFrame(rows, columns=columns)


def episode_match_jaccard(left: pd.DataFrame, right: pd.DataFrame, tolerance: float = 500.0) -> float:
    if left.empty and right.empty: return 1.0
    column = "peak_cycle_actual" if "peak_cycle_actual" in left.columns else "peak_cycle"
    used = np.zeros(len(right), dtype=bool); matches = 0
    for peak in left[column].to_numpy(float):
        available = np.flatnonzero((~used) & (np.abs(right[column].to_numpy(float) - peak) <= tolerance))
        if len(available):
            nearest = available[np.argmin(np.abs(right[column].to_numpy(float)[available] - peak))]
            used[nearest] = True; matches += 1
    union = len(left) + len(right) - matches
    return matches / union if union else 1.0
