from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ContinuousStateV42Config


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


def consensus_trajectories(records: list[ConfigurationRecord], config: ContinuousStateV42Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate only configuration rows that have finished their own calibration."""
    pieces: list[pd.DataFrame] = []
    for record in records:
        state = record.states.loc[record.states.start_cycle > config.consensus_emit_start_cycles].copy()
        state["configuration_id"] = record.config_id
        state["configuration_baseline_cycles"] = record.baseline_cycles
        state["configuration_distance_form"] = record.distance_form
        state["configuration_removed_feature_group"] = record.removed_feature_group
        state["configuration_combined_change_score"] = state.loc[:, ["directed_change_score", "acceleration_score", "abrupt_change_score"]].max(axis=1)
        state["configuration_any_change"] = state.loc[:, list(EVIDENCE_CONDITIONS.values())].max(axis=1)
        pieces.append(state)
    long = pd.concat(pieces, ignore_index=True)
    group_columns = ["protocol_id", "dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle"]
    rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    for key, group in long.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, key)); support = dict(zip(group_columns, key))
        row["effective_configuration_count"] = int(group.configuration_id.nunique())
        support["effective_configuration_count"] = int(group.configuration_id.nunique())
        for metric in CONTINUOUS_METRICS:
            summary = _metric_summary(group[metric])
            name = "multi_scale_rate_divergence" if metric == "A_state" else metric
            for statistic, value in summary.items(): row[f"{name}_{statistic}"] = value
        for name, column in EVIDENCE_CONDITIONS.items():
            value = float(group[column].mean())
            row[f"{name}_configuration_support"] = value
            support[f"{name}_configuration_support"] = value
        row["combined_change_score_q50"] = _metric_summary(group.configuration_combined_change_score)["q50"]
        row["change_configuration_support"] = float(group.configuration_any_change.mean())
        row["guard_configuration_fraction"] = float(group.is_restart_guard.mean())
        row["stop_boundary_configuration_fraction"] = float(group.crosses_stop_boundary.mean())
        support.update({key: row[key] for key in ("combined_change_score_q50", "change_configuration_support", "guard_configuration_fraction", "stop_boundary_configuration_fraction")})
        row["change_trigger"] = int(row["effective_configuration_count"] > 0 and row["change_configuration_support"] >= config.consensus_support_min and row["combined_change_score_q50"] >= config.consensus_score_min)
        support["change_trigger"] = row["change_trigger"]
        rows.append(row); support_rows.append(support)
    return pd.DataFrame(rows), pd.DataFrame(support_rows), long


def _episode_location_uncertainty(long: pd.DataFrame, start: float, end: float) -> tuple[float, int]:
    portion = long.loc[(long.center_cycle >= start) & (long.center_cycle <= end)]
    peaks: list[float] = []
    for _, group in portion.groupby("configuration_id"):
        if group.empty:
            continue
        peaks.append(float(group.loc[group.configuration_combined_change_score.idxmax(), "center_cycle"]))
    return (float(np.quantile(peaks, .75) - np.quantile(peaks, .25)) if len(peaks) >= 2 else 0.0, len(peaks))


def detect_change_episodes(consensus: pd.DataFrame, long: pd.DataFrame, config: ContinuousStateV42Config) -> pd.DataFrame:
    """Merge nearby support-qualified continuous changes into label-free intervals."""
    rows: list[dict[str, object]] = []
    for protocol, group in consensus.sort_values("center_cycle").groupby("protocol_id"):
        active = group.loc[group.change_trigger.eq(1)].copy()
        if active.empty:
            continue
        segments: list[pd.DataFrame] = []
        start_index = 0
        cycle = active.center_cycle.to_numpy(float)
        for index in range(1, len(active)):
            if cycle[index] - cycle[index - 1] > config.episode_merge_gap_cycles:
                segments.append(active.iloc[start_index:index]); start_index = index
        segments.append(active.iloc[start_index:])
        for segment in segments:
            start = float(segment.center_cycle.min()); end = float(segment.center_cycle.max())
            if end - start + float(segment.nominal_stride_cycles.iloc[0] if "nominal_stride_cycles" in segment else 5.0) < config.episode_min_cycles:
                continue
            peak_row = segment.loc[segment.combined_change_score_q50.idxmax()]
            composition = {name: float(segment[f"{name}_configuration_support"].mean()) for name in EVIDENCE_CONDITIONS}
            dominant = max(composition, key=composition.get)
            uncertainty, configuration_peaks = _episode_location_uncertainty(long.loc[long.protocol_id.eq(protocol)], start, end)
            rows.append({"protocol_id": protocol, "target_dataset": str(segment.dataset.iloc[0]), "start_cycle": start, "end_cycle": end,
                         "peak_cycle": float(peak_row.center_cycle), "peak_change_score": float(peak_row.combined_change_score_q50),
                         "configuration_support": float(segment.change_configuration_support.max()), "location_uncertainty": uncertainty,
                         "configuration_peak_count": configuration_peaks,
                         "evidence_composition": ";".join(f"{name}={composition[name]:.3f}" for name in ("directed", "rate_divergence", "abrupt")),
                         "directed_composition": composition["directed"], "rate_divergence_composition": composition["rate_divergence"], "abrupt_composition": composition["abrupt"],
                         "dominant_evidence": dominant,
                         "covers_guard_or_stop_boundary": bool((segment.guard_configuration_fraction > 0).any() or (segment.stop_boundary_configuration_fraction > 0).any())})
    return pd.DataFrame(rows, columns=["protocol_id", "target_dataset", "start_cycle", "end_cycle", "peak_cycle", "peak_change_score", "configuration_support", "location_uncertainty", "configuration_peak_count", "evidence_composition", "directed_composition", "rate_divergence_composition", "abrupt_composition", "dominant_evidence", "covers_guard_or_stop_boundary"])


def episode_match_jaccard(left: pd.DataFrame, right: pd.DataFrame, tolerance: float = 500.0) -> float:
    if left.empty and right.empty:
        return 1.0
    used = np.zeros(len(right), dtype=bool); matches = 0
    for peak in left.peak_cycle.to_numpy(float):
        available = np.flatnonzero((~used) & (np.abs(right.peak_cycle.to_numpy(float) - peak) <= tolerance))
        if len(available):
            nearest = available[np.argmin(np.abs(right.peak_cycle.to_numpy(float)[available] - peak))]
            used[nearest] = True; matches += 1
    union = len(left) + len(right) - matches
    return matches / union if union else 1.0
