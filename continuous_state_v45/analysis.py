from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .config import METRIC_OUTPUT_NAMES, STATE_METRICS, ContinuousStateV45Config


@dataclass(frozen=True)
class ConfigurationRecord:
    config_id: str
    baseline_cycles: int
    distance_form: str
    feature_variant: str
    corrdist_mode: str
    states: pd.DataFrame


def _summary(values: pd.Series) -> dict[str, float]:
    data = values.to_numpy(float); median = float(np.median(data))
    return {"q25": float(np.quantile(data, .25)), "q50": median, "q75": float(np.quantile(data, .75)), "mad": float(np.median(np.abs(data - median)))}


def consensus_trajectories(records: list[ConfigurationRecord], config: ContinuousStateV45Config) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    id_columns = ["dataset", "window_id", "window_index", "start_cycle_effective", "end_cycle_effective", "center_cycle_effective", "start_cycle_actual", "end_cycle_actual", "center_cycle_actual", "cycle_effective", "cycle_actual"]
    for record in records:
        state = record.states.loc[record.states.center_cycle_effective >= config.comparison_start_cycles, [*id_columns, *STATE_METRICS]].copy()
        state["configuration_id"] = record.config_id
        state["configuration_baseline_cycles"] = record.baseline_cycles
        state["configuration_distance_form"] = record.distance_form
        state["configuration_feature_variant"] = record.feature_variant
        state["configuration_corrdist_mode"] = record.corrdist_mode
        pieces.append(state)
    long = pd.concat(pieces, ignore_index=True)
    rows: list[dict[str, object]] = []
    for key, group in long.groupby(id_columns, sort=True):
        row = dict(zip(id_columns, key)); row["effective_configuration_count"] = int(group.configuration_id.nunique())
        for metric in STATE_METRICS:
            name = METRIC_OUTPUT_NAMES.get(metric, metric)
            for statistic, value in _summary(group[metric]).items():
                row[f"{name}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _trend_sign(cycles: np.ndarray, values: np.ndarray) -> int:
    if len(values) < 3 or np.ptp(cycles) <= 0:
        return 0
    return int(np.sign(np.polyfit(cycles, values, 1)[0]))


def _segment_agreement(cycles: np.ndarray, left: np.ndarray, right: np.ndarray, segments: int) -> float:
    edges = np.linspace(cycles.min(), cycles.max(), segments + 1); agreements: list[bool] = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (cycles >= start) & (cycles <= end)
        if mask.sum() >= 3:
            agreements.append(_trend_sign(cycles[mask], left[mask]) == _trend_sign(cycles[mask], right[mask]))
    return float(np.mean(agreements)) if agreements else np.nan


def _high_overlap(left: np.ndarray, right: np.ndarray, quantile: float) -> float:
    a, b = left >= np.quantile(left, quantile), right >= np.quantile(right, quantile)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def trajectory_stability(records: list[ConfigurationRecord], config: ContinuousStateV45Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in sorted({str(record.states.dataset.iloc[0]) for record in records}):
        available = [record for record in records if str(record.states.dataset.iloc[0]) == dataset]
        for left_record, right_record in combinations(available, 2):
            identifiers = ["window_index", "center_cycle_effective"]
            left = left_record.states.loc[left_record.states.center_cycle_effective >= config.comparison_start_cycles, [*identifiers, *STATE_METRICS]]
            right = right_record.states.loc[right_record.states.center_cycle_effective >= config.comparison_start_cycles, [*identifiers, *STATE_METRICS]]
            merged = left.merge(right, on=identifiers, suffixes=("_left", "_right"))
            if len(merged) < 10:
                continue
            cycles = merged.center_cycle_effective.to_numpy(float)
            for metric in STATE_METRICS:
                a, b = merged[f"{metric}_left"].to_numpy(float), merged[f"{metric}_right"].to_numpy(float)
                rows.append({"row_type": "pairwise", "dataset": dataset, "metric": METRIC_OUTPUT_NAMES.get(metric, metric),
                             "configuration_left": left_record.config_id, "configuration_right": right_record.config_id, "common_windows": int(len(merged)),
                             "full_spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
                             "segmented_trend_agreement": _segment_agreement(cycles, a, b, config.trend_segments),
                             "major_high_value_overlap": _high_overlap(a, b, config.high_value_quantile)})
    pairwise = pd.DataFrame(rows); summary: list[dict[str, object]] = []
    if not pairwise.empty:
        for (dataset, metric), group in pairwise.groupby(["dataset", "metric"]):
            summary.append({"row_type": "summary", "dataset": dataset, "metric": metric, "configuration_pair_count": int(len(group)),
                            "full_spearman": float(group.full_spearman.median()), "full_spearman_q25": float(group.full_spearman.quantile(.25)), "full_spearman_q75": float(group.full_spearman.quantile(.75)),
                            "segmented_trend_agreement": float(group.segmented_trend_agreement.median()), "major_high_value_overlap": float(group.major_high_value_overlap.median())})
    return pd.concat((pairwise, pd.DataFrame(summary)), ignore_index=True, sort=False)


def _display_standard(values: pd.Series, eps: float) -> pd.Series:
    median = values.median(); iqr = values.quantile(.75) - values.quantile(.25)
    return (values - median) / max(float(iqr), eps)


def v44_vs_v45(v45: pd.DataFrame, v44: pd.DataFrame, config: ContinuousStateV45Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []; plot_rows: list[pd.DataFrame] = []
    metric_pairs = (("D_state", "D_state_q50"), ("V1000_norm", "V1000_norm_q50"), ("multi_scale_rate_divergence", "multi_scale_rate_divergence_q50"), ("state_volatility", "state_volatility_q50"))
    for dataset in sorted(v45.dataset.unique()):
        left = v44.loc[(v44.dataset == dataset) & (v44.center_cycle_effective >= config.comparison_start_cycles)].copy()
        right = v45.loc[(v45.dataset == dataset) & (v45.center_cycle_effective >= config.comparison_start_cycles)].copy()
        for label, column in metric_pairs:
            merged = left.loc[:, ["window_index", "center_cycle_effective", "center_cycle_actual", column]].merge(right.loc[:, ["window_index", column]], on="window_index", suffixes=("_v44", "_v45"))
            a, b = merged[f"{column}_v44"].to_numpy(float), merged[f"{column}_v45"].to_numpy(float)
            rows.append({"dataset": dataset, "metric": label, "common_windows": int(len(merged)),
                         "full_spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
                         "segmented_trend_agreement": _segment_agreement(merged.center_cycle_effective.to_numpy(float), a, b, config.trend_segments),
                         "major_high_value_overlap": _high_overlap(a, b, config.high_value_quantile)})
            view = merged.loc[:, ["window_index", "center_cycle_effective", "center_cycle_actual"]].copy()
            view["dataset"] = dataset; view["metric"] = label
            view["v44_display_standard"] = _display_standard(merged[f"{column}_v44"], config.eps).to_numpy(float)
            view["v45_display_standard"] = _display_standard(merged[f"{column}_v45"], config.eps).to_numpy(float)
            plot_rows.append(view)
    return pd.DataFrame(rows), pd.concat(plot_rows, ignore_index=True)


def state_space_summary(consensus: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    columns = {"D_state": "D_state_q50", "V1000": "V1000_norm_q50", "multi_scale_rate_divergence": "multi_scale_rate_divergence_q50", "state_volatility": "state_volatility_q50"}
    for dataset, group in consensus.groupby("dataset", sort=True):
        ordered = group.sort_values("center_cycle_effective").copy()
        ordered["time_tertile"] = pd.qcut(ordered.center_cycle_effective, 3, labels=("early", "middle", "late"))
        for period, subset in ordered.groupby("time_tertile", observed=True):
            row: dict[str, object] = {"row_type": "time_tertile", "dataset": dataset, "segment": str(period), "windows": int(len(subset)),
                                      "effective_start": float(subset.center_cycle_effective.min()), "effective_end": float(subset.center_cycle_effective.max())}
            for label, column in columns.items():
                row[f"{label}_median"] = float(subset[column].median()); row[f"{label}_q25"] = float(subset[column].quantile(.25)); row[f"{label}_q75"] = float(subset[column].quantile(.75))
            rows.append(row)
        for left, right in combinations(columns, 2):
            rows.append({"row_type": "metric_complementarity", "dataset": dataset, "segment": "all", "metric_left": left, "metric_right": right,
                         "spearman": float(ordered[columns[left]].corr(ordered[columns[right]], method="spearman")), "windows": int(len(ordered))})
    return pd.DataFrame(rows)


def ry_path_audit(variant_consensus: dict[str, pd.DataFrame], canonical_details: dict[str, pd.DataFrame], config: ContinuousStateV45Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ("D_state_q50", "V1000_norm_q50", "multi_scale_rate_divergence_q50", "state_volatility_q50")
    for dataset in sorted({str(frame.dataset.iloc[0]) for frame in variant_consensus.values()}):
        for left_name, right_name in combinations(variant_consensus, 2):
            left = variant_consensus[left_name].loc[variant_consensus[left_name].dataset == dataset, ["window_index", *metrics]]
            right = variant_consensus[right_name].loc[variant_consensus[right_name].dataset == dataset, ["window_index", *metrics]]
            merged = left.merge(right, on="window_index", suffixes=("_left", "_right"))
            for metric in metrics:
                difference = float(np.median(np.abs(
                    _display_standard(merged[f"{metric}_left"], config.eps)
                    - _display_standard(merged[f"{metric}_right"], config.eps)
                )))
                rows.append({"row_type": "path_comparison", "dataset": dataset, "comparison": f"{left_name}_vs_{right_name}", "metric": metric.removesuffix("_q50"),
                             "common_windows": int(len(merged)), "spearman": float(merged[f"{metric}_left"].corr(merged[f"{metric}_right"], method="spearman")),
                             "median_absolute_display_difference": difference})
        detail = canonical_details[dataset]
        for group in ("rx", "ry", "rs"):
            column = f"D_{group}_contribution"
            if column in detail:
                rows.append({"row_type": "canonical_group_contribution", "dataset": dataset, "comparison": "full_with_corrdist", "metric": group,
                             "mean_contribution": float(detail[column].mean()), "p95_contribution": float(detail[column].quantile(.95)), "fraction_over_060": float((detail[column] > .60).mean()), "common_windows": int(len(detail))})
    return pd.DataFrame(rows)
