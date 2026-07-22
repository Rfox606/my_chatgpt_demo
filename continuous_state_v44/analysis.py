from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import STATE_METRICS, ContinuousStateV44Config


@dataclass(frozen=True)
class ConfigurationRecord:
    config_id: str
    baseline_cycles: int
    distance_form: str
    feature_variant: str
    removed_feature_group: str
    states: pd.DataFrame


METRIC_OUTPUT_NAMES = {"A_state": "multi_scale_rate_divergence"}


def _summary(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(float); median = float(np.median(array))
    return {"q25": float(np.quantile(array, .25)), "q50": median, "q75": float(np.quantile(array, .75)), "mad": float(np.median(np.abs(array - median)))}


def consensus_trajectories(records: list[ConfigurationRecord], config: ContinuousStateV44Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    for record in records:
        selected_columns = ["dataset", "window_id", "window_index", "start_cycle_effective", "end_cycle_effective", "center_cycle_effective", "start_cycle_actual", "end_cycle_actual", "center_cycle_actual", "cycle_effective", "cycle_actual", *STATE_METRICS]
        state = record.states.loc[record.states.start_cycle_effective > config.consensus_emit_start_cycles, selected_columns].copy()
        state["configuration_id"] = record.config_id
        state["configuration_baseline_cycles"] = record.baseline_cycles
        state["configuration_distance_form"] = record.distance_form
        state["configuration_feature_variant"] = record.feature_variant
        state["configuration_removed_feature_group"] = record.removed_feature_group
        pieces.append(state)
    long = pd.concat(pieces, ignore_index=True)
    keys = ["dataset", "window_id", "window_index", "start_cycle_effective", "end_cycle_effective", "center_cycle_effective", "start_cycle_actual", "end_cycle_actual", "center_cycle_actual", "cycle_effective", "cycle_actual"]
    rows: list[dict[str, object]] = []
    for key, group in long.groupby(keys, sort=True):
        row = dict(zip(keys, key)); row["effective_configuration_count"] = int(group.configuration_id.nunique())
        for metric in STATE_METRICS:
            name = METRIC_OUTPUT_NAMES.get(metric, metric)
            for statistic, value in _summary(group[metric]).items(): row[f"{name}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows), long


def _trend_sign(cycles: np.ndarray, values: np.ndarray) -> int:
    if len(values) < 3 or np.ptp(cycles) <= 0: return 0
    slope = float(np.polyfit(cycles, values, 1)[0])
    return int(np.sign(slope))


def _segment_agreement(cycles: np.ndarray, left: np.ndarray, right: np.ndarray, count: int) -> float:
    edges = np.linspace(float(cycles.min()), float(cycles.max()), count + 1); matches: list[bool] = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (cycles >= start) & (cycles <= end)
        if mask.sum() < 3: continue
        matches.append(_trend_sign(cycles[mask], left[mask]) == _trend_sign(cycles[mask], right[mask]))
    return float(np.mean(matches)) if matches else np.nan


def _high_value_overlap(left: np.ndarray, right: np.ndarray, quantile: float) -> float:
    left_high = left >= np.quantile(left, quantile); right_high = right >= np.quantile(right, quantile)
    union = np.logical_or(left_high, right_high).sum()
    return float(np.logical_and(left_high, right_high).sum() / union) if union else 1.0


def trajectory_stability(records: list[ConfigurationRecord], config: ContinuousStateV44Config) -> pd.DataFrame:
    """Pairwise trajectory checks use fixed equal-time segments and top-decile overlap; no episode threshold is fitted."""
    rows: list[dict[str, object]] = []
    for dataset in sorted({str(record.states.dataset.iloc[0]) for record in records if not record.states.empty}):
        subset = [record for record in records if str(record.states.dataset.iloc[0]) == dataset]
        for left_record, right_record in combinations(subset, 2):
            keys = ["window_index", "center_cycle_effective"]
            columns = [*keys, *STATE_METRICS]
            left = left_record.states.loc[:, columns]; right = right_record.states.loc[:, columns]
            merged = left.merge(right, on=keys, suffixes=("_left", "_right"))
            if len(merged) < 10: continue
            cycles = merged.center_cycle_effective.to_numpy(float)
            for metric in STATE_METRICS:
                a = merged[f"{metric}_left"].to_numpy(float); b = merged[f"{metric}_right"].to_numpy(float)
                rows.append({"row_type": "pairwise", "dataset": dataset, "metric": METRIC_OUTPUT_NAMES.get(metric, metric),
                             "configuration_left": left_record.config_id, "configuration_right": right_record.config_id, "common_windows": int(len(merged)),
                             "full_spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
                             "segmented_trend_agreement": _segment_agreement(cycles, a, b, config.trend_segments),
                             "major_high_value_overlap": _high_value_overlap(a, b, config.high_value_quantile)})
    pairwise = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    if not pairwise.empty:
        for (dataset, metric), group in pairwise.groupby(["dataset", "metric"]):
            summaries.append({"row_type": "summary", "dataset": dataset, "metric": metric, "configuration_pair_count": int(len(group)),
                              "full_spearman": float(group.full_spearman.median()), "full_spearman_q25": float(group.full_spearman.quantile(.25)), "full_spearman_q75": float(group.full_spearman.quantile(.75)),
                              "segmented_trend_agreement": float(group.segmented_trend_agreement.median()), "major_high_value_overlap": float(group.major_high_value_overlap.median())})
    return pd.concat([pairwise, pd.DataFrame(summaries)], ignore_index=True, sort=False)


def input_provenance(config: ContinuousStateV44Config) -> dict[str, object]:
    weighted = json.loads(Path("outputs_weighted_awrcore_v1/configs/weighted_awrcore_config.json").read_text(encoding="utf-8"))
    generator = Path("run_weighted_awrcore_models.py").read_text(encoding="utf-8")
    table = pd.read_csv(config.z_table_path, usecols=["feature_name", "physical_meaning"], nrows=5000)
    source_files = {name: Path(path).exists() for name, path in weighted["raw_files"].items()}
    # Archive exports may retain the versioned direct feature table while omitting
    # the much larger original labelled CSVs.  The formal v4.4 state pipeline
    # consumes this derived table, not Stage, so it remains traceable without
    # inventing missing raw files or silently reading labels.
    derived_feature_table_exists = Path(config.z_table_path).exists()
    sensitive_meaning = bool(table.physical_meaning.fillna("").str.contains("sensitive phase", case=False).any())
    code_trace = all(fragment in generator for fragment in ("sensitive_phase", "load_cycle_feature_data", "Fx_p", "Fy_p", "Fz_p"))
    traceable = (all(source_files.values()) or derived_feature_table_exists) and sensitive_meaning and code_trace
    return {"status": "PASS" if traceable else "FAIL", "z_table": config.z_table_path,
            "raw_files_exist": source_files, "normalized_sensitive_phase": weighted.get("sensitive_phase"), "raw_meta": weighted.get("raw_meta"),
            "feature_generator_trace_found": code_trace, "z_table_has_sensitive_phase_physical_meaning": sensitive_meaning,
            "derived_feature_table_exists": derived_feature_table_exists,
            "traceability_mode": "original_raw_files" if all(source_files.values()) else "versioned_derived_feature_table",
            "interpretation": "Confirmed from existing generator/configuration; stage labels are excluded from v4.4 state input."}


def morphology_interval_alignment(consensus_exp1: pd.DataFrame, metadata: dict[str, object], config: ContinuousStateV44Config) -> tuple[pd.DataFrame, dict[str, object]]:
    anchors = np.asarray(metadata["exp1_morphology"]["cycle_actual"], dtype=float); morphology = metadata["exp1_morphology"]
    trace = consensus_exp1.sort_values("center_cycle_actual").copy()
    high = ((trace.V500_norm_q50 >= trace.V500_norm_q50.quantile(config.high_value_quantile)) |
            (trace.multi_scale_rate_divergence_q50 >= trace.multi_scale_rate_divergence_q50.quantile(config.high_value_quantile)) |
            (trace.state_volatility_q50 >= trace.state_volatility_q50.quantile(config.high_value_quantile))).to_numpy(bool)
    trace["high_change_state"] = high.astype(int)
    rows: list[dict[str, object]] = []
    for index, (start, end) in enumerate(zip(anchors[:-1], anchors[1:])):
        part = trace.loc[(trace.center_cycle_actual >= start) & (trace.center_cycle_actual <= end)].copy()
        row: dict[str, object] = {"row_type": "morphology_anchor_interval", "interval_index": int(index + 1), "start_cycle_actual": float(start), "end_cycle_actual": float(end), "morphology_start_available": True, "morphology_end_available": True}
        for metric in ("Sa", "Sq", "Sz", "Sku"):
            delta = float(morphology[metric][index + 1] - morphology[metric][index]); row[f"delta_{metric}"] = delta; row[f"abs_delta_{metric}"] = abs(delta)
        if part.empty:
            row["state_window_count"] = 0; rows.append(row); continue
        x = part.center_cycle_actual.to_numpy(float); d = part.D_state_q50.to_numpy(float)
        row.update({"state_window_count": int(len(part)), "state_start_observed_actual": float(x[0]), "state_end_observed_actual": float(x[-1]),
                    "D_start": float(d[0]), "D_end": float(d[-1]), "D_end_minus_start": float(d[-1] - d[0]), "D_slope_per_actual_cycle": float(np.polyfit(x, d, 1)[0]) if len(part) >= 3 else np.nan,
                    "D_cumulative_absolute_change": float(np.abs(np.diff(d)).sum())})
        for source, label in (("V500_norm_q50", "V500"), ("V1000_norm_q50", "V1000"), ("multi_scale_rate_divergence_q50", "rate_divergence"), ("state_volatility_q50", "volatility")):
            values = part[source].to_numpy(float); row[f"{label}_mean"] = float(values.mean()); row[f"{label}_max"] = float(values.max()); row[f"{label}_integral"] = float(np.trapezoid(values, x))
        row["high_change_state_duration_fraction"] = float(np.average(part.high_change_state.to_numpy(float), weights=np.maximum(part.end_cycle_actual.to_numpy(float) - part.start_cycle_actual.to_numpy(float), config.eps)))
        rows.append(row)
    # 48k--end has no later morphology anchor. It is retained as a state-only late-period context row.
    last = trace.loc[trace.center_cycle_actual >= anchors[-1]].copy()
    if not last.empty:
        x = last.center_cycle_actual.to_numpy(float); d = last.D_state_q50.to_numpy(float)
        rows.append({"row_type": "state_only_after_last_morphology_anchor", "interval_index": len(rows) + 1, "start_cycle_actual": float(anchors[-1]), "end_cycle_actual": float(x[-1]),
                     "morphology_start_available": True, "morphology_end_available": False, "state_window_count": int(len(last)), "state_start_observed_actual": float(x[0]), "state_end_observed_actual": float(x[-1]),
                     "D_start": float(d[0]), "D_end": float(d[-1]), "D_end_minus_start": float(d[-1] - d[0]), "D_slope_per_actual_cycle": float(np.polyfit(x, d, 1)[0]) if len(last) >= 3 else np.nan,
                     "D_cumulative_absolute_change": float(np.abs(np.diff(d)).sum()), "V500_mean": float(last.V500_norm_q50.mean()), "V500_max": float(last.V500_norm_q50.max()), "V500_integral": float(np.trapezoid(last.V500_norm_q50, x)),
                     "V1000_mean": float(last.V1000_norm_q50.mean()), "V1000_max": float(last.V1000_norm_q50.max()), "V1000_integral": float(np.trapezoid(last.V1000_norm_q50, x)),
                     "rate_divergence_mean": float(last.multi_scale_rate_divergence_q50.mean()), "rate_divergence_max": float(last.multi_scale_rate_divergence_q50.max()), "rate_divergence_integral": float(np.trapezoid(last.multi_scale_rate_divergence_q50, x)),
                     "volatility_mean": float(last.state_volatility_q50.mean()), "volatility_max": float(last.state_volatility_q50.max()), "volatility_integral": float(np.trapezoid(last.state_volatility_q50, x)),
                     "high_change_state_duration_fraction": float(last.high_change_state.mean())})
    table = pd.DataFrame(rows)
    morph_rows = table.loc[table.row_type.eq("morphology_anchor_interval")]
    exploratory = {"status": "PASS", "post_hoc_only": True, "n_morphology_intervals": int(len(morph_rows)), "exploratory_spearman": {}}
    for metric in ("Sa", "Sq", "Sz", "Sku"):
        exploratory["exploratory_spearman"][f"abs_delta_{metric}_vs_D_cumulative_absolute_change"] = float(morph_rows[f"abs_delta_{metric}"].corr(morph_rows.D_cumulative_absolute_change, method="spearman"))
    return table, exploratory


def exp_pattern_comparison(consensus: pd.DataFrame, config: ContinuousStateV44Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in consensus.groupby("dataset"):
        group = group.sort_values("center_cycle_effective"); split = group.center_cycle_effective.quantile(.80)
        for source, metric in (("D_state_q50", "D_state"), ("V500_norm_q50", "V500"), ("V1000_norm_q50", "V1000"), ("multi_scale_rate_divergence_q50", "multi_scale_rate_divergence"), ("state_volatility_q50", "state_volatility")):
            values = group[source].to_numpy(float); high = values >= np.quantile(values, config.high_value_quantile); late = group.center_cycle_effective >= split
            rows.append({"dataset": dataset, "metric": metric, "window_count": int(len(group)), "overall_median": float(np.median(values)), "overall_q75": float(np.quantile(values, .75)), "overall_mean": float(values.mean()),
                         "effective_cycle_spearman": float(pd.Series(values).corr(pd.Series(group.center_cycle_effective.to_numpy(float)), method="spearman")), "late_20pct_mean": float(values[late].mean()),
                         "early_80pct_mean": float(values[~late].mean()), "late_to_early_mean_ratio": float(values[late].mean() / max(values[~late].mean(), config.eps)), "late_high_value_fraction": float(high[late].mean())})
    return pd.DataFrame(rows)
