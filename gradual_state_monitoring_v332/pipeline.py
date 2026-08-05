from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gradual_state_monitoring_v33.data import load_window_data

from .config import GradualStateMonitoringV332Config
from .forecasting import ForecastRun, run_label_free_forecasts
from .labels import (
    attach_posthoc_stage, boundary_detection, effective_to_actual, load_existing_stage_mapping,
    ordered_state_mapping, stage_boundaries, stage_stability_metrics, stable_region_mask,
)
from .online import validate_label_free_online_frame
from .plots import write_alignment_figure, write_boundary_summary, write_stage_stability_summary
from .report import write_report
from .state import replay_model_prediction_surprise


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value)!r}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _require_v331_outputs(config: GradualStateMonitoringV332Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(config.v331_output_dir)
    state_path = root / "v331_state_path.csv"
    evidence_path = root / "v331_change_evidence.csv"
    if not state_path.exists() or not evidence_path.exists():
        raise FileNotFoundError("V3.3.2 evaluates the committed V3.3.1 output; the required V3.3.1 files are missing")
    state = pd.read_csv(state_path)
    evidence = pd.read_csv(evidence_path)
    required_state = {"dataset", "input_index", "cycle", "online_state", "local_state_id"}
    if not required_state.issubset(state.columns):
        raise ValueError("V3.3.1 state path schema does not support label evaluation")
    return state, evidence


def _boundary_metrics_by_dataset(posthoc: pd.DataFrame, boundaries: pd.DataFrame, config: GradualStateMonitoringV332Config,
                                 *, model: str, setting: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    logs: list[pd.DataFrame] = []
    for dataset, group in posthoc.groupby("dataset", sort=True):
        local_boundaries = boundaries[boundaries.dataset == dataset]
        for tolerance in config.boundary_tolerances_actual_cycles:
            metric, log = boundary_detection(group, local_boundaries, tolerance, model=model, setting=setting)
            metric["dataset"] = dataset
            metrics.append(metric)
            logs.append(log)
    return pd.DataFrame(metrics), pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()


def _forecast_metrics(predictions: pd.DataFrame, segments: pd.DataFrame, boundaries: pd.DataFrame,
                      config: GradualStateMonitoringV332Config) -> pd.DataFrame:
    """Label-derived metrics are calculated only now, after all predictors are frozen."""
    values = predictions[predictions.target_available == 1].copy()
    values["origin_actual_cycle"] = np.nan
    values["stage_interior"] = False
    for dataset, index in values.groupby("target_dataset", sort=False).groups.items():
        positions = list(index)
        effective = values.loc[positions, "origin_cycle"].to_numpy(float)
        actual = effective_to_actual(effective, str(dataset), segments)
        actual_boundaries = boundaries.loc[boundaries.dataset == dataset, "boundary_actual_cycle"].to_numpy(float)
        values.loc[positions, "origin_actual_cycle"] = actual
        values.loc[positions, "stage_interior"] = stable_region_mask(actual, actual_boundaries, config.stage_buffer_actual_cycles)
    group_columns = ["training_setting", "source_dataset", "target_dataset", "model", "horizon", "training_examples", "prediction_start_index"]
    rows: list[dict[str, object]] = []
    for key, group in values.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, key))
        stable = group[group.stage_interior]
        transition = group[~group.stage_interior]
        row.update({
            "evaluated_windows": int(len(group)), "MAE": float(group.mae.mean()),
            "stable_interval_MAE": float(stable.mae.mean()) if len(stable) else np.nan,
            "transition_interval_MAE": float(transition.mae.mean()) if len(transition) else np.nan,
            "evaluation_region": "complete_predeclared_suffix",
        })
        rows.append(row)
    metric = pd.DataFrame(rows)
    reference_keys = ["training_setting", "source_dataset", "target_dataset", "horizon"]
    references = metric[metric.model == "Persistence"].loc[:, [*reference_keys, "MAE"]].rename(columns={"MAE": "persistence_MAE"})
    metric = metric.merge(references, on=reference_keys, how="left")
    metric["relative_improvement_vs_Persistence"] = (metric.persistence_MAE - metric.MAE) / metric.persistence_MAE
    return metric.drop(columns="persistence_MAE")


def _model_comparison_row(model: str, setting: str, dataset: str, state: pd.DataFrame, stable: pd.DataFrame,
                          boundary: pd.DataFrame, config: GradualStateMonitoringV332Config) -> dict[str, object]:
    selected_stable = stable[(stable.dataset == dataset) & (stable.model == model) & (stable.setting == setting)]
    selected_boundary = boundary[(boundary.dataset == dataset) & (boundary.model == model) & (boundary.setting == setting)
                                 & (boundary.tolerance_actual_cycles == config.primary_boundary_tolerance_actual_cycles)]
    values = state.online_state.to_numpy(str)
    state_switches = int(np.sum(values[1:] != values[:-1])) if len(values) > 1 else 0
    return {
        "training_setting": setting, "dataset": dataset, "model": model,
        "boundary_precision": float(selected_boundary.boundary_precision.iloc[0]) if len(selected_boundary) else np.nan,
        "boundary_recall": float(selected_boundary.boundary_recall.iloc[0]) if len(selected_boundary) else np.nan,
        "boundary_f1": float(selected_boundary.boundary_f1.iloc[0]) if len(selected_boundary) else np.nan,
        "mean_detection_delay": float(selected_boundary.mean_detection_delay.iloc[0]) if len(selected_boundary) else np.nan,
        "stage_stable_rate": float(selected_stable.stage_stable_rate.mean()) if len(selected_stable) else np.nan,
        "false_alarms_per_1000_windows": float(1000.0 * selected_boundary.false_alarm_count.iloc[0] / len(state)) if len(selected_boundary) and len(state) else np.nan,
        "transition_fraction": float(np.mean(values == "TRANSITION")) if len(values) else np.nan,
        "state_fragmentation": state_switches,
        "state_switches_per_1000_windows": float(1000.0 * state_switches / len(state)) if len(state) else np.nan,
        "state_machine": "V3.3.1_fixed",
        "other_evidence": "V3.3.1_fixed_distribution_shift_and_drift_velocity",
    }


def _gates(forecast: pd.DataFrame, stable: pd.DataFrame, boundary: pd.DataFrame, comparison: pd.DataFrame,
           config: GradualStateMonitoringV332Config) -> dict[str, object]:
    long = forecast[forecast.horizon.isin((100, 300))]
    nonlinear = long[long.model.isin(("TCN", "GRU"))]
    gate_a = bool((nonlinear.relative_improvement_vs_Persistence > 0).any())
    baseline_boundary = boundary[(boundary.model == "V331") & (boundary.setting == "v331_frozen") &
                                 (boundary.tolerance_actual_cycles == config.primary_boundary_tolerance_actual_cycles)]
    b_rows: list[dict[str, object]] = []
    for dataset in ("Exp1", "Exp2"):
        row = baseline_boundary[baseline_boundary.dataset == dataset]
        recall = float(row.boundary_recall.iloc[0]) if len(row) else np.nan
        precision = float(row.boundary_precision.iloc[0]) if len(row) else np.nan
        b_rows.append({"dataset": dataset, "boundary_recall": recall, "boundary_precision": precision,
                       "status": "PASS" if recall >= config.gate_boundary_recall_min and precision >= config.gate_boundary_precision_min else "FAIL"})
    gate_b = bool(b_rows and all(row["status"] == "PASS" for row in b_rows))
    baseline_stable = stable[(stable.model == "V331") & (stable.setting == "v331_frozen")]
    c_rows: list[dict[str, object]] = []
    for dataset in ("Exp1", "Exp2"):
        row = baseline_stable[baseline_stable.dataset == dataset]
        mean_rate = float(row.stage_stable_rate.mean()) if len(row) else np.nan
        false_rate = float(row.false_switches_per_1000_windows.mean()) if len(row) else np.nan
        c_rows.append({"dataset": dataset, "mean_stage_stable_rate": mean_rate, "false_switches_per_1000_windows": false_rate,
                       "status": "PASS" if mean_rate >= config.gate_stage_stable_rate_min and false_rate <= config.gate_false_switches_per_1000_max else "FAIL"})
    gate_c = bool(c_rows and all(row["status"] == "PASS" for row in c_rows))
    d_rows: list[dict[str, object]] = []
    baseline_compare = comparison[comparison.model == "V331"].set_index("dataset")
    for _, candidate in comparison[comparison.model.isin(("TCN", "GRU"))].iterrows():
        if candidate.dataset not in baseline_compare.index:
            continue
        reference = baseline_compare.loc[candidate.dataset]
        f1_gain = float(candidate.boundary_f1 - reference.boundary_f1)
        false_reduction = float((reference.false_alarms_per_1000_windows - candidate.false_alarms_per_1000_windows) / max(reference.false_alarms_per_1000_windows, config.eps))
        other_not_worse = bool(candidate.stage_stable_rate >= reference.stage_stable_rate - 0.05 and candidate.boundary_f1 >= reference.boundary_f1 - 0.05)
        value = bool((f1_gain > 0 or false_reduction >= config.gate_false_alarm_reduction_min) and other_not_worse)
        d_rows.append({"dataset": candidate.dataset, "setting": candidate.training_setting, "model": candidate.model,
                       "boundary_f1_gain_vs_v331": f1_gain, "false_alarm_reduction_vs_v331": false_reduction,
                       "other_metric_not_materially_worse": other_not_worse, "value_status": "PASS" if value else "FAIL"})
    gate_d = bool(any(row["value_status"] == "PASS" for row in d_rows))
    return {
        "gate_A_forecast": {"status": "PASS" if gate_a else "FAIL", "criterion": "A TCN or GRU has lower MAE than Persistence on a predeclared 100 or 300 horizon.",
                            "qualifying_rows": int((nonlinear.relative_improvement_vs_Persistence > 0).sum())},
        "gate_B_label_boundary_detection": {"status": "PASS" if gate_b else "FAIL", "criterion": "Every experiment at ±250 actual cycles: recall >= 0.75 and precision >= 0.50.", "datasets": b_rows},
        "gate_C_stage_interior_stability": {"status": "PASS" if gate_c else "FAIL", "criterion": "Every experiment: mean stable rate >= 0.70 and false switches <= 5 / 1000 stable windows.", "datasets": c_rows},
        "gate_D_model_upgrade_value": {"status": "PASS" if gate_d else "FAIL", "criterion": "TCN/GRU improves boundary F1 or reduces false alarms by >=20%, without material worsening of the other state metric.", "candidates": d_rows},
        "overall_status": "PASS" if gate_a and gate_b and gate_c and gate_d else "FAIL",
        "primary_conclusion_basis": ["label_boundary_detection", "stage_interior_stability"],
        "v331_output_reused_without_modification": True,
        "stage_used_online": False,
    }


def run_pipeline(config: GradualStateMonitoringV332Config) -> dict[str, Any]:
    root = config.paths()["root"]
    _write_json(root / "v332_config.json", config.jsonable())
    # Label-free phase.  The imported V3.3.1 path was frozen at commit 6162244.
    raw = load_window_data(config.input_path)
    frames = {str(dataset): validate_label_free_online_frame(group.reset_index(drop=True), config)
              for dataset, group in raw.groupby("dataset", sort=True)}
    if set(frames) != {"Exp1", "Exp2"}:
        raise ValueError("V3.3.2 predeclared experiment comparison requires Exp1 and Exp2")
    v331_state, v331_evidence = _require_v331_outputs(config)
    forecast_runs = run_label_free_forecasts(frames, config)
    forecast_predictions = pd.concat([run.predictions for run in forecast_runs], ignore_index=True)
    model_state_paths: list[pd.DataFrame] = []
    model_evidence: list[pd.DataFrame] = []
    model_transitions: list[pd.DataFrame] = []
    for run in forecast_runs:
        target = frames[run.target_dataset]
        for model, error in run.h20_error_at_arrival.items():
            state, evidence, transitions = replay_model_prediction_surprise(target, error, config)
            state["model"] = model; state["training_setting"] = run.setting
            evidence["model"] = model; evidence["training_setting"] = run.setting
            if len(transitions):
                transitions["model"] = model; transitions["training_setting"] = run.setting
            model_state_paths.append(state); model_evidence.append(evidence); model_transitions.append(transitions)
    # Offline-only phase starts here.  No label object was passed to the preceding calls.
    segments, provenance = load_existing_stage_mapping(config)
    boundaries = stage_boundaries(segments)
    _write_json(root / "v332_label_provenance.json", provenance)
    boundaries.to_csv(root / "v332_stage_boundaries.csv", index=False)
    v331_posthoc = attach_posthoc_stage(v331_state, segments)
    v331_posthoc["model"] = "V331"; v331_posthoc["training_setting"] = "v331_frozen"
    all_posthoc: list[pd.DataFrame] = [v331_posthoc]
    for path in model_state_paths:
        all_posthoc.append(attach_posthoc_stage(path, segments))
    model_paths_posthoc = pd.concat(all_posthoc[1:], ignore_index=True)
    model_paths_posthoc.to_csv(root / "v332_model_state_paths.csv", index=False)
    if model_transitions:
        pd.concat(model_transitions, ignore_index=True).to_csv(root / "v332_model_state_transition_log.csv", index=False)
    forecast_metric = _forecast_metrics(forecast_predictions, segments, boundaries, config)
    forecast_metric.to_csv(root / "v332_forecast_metrics.csv", index=False)
    stable_parts: list[pd.DataFrame] = []
    boundary_parts: list[pd.DataFrame] = []
    log_parts: list[pd.DataFrame] = []
    for model, setting, posthoc in [("V331", "v331_frozen", v331_posthoc)]:
        stable_parts.append(stage_stability_metrics(posthoc, boundaries, config, model=model, setting=setting))
        metrics, log = _boundary_metrics_by_dataset(posthoc, boundaries, config, model=model, setting=setting)
        boundary_parts.append(metrics); log_parts.append(log)
    for (model, setting), posthoc in model_paths_posthoc.groupby(["model", "training_setting"], sort=True):
        stable_parts.append(stage_stability_metrics(posthoc, boundaries, config, model=str(model), setting=str(setting)))
        metrics, log = _boundary_metrics_by_dataset(posthoc, boundaries, config, model=str(model), setting=str(setting))
        boundary_parts.append(metrics); log_parts.append(log)
    stable_metrics = pd.concat(stable_parts, ignore_index=True)
    boundary_metrics = pd.concat(boundary_parts, ignore_index=True)
    match_log = pd.concat(log_parts, ignore_index=True)
    stable_metrics.to_csv(root / "v332_stage_stability_metrics.csv", index=False)
    boundary_metrics.to_csv(root / "v332_boundary_detection_metrics.csv", index=False)
    match_log.to_csv(root / "v332_boundary_match_log.csv", index=False)
    mapping, confusion, mapping_summary = ordered_state_mapping(v331_posthoc, model="V331", setting="v331_frozen")
    pd.DataFrame(mapping_summary).to_csv(root / "v332_ordered_mapping_summary.csv", index=False)
    mapping.to_csv(root / "v332_ordered_state_mapping.csv", index=False)
    confusion.to_csv(root / "v332_confusion_matrix.csv", index=False)
    comparison_rows: list[dict[str, object]] = []
    for dataset, state in v331_posthoc.groupby("dataset", sort=True):
        comparison_rows.append(_model_comparison_row("V331", "v331_frozen", str(dataset), state, stable_metrics, boundary_metrics, config))
    for (model, setting, dataset), state in model_paths_posthoc.groupby(["model", "training_setting", "dataset"], sort=True):
        comparison_rows.append(_model_comparison_row(str(model), str(setting), str(dataset), state, stable_metrics, boundary_metrics, config))
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(root / "v332_model_state_detection_comparison.csv", index=False)
    gates = _gates(forecast_metric, stable_metrics, boundary_metrics, comparison, config)
    _write_json(root / "v332_gate_decision.json", gates)
    # Stage-alignment figures intentionally use only the frozen V3.3.1 trajectory.
    v331_evidence = v331_evidence.copy()
    for dataset in ("Exp1", "Exp2"):
        write_alignment_figure(v331_posthoc, v331_evidence, boundaries, match_log[(match_log.model == "V331") & (match_log.setting == "v331_frozen")], dataset,
                               root / f"fig_v332_{dataset.lower()}_stage_alignment.png")
    write_boundary_summary(boundary_metrics, root / "fig_v332_boundary_detection_summary.png")
    write_stage_stability_summary(stable_metrics, root / "fig_v332_stage_stability_summary.png")
    report = write_report(root, config, provenance, boundaries, stable_metrics, boundary_metrics, match_log, mapping_summary, forecast_metric, comparison, gates)
    return {"output_dir": str(root), "gate": gates, "report": str(report), "v331_rows": int(len(v331_state))}
