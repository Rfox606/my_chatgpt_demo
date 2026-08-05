from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import GradualStateMonitoringV33Config
from .data import MAIN_FEATURES, load_window_data
from .evidence import CausalEvidenceEngine
from .evaluation import (
    change_detection_metrics, delayed_entry_evaluation, evidence_ablation, forecast_metrics, horizon_ablation,
    gate_decision, prefix_causality_evaluation, source_initialisation_metrics, synthetic_acceptance,
    window_overlap_ablation,
)
from .features import build_causal_descriptors
from .forecasting import run_causal_benchmark_forecasts, run_dual_timescale, run_source_initialised_forecasts
from .report import write_report
from .state_machine import GradualStateMachine


@dataclass
class MonitorResult:
    evidence: pd.DataFrame
    state_path: pd.DataFrame
    dual_predictions: pd.DataFrame
    transitions: pd.DataFrame


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value)!r}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def run_monitor(frame: pd.DataFrame, config: GradualStateMonitoringV33Config, *, include_dual: bool = True) -> MonitorResult:
    """The causal online core used for real, delayed-entry, prefix, and synthetic runs."""
    ordered = frame.sort_values("window_index", kind="stable").reset_index(drop=True)
    values = ordered.loc[:, MAIN_FEATURES].to_numpy(float)
    if include_dual:
        descriptors = build_causal_descriptors(values)
        dual = run_dual_timescale(descriptors, values, config)
    else:
        dual = pd.DataFrame({
            "input_index": np.arange(len(values), dtype=int),
            "slow_prediction_error": np.full(len(values), np.nan),
            "fast_prediction_error": np.full(len(values), np.nan),
            "fast_slow_prediction_difference": np.full(len(values), np.nan),
        })
    evidence_engine = CausalEvidenceEngine(values.shape[1], config)
    machine = GradualStateMachine(values.shape[1], config)
    evidence_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    for index, row in ordered.iterrows():
        dual_row = dual.iloc[index]
        evidence = evidence_engine.step(values[index], float(dual_row.slow_prediction_error), float(dual_row.fast_prediction_error))
        distance = evidence_engine.scale.distance(values[index], machine.reference)
        state = machine.step(
            index, float(row.center_cycle), values[index], evidence,
            ready=bool(evidence_engine.scale.frozen and index >= config.history_windows - 1), distance=distance,
        )
        common = {"dataset": str(row.dataset), "input_index": index, "window_index": int(row.window_index), "cycle": float(row.center_cycle)}
        evidence_rows.append({**common, **evidence})
        state_rows.append({**common, **state})
    evidence_frame = pd.DataFrame(evidence_rows)
    state_frame = pd.DataFrame(state_rows)
    dual = pd.concat((ordered.loc[:, ["dataset", "window_index", "center_cycle"]].rename(columns={"center_cycle": "cycle"}).reset_index(drop=True), dual), axis=1)
    transitions = pd.DataFrame(machine.events, columns=("input_index", "cycle", "from_state", "to_state", "local_state_id", "state_start_cycle", "reason"))
    if len(transitions):
        transitions.insert(0, "dataset", str(ordered.dataset.iloc[0]))
    return MonitorResult(evidence_frame, state_frame, dual, transitions)


def _combined_online_predictions(benchmark: pd.DataFrame, dual: pd.DataFrame, ordered: pd.DataFrame) -> pd.DataFrame:
    benchmark = benchmark.merge(ordered.loc[:, ["window_index", "center_cycle"]].reset_index().rename(columns={"index": "input_index", "center_cycle": "origin_cycle"}), on="input_index", how="left")
    benchmark["row_type"] = "benchmark_forecast"
    online = dual.copy()
    online["dataset"] = str(ordered.dataset.iloc[0])
    online["model"] = "SlowFast_RLS"
    online["horizon"] = np.nan
    online["row_type"] = "dual_timescale"
    return pd.concat((benchmark, online), ignore_index=True, sort=False)


def run_pipeline(config: GradualStateMonitoringV33Config, *, run_benchmarks: bool = True) -> dict[str, Any]:
    paths = config.paths()
    _write_json(paths["root"] / "v33_config.json", config.jsonable())
    source = load_window_data(config.input_path)
    all_evidence: list[pd.DataFrame] = []
    all_state: list[pd.DataFrame] = []
    all_transitions: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    all_change_metrics: list[pd.DataFrame] = []
    all_ablations: list[pd.DataFrame] = []
    all_overlap: list[pd.DataFrame] = []
    all_delayed: list[pd.DataFrame] = []
    prefixes: dict[str, Any] = {}
    previous_descriptors: np.ndarray | None = None
    previous_values: np.ndarray | None = None
    for dataset, frame in source.groupby("dataset", sort=True):
        local = frame.reset_index(drop=True)
        monitored = run_monitor(local, config)
        all_evidence.append(monitored.evidence)
        all_state.append(monitored.state_path)
        all_transitions.append(monitored.transitions)
        all_change_metrics.append(change_detection_metrics(monitored.evidence, monitored.state_path))
        all_ablations.append(evidence_ablation(monitored.evidence, monitored.state_path, config))
        all_overlap.append(window_overlap_ablation(local, monitored.state_path, config))
        all_delayed.append(delayed_entry_evaluation(local, monitored.state_path, config))
        prefixes[str(dataset)] = prefix_causality_evaluation(local, config, full=monitored)
        if run_benchmarks:
            values = local.loc[:, MAIN_FEATURES].to_numpy(float)
            descriptors = build_causal_descriptors(values)
            benchmark = run_causal_benchmark_forecasts(descriptors, values, config, MAIN_FEATURES)
            if config.source_initialisation_enabled and previous_descriptors is not None and previous_values is not None:
                source_initialised = run_source_initialised_forecasts(previous_descriptors, previous_values, descriptors, values, config, MAIN_FEATURES)
                benchmark = pd.concat((benchmark, source_initialised), ignore_index=True, sort=False)
            benchmark.insert(0, "dataset", str(dataset))
            all_predictions.append(_combined_online_predictions(benchmark, monitored.dual_predictions, local))
            all_metrics.append(forecast_metrics(benchmark, monitored.state_path))
            previous_descriptors, previous_values = descriptors, values
        else:
            all_predictions.append(monitored.dual_predictions.assign(dataset=str(dataset), row_type="dual_timescale", model="SlowFast_RLS"))
    evidence = pd.concat(all_evidence, ignore_index=True)
    states = pd.concat(all_state, ignore_index=True)
    transitions = pd.concat(all_transitions, ignore_index=True) if any(len(part) for part in all_transitions) else pd.DataFrame(columns=("dataset", "input_index", "cycle", "from_state", "to_state", "local_state_id", "state_start_cycle", "reason"))
    predictions = pd.concat(all_predictions, ignore_index=True, sort=False)
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    source_metrics = source_initialisation_metrics(predictions[predictions.row_type == "benchmark_forecast"].copy()) if run_benchmarks else pd.DataFrame()
    change_metrics = pd.concat(all_change_metrics, ignore_index=True)
    ablations = pd.concat([*all_ablations, *all_overlap, horizon_ablation(metrics)], ignore_index=True, sort=False)
    delayed = pd.concat(all_delayed, ignore_index=True) if any(len(part) for part in all_delayed) else pd.DataFrame()
    prefix = {"status": "PASS" if all(item["status"] == "PASS" for item in prefixes.values()) else "FAIL", "datasets": prefixes}
    synthesis = synthetic_acceptance(config)
    gates = gate_decision(metrics, synthesis, prefix, delayed, change_metrics, source_metrics, config) if len(metrics) else {"overall_status": "NOT_EVALUATED"}
    evidence.to_csv(paths["root"] / "v33_change_evidence.csv", index=False)
    states.to_csv(paths["root"] / "v33_state_path.csv", index=False)
    transitions.to_csv(paths["root"] / "v33_state_transition_log.csv", index=False)
    predictions.to_csv(paths["root"] / "v33_online_predictions.csv", index=False)
    metrics.to_csv(paths["root"] / "v33_forecast_metrics.csv", index=False)
    source_metrics.to_csv(paths["root"] / "v33_source_initialisation_metrics.csv", index=False)
    delayed.to_csv(paths["root"] / "v33_delayed_entry_evaluation.csv", index=False)
    ablations.to_csv(paths["root"] / "v33_ablation_results.csv", index=False)
    synthesis.to_csv(paths["root"] / "v33_synthetic_test_results.csv", index=False)
    change_metrics.to_csv(paths["root"] / "v33_change_detection_metrics.csv", index=False)
    _write_json(paths["root"] / "v33_prefix_causality.json", prefix)
    _write_json(paths["root"] / "v33_gate_decision.json", gates)
    write_report(paths["root"] / "gradual_state_monitoring_v33_report.md", config, metrics, change_metrics, synthesis, delayed, gates)
    return {
        "output_dir": str(paths["root"]), "forecast_metrics": metrics, "change_metrics": change_metrics,
        "synthetic": synthesis, "delayed": delayed, "prefix": prefix, "gates": gates,
    }
