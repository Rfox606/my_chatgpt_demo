from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gradual_state_monitoring_v33.data import MAIN_FEATURES, load_window_data, make_synthetic_sequence
from gradual_state_monitoring_v33.evaluation import forecast_metrics
from gradual_state_monitoring_v33.features import build_causal_descriptors
from gradual_state_monitoring_v33.forecasting import run_causal_benchmark_forecasts

from .config import GradualStateMonitoringV331Config
from .engine import (
    NEW_STABLE, STABLE, TEMPORARY_DISTURBANCE, TRANSITION, LocalStateLibrary,
    MonitorResult, run_monitor,
)
from .report import write_report


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


def _transition_runs(state: pd.DataFrame) -> list[tuple[float, float]]:
    mask = state.online_state.to_numpy(str) == TRANSITION
    cycle = state.cycle.to_numpy(float)
    runs: list[tuple[float, float]] = []
    start: float | None = None
    for active, current in zip(mask, cycle):
        if active and start is None:
            start = current
        elif not active and start is not None:
            runs.append((start, previous)); start = None
        previous = current
    if start is not None:
        runs.append((start, cycle[-1]))
    return runs


def _safe_mae(left: pd.Series, right: pd.Series) -> float:
    values = np.abs(left.to_numpy(float) - right.to_numpy(float))
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def _summarise_path(result: MonitorResult, dataset: str, variant: str, full_state: pd.DataFrame | None = None) -> dict[str, object]:
    state = result.state_path
    values = state.online_state.to_numpy(str)
    switches = int(np.sum(values[1:] != values[:-1])) if len(values) > 1 else 0
    runs = _transition_runs(state)
    agreement = np.nan
    if full_state is not None:
        merged = state.loc[:, ["cycle", "online_state"]].merge(full_state.loc[:, ["cycle", "online_state"]], on="cycle", suffixes=("_variant", "_full"), how="inner")
        agreement = float((merged.online_state_variant == merged.online_state_full).mean()) if len(merged) else np.nan
    return {
        "dataset": dataset, "variant": variant, "windows": int(len(state)),
        "transition_fraction": float((values == TRANSITION).mean()),
        "transition_episode_count": len(runs), "new_state_count": int((result.transitions.to_state == NEW_STABLE).sum()) if len(result.transitions) else 0,
        "temporary_disturbance_count": int((result.transitions.to_state == TEMPORARY_DISTURBANCE).sum()) if len(result.transitions) else 0,
        "state_switches": switches, "local_state_count": int(len(result.local_state_library)),
        "state_agreement_with_full": agreement,
        "dual_model_enabled": 1,
    }


def _transition_iou(left: pd.DataFrame, right: pd.DataFrame) -> tuple[float, float, float]:
    merged = left.loc[:, ["cycle", "online_state"]].merge(right.loc[:, ["cycle", "online_state"]], on="cycle", suffixes=("_full", "_delayed"), how="inner")
    full = merged.online_state_full.to_numpy(str) == TRANSITION
    delayed = merged.online_state_delayed.to_numpy(str) == TRANSITION
    union = int(np.sum(full | delayed))
    iou = float(np.sum(full & delayed) / union) if union else 1.0
    full_runs = _transition_runs(left); delayed_runs = _transition_runs(right)
    full_start, full_end = (full_runs[0][0], full_runs[-1][1]) if full_runs else (np.nan, np.nan)
    delayed_start, delayed_end = (delayed_runs[0][0], delayed_runs[-1][1]) if delayed_runs else (np.nan, np.nan)
    return iou, float(abs(full_start - delayed_start)) if np.isfinite(full_start) and np.isfinite(delayed_start) else np.nan, float(abs(full_end - delayed_end)) if np.isfinite(full_end) and np.isfinite(delayed_end) else np.nan


def delayed_entry_evaluation(frame: pd.DataFrame, full: MonitorResult, config: GradualStateMonitoringV331Config, *, variant: str) -> pd.DataFrame:
    """Replay each late entry with Slow/Fast RLS enabled, then compare all requested signals."""
    rows: list[dict[str, object]] = []
    for offset in config.delayed_entry_offsets:
        if len(frame) <= offset + config.delayed_entry_warmup:
            continue
        delayed = run_monitor(frame.iloc[offset:].reset_index(drop=True), config, descriptor_scaler_mode="frozen_descriptor_scaler", include_dual=True)
        evidence = full.evidence.merge(delayed.evidence, on="cycle", suffixes=("_full", "_delayed"), how="inner")
        dual = full.dual_predictions.merge(delayed.dual_predictions, on="cycle", suffixes=("_full", "_delayed"), how="inner")
        after = evidence[evidence.cycle >= float(frame.center_cycle.iloc[offset + config.delayed_entry_warmup])]
        after_dual = dual[dual.cycle >= float(frame.center_cycle.iloc[offset + config.delayed_entry_warmup])]
        full_state = full.state_path[full.state_path.cycle.isin(after.cycle)]
        delayed_state = delayed.state_path[delayed.state_path.cycle.isin(after.cycle)]
        aligned_state = full_state.loc[:, ["cycle", "online_state"]].merge(delayed_state.loc[:, ["cycle", "online_state"]], on="cycle", suffixes=("_full", "_delayed"), how="inner")
        iou, start_difference, end_difference = _transition_iou(full_state, delayed_state)
        rows.append({
            "dataset": str(frame.dataset.iloc[0]), "variant": variant, "entry_offset": offset,
            "common_future_windows": int(len(after)), "dual_model_enabled": 1,
            "state_agreement": float((aligned_state.online_state_full == aligned_state.online_state_delayed).mean()) if len(aligned_state) else np.nan,
            "evidence_mae": _safe_mae(after.change_evidence_full, after.change_evidence_delayed),
            "prediction_surprise_mae": _safe_mae(after.prediction_surprise_full, after.prediction_surprise_delayed),
            "distribution_shift_mae": _safe_mae(after.distribution_shift_full, after.distribution_shift_delayed),
            "drift_velocity_mae": _safe_mae(after.drift_velocity_full, after.drift_velocity_delayed),
            "slow_prediction_error_mae": _safe_mae(after_dual.slow_prediction_error_full, after_dual.slow_prediction_error_delayed),
            "fast_prediction_error_mae": _safe_mae(after_dual.fast_prediction_error_full, after_dual.fast_prediction_error_delayed),
            "fast_slow_prediction_difference_mae": _safe_mae(after_dual.fast_slow_prediction_difference_full, after_dual.fast_slow_prediction_difference_delayed),
            "transition_interval_iou": iou, "transition_start_difference": start_difference,
            "transition_end_difference": end_difference,
            "local_state_count_difference": abs(len(full.local_state_library) - len(delayed.local_state_library)),
        })
    return pd.DataFrame(rows)


def prefix_causality(frame: pd.DataFrame, full: MonitorResult, config: GradualStateMonitoringV331Config) -> dict[str, object]:
    tail = frame.tail(min(48, len(frame))).copy().reset_index(drop=True)
    tail.loc[:, MAIN_FEATURES] += np.linspace(.5, 1.1, len(MAIN_FEATURES))
    tail["window_index"] = np.arange(int(frame.window_index.max()) + 1, int(frame.window_index.max()) + 1 + len(tail))
    gap = float(np.median(np.diff(frame.center_cycle.to_numpy(float)))) if len(frame) > 1 else 1.0
    tail["center_cycle"] = frame.center_cycle.iloc[-1] + gap * np.arange(1, len(tail) + 1)
    replay = run_monitor(pd.concat((frame, tail), ignore_index=True), config, descriptor_scaler_mode="frozen_descriptor_scaler", include_dual=True)
    state = full.state_path.merge(replay.state_path, on="input_index", suffixes=("_full", "_replay"), how="inner")
    evidence = full.evidence.merge(replay.evidence, on="input_index", suffixes=("_full", "_replay"), how="inner")
    dual = full.dual_predictions.merge(replay.dual_predictions, on="input_index", suffixes=("_full", "_replay"), how="inner")
    max_difference = 0.0
    for name in ("change_evidence", "prediction_surprise", "distribution_shift", "drift_velocity", "model_divergence"):
        values = np.abs(evidence[f"{name}_full"].to_numpy(float) - evidence[f"{name}_replay"].to_numpy(float))
        if np.isfinite(values).any():
            max_difference = max(max_difference, float(np.nanmax(values)))
    for name in ("slow_prediction_error", "fast_prediction_error", "fast_slow_prediction_difference"):
        values = np.abs(dual[f"{name}_full"].to_numpy(float) - dual[f"{name}_replay"].to_numpy(float))
        if np.isfinite(values).any():
            max_difference = max(max_difference, float(np.nanmax(values)))
    exact_state = bool((state.online_state_full == state.online_state_replay).all())
    return {"status": "PASS" if exact_state and max_difference <= 1e-12 else "FAIL", "rows_compared": int(len(state)),
            "state_path_exactly_equal": exact_state, "max_abs_numeric_difference": max_difference}


def synthetic_acceptance(config: GradualStateMonitoringV331Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for number, kind in enumerate(("stable", "step", "ramp", "spike")):
        result = run_monitor(make_synthetic_sequence(kind, seed=config.random_seed + number), config, descriptor_scaler_mode="frozen_descriptor_scaler", include_dual=True)
        state = result.state_path.online_state.to_numpy(str)
        transition = int((state == TRANSITION).sum())
        maximum = max([end - start + 1 for start, end in _runs_indices(state == TRANSITION)] or [0])
        new = int((state == NEW_STABLE).sum()); temporary = int((state == TEMPORARY_DISTURBANCE).sum())
        passed = (
            transition <= 1 if kind == "stable" else
            transition > 0 and new > 0 if kind == "step" else
            maximum >= config.transition_persistence * 3 if kind == "ramp" else
            temporary > 0 and new == 0
        )
        rows.append({"synthetic_case": kind, "transition_windows": transition, "max_transition_width": maximum,
                     "new_stable_windows": new, "temporary_disturbance_windows": temporary,
                     "status": "PASS" if passed else "FAIL"})
    return pd.DataFrame(rows)


def _runs_indices(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []; start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        if not active and start is not None:
            runs.append((start, index - 1)); start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def state_library_validation(config: GradualStateMonitoringV331Config) -> dict[str, object]:
    library = LocalStateLibrary(config)
    scale = np.ones(2)
    first = library.initialise(np.array((0.0, 0.0)), scale, 0.0)
    second, _, _, _ = library.confirm(np.array((5.0, 5.0)), scale, 1.0)
    third, _, _, _ = library.confirm(np.array((12.0, 12.0)), scale, 2.0)
    reused, action, _, _ = library.confirm(np.array((5.1, 5.0)), scale, 3.0)
    return {"status": "PASS" if (first, second, third, reused, action) == (1, 2, 3, 2, "reused") else "FAIL",
            "path": [first, second, third, reused], "reuse_action": action}


def window_overlap_evaluation(frame: pd.DataFrame, full: MonitorResult, config: GradualStateMonitoringV331Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for multiplier in config.overlap_stride_multipliers:
        if multiplier == 1:
            result = full
        else:
            result = run_monitor(frame.iloc[::multiplier].reset_index(drop=True), config, descriptor_scaler_mode="frozen_descriptor_scaler", include_dual=True)
        common = full.state_path.loc[:, ["cycle", "online_state"]].merge(result.state_path.loc[:, ["cycle", "online_state"]], on="cycle", suffixes=("_full", "_overlap"), how="inner")
        warmup = min(len(frame) - 1, config.descriptor_scaler_calibration_windows * multiplier)
        common = common[common.cycle >= float(frame.center_cycle.iloc[warmup])]
        rows.append({"dataset": str(frame.dataset.iloc[0]), "stride_multiplier": multiplier,
                     "effective_window_stride": config.base_window_stride * multiplier,
                     "window_overlap_ratio": max(0.0, 1.0 - config.base_window_stride * multiplier / config.base_window_width),
                     "common_windows_after_warmup": int(len(common)),
                     "state_agreement": float((common.online_state_full == common.online_state_overlap).mean()) if len(common) else np.nan,
                     "dual_model_enabled": 1})
    return pd.DataFrame(rows)


def event_boundary_audit(transitions: pd.DataFrame, config: GradualStateMonitoringV331Config) -> pd.DataFrame:
    """Post-hoc only: mark event proximity to mapping/data-group segment boundaries without filtering events."""
    mapping = json.loads(Path(config.cycle_mapping_path).read_text(encoding="utf-8"))
    boundaries: dict[str, list[float]] = {}
    for segment in mapping["segments"]:
        boundaries.setdefault(segment["dataset"], []).extend((float(segment["effective_start"]), float(segment["effective_end"])))
    rows: list[dict[str, object]] = []
    for _, event in transitions.iterrows():
        values = boundaries.get(str(event.dataset), [])
        nearest = min(values, key=lambda value: abs(float(event.cycle) - value)) if values else np.nan
        distance = abs(float(event.cycle) - nearest) if np.isfinite(nearest) else np.nan
        rows.append({"dataset": event.dataset, "event_cycle": float(event.cycle), "event_type": f"{event.from_state}_to_{event.to_state}",
                     "nearest_group_boundary_cycle": nearest, "nearest_group_boundary_distance": distance,
                     "within_boundary_guard": int(np.isfinite(distance) and distance <= config.boundary_guard_cycles),
                     "audit_only": 1})
    return pd.DataFrame(rows)


def _static_forecasts(frame: pd.DataFrame, state: pd.DataFrame, config: GradualStateMonitoringV331Config) -> pd.DataFrame:
    values = frame.loc[:, MAIN_FEATURES].to_numpy(float)
    forecast = run_causal_benchmark_forecasts(build_causal_descriptors(values), values, config, MAIN_FEATURES)
    forecast.insert(0, "dataset", str(frame.dataset.iloc[0]))
    return forecast_metrics(forecast, state)


def run_pipeline(config: GradualStateMonitoringV331Config) -> dict[str, Any]:
    paths = config.paths(); root = paths["root"]
    _write_json(root / "v331_config.json", config.jsonable())
    source = load_window_data(config.input_path)
    canonical: dict[str, MonitorResult] = {}
    all_evidence: list[pd.DataFrame] = []; all_state: list[pd.DataFrame] = []; all_transitions: list[pd.DataFrame] = []
    all_library: list[pd.DataFrame] = []; all_reuse: list[pd.DataFrame] = []
    scaler_rows: list[dict[str, object]] = []; forecast_parts: list[pd.DataFrame] = []
    ablation_rows: list[dict[str, object]] = []; ablation_transitions: list[pd.DataFrame] = []
    delayed_parts: list[pd.DataFrame] = []; overlap_parts: list[pd.DataFrame] = []; divergence_rows: list[dict[str, object]] = []
    prefix: dict[str, object] = {}; library_checks: dict[str, object] = {}
    for dataset, grouped in source.groupby("dataset", sort=True):
        frame = grouped.reset_index(drop=True)
        full = run_monitor(frame, config, descriptor_scaler_mode="frozen_descriptor_scaler", include_dual=True)
        canonical[str(dataset)] = full
        all_evidence.append(full.evidence); all_state.append(full.state_path); all_transitions.append(full.transitions)
        all_library.append(full.local_state_library); all_reuse.append(full.state_reuse_log)
        prefix[str(dataset)] = prefix_causality(frame, full, config)
        library_checks[str(dataset)] = state_library_validation(config)
        forecast_parts.append(_static_forecasts(frame, full.state_path, config))
        rolling = run_monitor(frame, config, descriptor_scaler_mode="rolling_descriptor_scaler_v33", include_dual=True)
        for mode, result in (("frozen_descriptor_scaler", full), ("rolling_descriptor_scaler_v33", rolling)):
            valid = result.dual_predictions
            scaler_rows.append({"dataset": dataset, "scaler_mode": mode, "slow_mae": float(valid.slow_prediction_error.mean(skipna=True)),
                                "fast_mae": float(valid.fast_prediction_error.mean(skipna=True)),
                                "state_switches": _summarise_path(result, str(dataset), mode)["state_switches"],
                                "transition_fraction": _summarise_path(result, str(dataset), mode)["transition_fraction"],
                                "state_agreement_with_frozen": _summarise_path(result, str(dataset), mode, full.state_path)["state_agreement_with_full"]})
        variants = {
            "full": config.evidence_weights,
            "without_prediction_surprise": {**config.evidence_weights, "prediction_surprise": 0.0},
            "without_distribution_shift": {**config.evidence_weights, "distribution_shift": 0.0},
            "without_drift_velocity": {**config.evidence_weights, "drift_velocity": 0.0},
        }
        for variant, weights in variants.items():
            result = full if variant == "full" else run_monitor(frame, replace(config, evidence_weights=weights), descriptor_scaler_mode="frozen_descriptor_scaler", include_dual=True)
            delayed = delayed_entry_evaluation(frame, result, replace(config, evidence_weights=weights), variant=variant)
            delayed_parts.append(delayed)
            summary = _summarise_path(result, str(dataset), variant, full.state_path)
            summary["delayed_state_agreement_mean"] = float(delayed.state_agreement.mean()) if len(delayed) else np.nan
            summary["delayed_transition_iou_mean"] = float(delayed.transition_interval_iou.mean()) if len(delayed) else np.nan
            ablation_rows.append(summary)
            if len(result.transitions):
                events = result.transitions.copy(); events.insert(2, "variant", variant); ablation_transitions.append(events)
        with_divergence = run_monitor(frame, config, descriptor_scaler_mode="frozen_descriptor_scaler", include_model_divergence=True, include_dual=True)
        for name, result in (("without_model_divergence", full), ("with_model_divergence", with_divergence)):
            summary = _summarise_path(result, str(dataset), name, full.state_path)
            divergence_rows.append(summary)
        overlap_parts.append(window_overlap_evaluation(frame, full, config))
    evidence = pd.concat(all_evidence, ignore_index=True); states = pd.concat(all_state, ignore_index=True)
    transitions = pd.concat(all_transitions, ignore_index=True) if any(len(item) for item in all_transitions) else pd.DataFrame()
    library = pd.concat(all_library, ignore_index=True); reuse = pd.concat(all_reuse, ignore_index=True) if any(len(item) for item in all_reuse) else pd.DataFrame()
    forecast = pd.concat(forecast_parts, ignore_index=True); scaler = pd.DataFrame(scaler_rows)
    ablation = pd.DataFrame(ablation_rows); ablation_log = pd.concat(ablation_transitions, ignore_index=True) if ablation_transitions else pd.DataFrame()
    delayed = pd.concat(delayed_parts, ignore_index=True); overlap = pd.concat(overlap_parts, ignore_index=True)
    divergence = pd.DataFrame(divergence_rows); synthetic = synthetic_acceptance(config)
    event_audit = event_boundary_audit(transitions, config)
    prefix_status = {"status": "PASS" if all(value["status"] == "PASS" for value in prefix.values()) else "FAIL", "datasets": prefix}
    gate = gate_decision(forecast, synthetic, ablation, overlap, delayed, prefix_status, library_checks, config)
    evidence.to_csv(root / "v331_change_evidence.csv", index=False); states.to_csv(root / "v331_state_path.csv", index=False)
    transitions.to_csv(root / "v331_state_transition_log.csv", index=False); library.to_csv(root / "v331_local_state_library.csv", index=False)
    reuse.to_csv(root / "v331_state_reuse_log.csv", index=False); delayed.to_csv(root / "v331_delayed_entry_evaluation.csv", index=False)
    ablation.to_csv(root / "v331_evidence_ablation_state_metrics.csv", index=False); ablation_log.to_csv(root / "v331_evidence_ablation_transition_log.csv", index=False)
    overlap.to_csv(root / "v331_window_overlap_evaluation.csv", index=False); event_audit.to_csv(root / "v331_event_boundary_audit.csv", index=False)
    synthetic.to_csv(root / "v331_synthetic_test_results.csv", index=False); forecast.to_csv(root / "v331_forecast_metrics.csv", index=False)
    scaler.to_csv(root / "v331_descriptor_scaler_ablation.csv", index=False); divergence.to_csv(root / "v331_model_divergence_ablation.csv", index=False)
    _write_json(root / "v331_prefix_causality.json", prefix_status); _write_json(root / "v331_state_library_validation.json", library_checks)
    _write_json(root / "v331_gate_decision.json", gate)
    write_report(root / "gradual_state_monitoring_v331_report.md", config, forecast, scaler, ablation, overlap, delayed, synthetic, divergence, event_audit, gate)
    return {"output_dir": str(root), "gate": gate, "forecast": forecast, "ablation": ablation, "delayed": delayed}


def gate_decision(forecast: pd.DataFrame, synthetic: pd.DataFrame, ablation: pd.DataFrame, overlap: pd.DataFrame, delayed: pd.DataFrame,
                  prefix: dict[str, object], library_checks: dict[str, object], config: GradualStateMonitoringV331Config) -> dict[str, object]:
    medium_long = forecast[forecast.horizon >= 100]
    wins = []
    for _, group in medium_long.groupby(["dataset", "horizon"]):
        persistence = group.loc[group.model == "Persistence", "mae"]
        learning = group.loc[group.model.isin(("Ridge", "RBF_Ridge")), "mae"]
        if len(persistence) and len(learning):
            wins.append(float(learning.min()) < float(persistence.iloc[0]))
    gate_a = bool(any(wins) and np.isfinite(forecast.stable_mae).any())
    synthetic_map = dict(zip(synthetic.synthetic_case, synthetic.status))
    b1 = bool(all(synthetic_map.get(name) == "PASS" for name in ("stable", "step", "ramp", "spike")))
    full = ablation[ablation.variant == "full"].copy()
    overlap50 = overlap[overlap.window_overlap_ratio == .50].set_index("dataset").state_agreement.to_dict()
    overlap0 = overlap[overlap.window_overlap_ratio == 0.0].set_index("dataset").state_agreement.to_dict()
    b2_rows = []
    for dataset, row in full.set_index("dataset").iterrows():
        stability = bool(np.isfinite(ablation[ablation.dataset == dataset].state_agreement_with_full).all())
        result = {
            "dataset": dataset, "transition_fraction": float(row.transition_fraction), "transition_fraction_pass": bool(row.transition_fraction < config.b2_transition_fraction_limit),
            "overlap_50_state_agreement": float(overlap50.get(dataset, np.nan)), "overlap_50_pass": bool(overlap50.get(dataset, -np.inf) >= config.b2_overlap50_agreement_min),
            "overlap_0_state_agreement": float(overlap0.get(dataset, np.nan)), "overlap_0_pass": bool(overlap0.get(dataset, -np.inf) >= config.b2_overlap0_agreement_min),
            "evidence_ablation_stability": stability,
        }
        result["status"] = "PASS" if all((result["transition_fraction_pass"], result["overlap_50_pass"], result["overlap_0_pass"], stability)) else "FAIL"
        b2_rows.append(result)
    b2 = bool(b2_rows and all(row["status"] == "PASS" for row in b2_rows))
    c_rows = []
    for dataset, rows in delayed.groupby("dataset"):
        pass_state = bool((rows.state_agreement >= config.delayed_state_agreement_threshold).all())
        pass_iou = bool((rows.transition_interval_iou >= config.delayed_transition_iou_threshold).all())
        pass_count = bool((rows.local_state_count_difference <= config.delayed_local_state_difference_limit).all())
        c_rows.append({"dataset": dataset, "state_agreement_pass": pass_state, "transition_iou_pass": pass_iou,
                       "local_state_count_pass": pass_count, "dual_model_enabled": bool((rows.dual_model_enabled == 1).all())})
    c = bool(prefix["status"] == "PASS" and c_rows and all(all(row[key] for key in ("state_agreement_pass", "transition_iou_pass", "local_state_count_pass", "dual_model_enabled")) for row in c_rows)
             and all(item["status"] == "PASS" for item in library_checks.values()))
    return {
        "gate_A_stable_dynamics": {"status": "PASS" if gate_a else "FAIL", "mid_or_long_learning_beats_persistence": bool(any(wins)),
                                    "stable_interval_error_available": bool(np.isfinite(forecast.stable_mae).any())},
        "gate_B1_synthetic_gradual_detection": {"status": "PASS" if b1 else "FAIL", "synthetic": synthetic_map},
        "gate_B2_real_data_stability": {"status": "PASS" if b2 else "FAIL", "datasets": b2_rows},
        "gate_B_gradual_detection": {"status": "PASS" if b1 and b2 else "FAIL"},
        "gate_C_new_state_and_online_consistency": {"status": "PASS" if c else "FAIL", "prefix_causality": prefix,
                                                       "delayed_entry": c_rows, "state_library_validation": library_checks},
        "overall_status": "PASS" if gate_a and b1 and b2 and c else "FAIL",
        "posthoc_interpretation_only": True,
        "shared_cross_experiment_dynamics_supported": False,
    }
