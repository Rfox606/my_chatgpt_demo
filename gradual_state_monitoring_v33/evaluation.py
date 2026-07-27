from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import GradualStateMonitoringV33Config
from .data import MAIN_FEATURES, make_synthetic_sequence


def forecast_metrics(forecasts: pd.DataFrame, state_path: pd.DataFrame) -> pd.DataFrame:
    state_keys = state_path.loc[:, ["input_index", "online_state"]].rename(columns={"online_state": "origin_state"})
    scored = forecasts.merge(state_keys, on="input_index", how="left")
    scored = scored[(scored.target_available == 1) & (scored.prediction_available == 1)].copy()
    rows: list[dict[str, object]] = []
    for (dataset, model, horizon), group in scored.groupby(["dataset", "model", "horizon"], sort=True):
        stable = group[group.origin_state == "STABLE"]
        rows.append({
            "dataset": dataset, "model": model, "horizon": int(horizon), "n": int(len(group)),
            "mae": float(group.mae.mean()), "stable_n": int(len(stable)),
            "stable_mae": float(stable.mae.mean()) if len(stable) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["dataset", "horizon", "model"]).reset_index(drop=True)


def change_detection_metrics(evidence: pd.DataFrame, state_path: pd.DataFrame) -> pd.DataFrame:
    merged = evidence.merge(state_path.loc[:, ["input_index", "online_state"]], on="input_index", how="left")
    rows: list[dict[str, object]] = []
    for dataset, group in merged.groupby("dataset", sort=True):
        stable = group[group.online_state == "STABLE"]
        transition = group[group.online_state == "TRANSITION"]
        states = group.online_state.to_numpy(str)
        switches = int(np.sum(states[1:] != states[:-1])) if len(states) > 1 else 0
        run_lengths: list[int] = []
        run = 0
        for state in states:
            if state == "TRANSITION":
                run += 1
            elif run:
                run_lengths.append(run); run = 0
        if run:
            run_lengths.append(run)
        rows.append({
            "dataset": dataset,
            "stable_change_evidence_mean": float(stable.change_evidence.mean()) if len(stable) else np.nan,
            "transition_change_evidence_mean": float(transition.change_evidence.mean()) if len(transition) else np.nan,
            "stable_prediction_surprise_mean": float(stable.prediction_surprise.mean()) if len(stable) else np.nan,
            "transition_prediction_surprise_mean": float(transition.prediction_surprise.mean()) if len(transition) else np.nan,
            "transition_windows": int(len(transition)),
            "transition_episode_count": int(len(run_lengths)),
            "mean_transition_width": float(np.mean(run_lengths)) if run_lengths else 0.0,
            "max_transition_width": int(max(run_lengths)) if run_lengths else 0,
            "new_stable_windows": int((group.online_state == "NEW_STABLE").sum()),
            "state_switches": switches,
        })
    return pd.DataFrame(rows)


def evidence_ablation(evidence: pd.DataFrame, state_path: pd.DataFrame, config: GradualStateMonitoringV33Config) -> pd.DataFrame:
    merged = evidence.merge(state_path.loc[:, ["input_index", "online_state"]], on="input_index", how="left")
    variants = {
        "full": config.evidence_weights,
        "without_prediction_surprise": {**config.evidence_weights, "prediction_surprise": 0.0},
        "without_distribution_shift": {**config.evidence_weights, "distribution_shift": 0.0},
        "without_drift_velocity": {**config.evidence_weights, "drift_velocity": 0.0},
    }
    rows: list[dict[str, object]] = []
    channels = ("prediction_surprise", "distribution_shift", "drift_velocity")
    for name, weights in variants.items():
        denominator = sum(weights.values())
        score = sum(weights[channel] * merged[channel] for channel in channels) / denominator if denominator else pd.Series(0.0, index=merged.index)
        for dataset, indices in merged.groupby("dataset", sort=True).groups.items():
            group = merged.loc[indices]
            local_score = score.loc[indices]
            stable = local_score[group.online_state == "STABLE"]
            transition = local_score[group.online_state == "TRANSITION"]
            rows.append({
                "dataset": dataset, "ablation": name, "prediction_weight": weights["prediction_surprise"],
                "shift_weight": weights["distribution_shift"], "velocity_weight": weights["drift_velocity"],
                "stable_mean": float(stable.mean()) if len(stable) else np.nan,
                "transition_mean": float(transition.mean()) if len(transition) else np.nan,
                "transition_minus_stable": float(transition.mean() - stable.mean()) if len(stable) and len(transition) else np.nan,
            })
    return pd.DataFrame(rows)


def horizon_ablation(metrics: pd.DataFrame) -> pd.DataFrame:
    """Expose all horizon/model comparisons in the shared ablation artifact."""
    if metrics.empty:
        return pd.DataFrame()
    output = metrics.copy()
    output.insert(1, "ablation", output.horizon.map(lambda value: f"forecast_horizon_{int(value)}"))
    output.insert(1, "ablation_type", "forecast_horizon")
    return output


def source_initialisation_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Compare source-initialised and scratch learned models over exactly the same early origins."""
    rows: list[dict[str, object]] = []
    if forecasts.empty or not forecasts.model.astype(str).str.startswith("SourceInit_").any():
        return pd.DataFrame(columns=("dataset", "horizon", "model", "scratch_model", "n", "source_init_mae", "scratch_mae", "relative_improvement", "source_init_better"))
    for (dataset, horizon, model), source in forecasts[forecasts.model.astype(str).str.startswith("SourceInit_")].groupby(["dataset", "horizon", "model"], sort=True):
        scratch_model = str(model).removeprefix("SourceInit_")
        origins = source.input_index.unique()
        scratch = forecasts[(forecasts.dataset == dataset) & (forecasts.horizon == horizon) & (forecasts.model == scratch_model) & (forecasts.input_index.isin(origins))]
        if not len(scratch):
            continue
        source_mae = float(source.mae.mean())
        scratch_mae = float(scratch.mae.mean())
        rows.append({
            "dataset": dataset, "horizon": int(horizon), "model": model, "scratch_model": scratch_model,
            "n": int(len(source)), "source_init_mae": source_mae, "scratch_mae": scratch_mae,
            "relative_improvement": float((scratch_mae - source_mae) / max(scratch_mae, 1e-12)),
            "source_init_better": int(source_mae < scratch_mae),
        })
    return pd.DataFrame(rows)


def window_overlap_ablation(frame: pd.DataFrame, full_state: pd.DataFrame, config: GradualStateMonitoringV33Config) -> pd.DataFrame:
    """Subsample existing 20-cycle windows to evaluate 75%, 50%, and 0% overlap causally."""
    from .pipeline import run_monitor

    rows: list[dict[str, object]] = []
    width = max(config.base_window_width, 1)
    for multiplier in config.overlap_stride_multipliers:
        stride = config.base_window_stride * multiplier
        overlap = max(0.0, 1.0 - stride / width)
        if multiplier == 1:
            observed = full_state.loc[:, ["cycle", "online_state"]].copy()
        else:
            observed = run_monitor(frame.iloc[::multiplier].reset_index(drop=True), config, include_dual=False).state_path.loc[:, ["cycle", "online_state"]]
        common = full_state.loc[:, ["cycle", "online_state"]].merge(observed, on="cycle", suffixes=("_full", "_ablated"), how="inner")
        warmup_index = min(len(frame) - 1, config.history_windows * multiplier)
        common = common[common.cycle >= float(frame.center_cycle.iloc[warmup_index])]
        agreement = float((common.online_state_full == common.online_state_ablated).mean()) if len(common) else np.nan
        rows.append({
            "dataset": str(frame.dataset.iloc[0]), "ablation_type": "window_overlap", "ablation": f"stride_x{multiplier}",
            "effective_window_stride": stride, "window_overlap_ratio": overlap,
            "common_windows_after_warmup": int(len(common)), "state_agreement_vs_base": agreement,
        })
    return pd.DataFrame(rows)


def delayed_entry_evaluation(frame: pd.DataFrame, full_state: pd.DataFrame, config: GradualStateMonitoringV33Config) -> pd.DataFrame:
    from .pipeline import run_monitor

    rows: list[dict[str, object]] = []
    for offset in config.delayed_entry_offsets:
        if len(frame) <= offset + config.delayed_entry_warmup:
            continue
        delayed = run_monitor(frame.iloc[offset:].reset_index(drop=True), config, include_dual=False)
        common = full_state.loc[:, ["cycle", "online_state"]].merge(
            delayed.state_path.loc[:, ["cycle", "online_state"]], on="cycle", suffixes=("_full", "_delayed"), how="inner",
        )
        common = common[common.cycle >= float(frame.center_cycle.iloc[offset + config.delayed_entry_warmup])]
        agreement = float((common.online_state_full == common.online_state_delayed).mean()) if len(common) else np.nan
        final = common.tail(min(100, len(common)))
        final_agreement = float((final.online_state_full == final.online_state_delayed).mean()) if len(final) else np.nan
        rows.append({
            "dataset": str(frame.dataset.iloc[0]), "entry_offset": offset, "common_future_windows": int(len(common)),
            "agreement_after_warmup": agreement, "agreement_final_100": final_agreement,
            "converged": int(np.isfinite(final_agreement) and final_agreement >= config.delayed_convergence_threshold),
        })
    return pd.DataFrame(rows)


def prefix_causality_evaluation(frame: pd.DataFrame, config: GradualStateMonitoringV33Config, full: object | None = None) -> dict[str, Any]:
    from .pipeline import run_monitor

    full = full if full is not None else run_monitor(frame, config)
    tail = frame.tail(min(48, len(frame))).copy().reset_index(drop=True)
    tail.loc[:, MAIN_FEATURES] += np.linspace(0.5, 1.1, len(MAIN_FEATURES))
    tail["window_index"] = np.arange(int(frame.window_index.max()) + 1, int(frame.window_index.max()) + 1 + len(tail))
    gap = float(np.median(np.diff(frame.center_cycle.to_numpy(float)))) if len(frame) > 1 else 1.0
    tail["center_cycle"] = frame.center_cycle.iloc[-1] + gap * np.arange(1, len(tail) + 1)
    extended = run_monitor(pd.concat((frame, tail), ignore_index=True), config)
    left = full.state_path.merge(extended.state_path, on="input_index", suffixes=("_full", "_extended"), how="inner")
    state_equal = bool((left.online_state_full == left.online_state_extended).all())
    numeric = ("change_evidence", "prediction_surprise", "distribution_shift", "drift_velocity")
    evidence_left = full.evidence.merge(extended.evidence, on="input_index", suffixes=("_full", "_extended"), how="inner")
    maximum = 0.0
    for name in numeric:
        values = np.abs(evidence_left[f"{name}_full"].to_numpy(float) - evidence_left[f"{name}_extended"].to_numpy(float))
        if np.isfinite(values).any():
            maximum = max(maximum, float(np.nanmax(values)))
    distance = np.abs(left.distance_from_local_anchor_full.to_numpy(float) - left.distance_from_local_anchor_extended.to_numpy(float))
    if np.isfinite(distance).any():
        maximum = max(maximum, float(np.nanmax(distance)))
    return {
        "status": "PASS" if state_equal and maximum <= 1e-12 else "FAIL",
        "rows_compared": int(len(left)), "state_path_exactly_equal": state_equal,
        "max_abs_numeric_difference": maximum,
    }


def synthetic_acceptance(config: GradualStateMonitoringV33Config) -> pd.DataFrame:
    from .pipeline import run_monitor

    rows: list[dict[str, object]] = []
    for index, kind in enumerate(("stable", "step", "ramp", "spike")):
        result = run_monitor(make_synthetic_sequence(kind, seed=config.random_seed + index), config)
        states = result.state_path.online_state.to_numpy(str)
        transitions = int((states == "TRANSITION").sum())
        new_stable = int((states == "NEW_STABLE").sum())
        temporary = int((states == "TEMPORARY_DISTURBANCE").sum())
        max_transition = 0
        run = 0
        for state in states:
            if state == "TRANSITION":
                run += 1; max_transition = max(max_transition, run)
            else:
                run = 0
        if kind == "stable":
            passed = transitions <= config.stable_false_transition_limit
        elif kind == "step":
            passed = transitions > 0 and new_stable > 0
        elif kind == "ramp":
            passed = max_transition >= config.transition_persistence * 3
        else:
            passed = temporary > 0 and new_stable == 0
        rows.append({
            "synthetic_case": kind, "transition_windows": transitions, "max_transition_width": max_transition,
            "new_stable_windows": new_stable, "temporary_disturbance_windows": temporary,
            "status": "PASS" if passed else "FAIL",
        })
    return pd.DataFrame(rows)


def gate_decision(metrics: pd.DataFrame, synthesis: pd.DataFrame, prefix: dict[str, Any], delayed: pd.DataFrame, state_metrics: pd.DataFrame, source_metrics: pd.DataFrame, config: GradualStateMonitoringV33Config) -> dict[str, Any]:
    medium_long = metrics[metrics.horizon >= 100]
    wins: list[bool] = []
    for _, group in medium_long.groupby(["dataset", "horizon"]):
        persistence = group.loc[group.model == "Persistence", "mae"]
        learned = group.loc[group.model.isin(("Ridge", "RBF_Ridge")), "mae"]
        if len(persistence) and len(learned):
            wins.append(float(learned.min()) < float(persistence.iloc[0]))
    stable_controlled = bool(np.isfinite(metrics.stable_mae).any())
    gate_a = bool(any(wins) and stable_controlled)
    source_benefit = bool(len(source_metrics) and (source_metrics.source_init_better == 1).any())
    synth = dict(zip(synthesis.synthetic_case, synthesis.status))
    real_explainable = bool((state_metrics.state_switches <= config.real_switch_limit).all()) if len(state_metrics) else False
    gate_b = bool(synth.get("ramp") == "PASS" and synth.get("stable") == "PASS" and synth.get("spike") == "PASS" and real_explainable)
    delayed_converged = bool(len(delayed) and (delayed.converged == 1).all())
    gate_c = bool(synth.get("step") == "PASS" and prefix["status"] == "PASS" and delayed_converged)
    return {
        "gate_A_stable_dynamics": {
            "status": "PASS" if gate_a else "FAIL", "mid_or_long_learning_model_beats_persistence": bool(any(wins)),
            "stable_interval_error_available": stable_controlled,
            "source_initialisation_early_benefit": source_benefit,
        },
        "gate_B_gradual_detection": {
            "status": "PASS" if gate_b else "FAIL", "synthetic": synth,
            "real_state_switches_within_limit": real_explainable,
        },
        "gate_C_new_state_and_online_consistency": {
            "status": "PASS" if gate_c else "FAIL", "prefix_causality": prefix,
            "delayed_entry_converged": delayed_converged,
        },
        "overall_status": "PASS" if gate_a and gate_b and gate_c else "FAIL",
        "shared_cross_experiment_dynamics_supported": False,
        "shared_cross_experiment_dynamics_reason": "This V3.3 implementation deliberately creates local states only; it does not test a shared cross-experiment dynamic model.",
    }
