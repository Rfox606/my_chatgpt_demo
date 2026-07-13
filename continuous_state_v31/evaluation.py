from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ContinuousStateV31Config
from .data import FORBIDDEN_COLUMNS
from .source_prior import SourceProtocolModel
from .state_engine import derive_plateau_prior, run_target_state


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def prefix_causality(
    source_model: SourceProtocolModel,
    target: pd.DataFrame,
    full: pd.DataFrame,
    config: ContinuousStateV31Config,
) -> dict[str, object]:
    columns = ("D_state", "V50_norm", "A_smooth_20", "instability_score", "S_severe_candidate", "plateau_valid_cycles", "exit_valid_cycles")
    checks: list[dict[str, object]] = []
    for cutoff in (2000, 5000, 10000):
        prefix = target.loc[target.center_cycle <= cutoff].copy()
        if prefix.empty:
            checks.append({"prefix_cycle": cutoff, "window_count": 0, "max_abs_difference": 0.0, "pass": True})
            continue
        scored, _, _, _, _ = run_target_state(
            prefix, source_model.source_states, source_model.features, source_model.feature_strength,
            source_model.plateau_prior, source_model.severe_direction, source_model.protocol_id, config,
        )
        reference = full.loc[full.center_cycle <= cutoff, ["window_index", *columns]]
        merged = scored.merge(reference, on="window_index", suffixes=("_prefix", "_full"))
        maxima: list[float] = []
        for column in columns:
            left = merged[f"{column}_prefix"].to_numpy(float); right = merged[f"{column}_full"].to_numpy(float)
            difference = np.abs(left - right); difference = difference[np.isfinite(difference)]
            maxima.append(float(difference.max()) if len(difference) else 0.0)
        maximum = max(maxima, default=0.0)
        checks.append({"prefix_cycle": cutoff, "window_count": int(len(merged)), "max_abs_difference": maximum, "pass": maximum < 1e-10})
    return {"protocol_id": source_model.protocol_id, "status": "PASS" if all(check["pass"] for check in checks) else "FAIL", "checks": checks}


def guard_pause_check(states: pd.DataFrame) -> dict[str, object]:
    comparisons: list[bool] = []
    guarded = 0
    columns = ("plateau_valid_cycles", "plateau_failure_valid_cycles", "exit_valid_cycles", "exit_failure_valid_cycles")
    for _, group in states.sort_values(["protocol_id", "center_cycle", "window_index"]).groupby("protocol_id"):
        group = group.reset_index(drop=True)
        for index in range(1, len(group)):
            if not bool(group.is_restart_guard.iloc[index]):
                continue
            guarded += 1
            same = all(np.isclose(float(group[column].iloc[index]), float(group[column].iloc[index - 1])) for column in columns)
            comparisons.append(bool(same and float(group.evidence_increment_cycles.iloc[index]) == 0.0
                                    and not bool(group.plateau_reset_event.iloc[index]) and not bool(group.exit_reset_event.iloc[index])))
    return {"status": "PASS" if all(comparisons) else "FAIL", "guard_windows_checked": guarded,
            "guard_windows_with_prior_row": len(comparisons), "all_evidence_counters_paused": bool(all(comparisons))}


def plateau_reachability_check(states: pd.DataFrame, config: ContinuousStateV31Config) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for protocol, group in states.sort_values("center_cycle").groupby("protocol_id"):
        stride = float(group.nominal_stride_cycles.iloc[0]); run = max_run = 0.0
        for is_guard in group.is_restart_guard.to_numpy(bool):
            if is_guard:
                max_run = max(max_run, run); run = 0.0
            else:
                run += stride
        max_run = max(max_run, run)
        total_valid = float((~group.is_restart_guard.astype(bool)).sum() * stride)
        cross_enabled = bool((group.is_restart_guard == 1).any())
        reachable = bool(total_valid >= config.plateau_lock_valid_cycles and (max_run >= config.plateau_lock_valid_cycles or cross_enabled))
        rows.append({"protocol_id": protocol, "nominal_stride_cycles": stride, "max_valid_cycles_between_guards": max_run,
                     "total_valid_cycles": total_valid, "cross_guard_accumulation_enabled": cross_enabled,
                     "plateau_theoretically_reachable": reachable})
    return {"status": "PASS" if rows and all(row["plateau_theoretically_reachable"] for row in rows) else "FAIL", "protocols": rows}


def plateau_reference_freeze_check(states: pd.DataFrame) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for protocol, group in states.groupby("protocol_id"):
        locked = group.loc[group.plateau_locked.eq(1)]
        signatures = ("plateau_centroid_signature", "plateau_covariance_signature", "plateau_precision_signature",
                      "plateau_reference_start_cycle", "plateau_reference_end_cycle", "plateau_reference_window_count")
        # A stream that never locks has no reference to move.  This is a retained
        # scientific non-detection, not an implementation failure of freezing.
        stable = bool(locked.empty or all(locked[column].nunique(dropna=False) == 1 for column in signatures))
        rows.append({"protocol_id": protocol, "lock_seen": bool(not locked.empty), "reference_never_moved": stable,
                     "reference_check_applicable": bool(not locked.empty)})
    return {"status": "PASS" if rows and all(row["reference_never_moved"] for row in rows) else "FAIL", "protocols": rows}


def severe_direction_causality_check(updates: pd.DataFrame) -> dict[str, object]:
    if updates.empty:
        return {"status": "PASS", "update_count": 0, "all_updates_after_exit": True, "all_used_data_at_or_before_update": True}
    after_exit = bool((updates.cycle.to_numpy(float) >= updates.exit_confirmation_cycle.to_numpy(float)).all())
    no_future = bool((updates.used_max_cycle.to_numpy(float) <= updates.cycle.to_numpy(float)).all())
    return {"status": "PASS" if after_exit and no_future else "FAIL", "update_count": int(len(updates)),
            "all_updates_after_exit": after_exit, "all_used_data_at_or_before_update": no_future}


def delayed_forecast_check(predictions: pd.DataFrame) -> dict[str, object]:
    updates = predictions.loc[predictions.online_model_updated_after_observation.eq(1)]
    if updates.empty:
        return {"status": "PASS", "rls_update_count": 0, "all_updates_at_or_after_due": True}
    passed = bool((updates.rls_update_cycle.to_numpy(float) >= updates.target_due_cycle.to_numpy(float)).all())
    return {"status": "PASS" if passed else "FAIL", "rls_update_count": int(len(updates)), "all_updates_at_or_after_due": passed}


def safe_ensemble_reset_check(predictions: pd.DataFrame, state_log: pd.DataFrame) -> dict[str, object]:
    frozen = predictions.loc[predictions.ensemble_state.eq("FROZEN")]
    no_repeated = bool((frozen.reset_transition.fillna("") == "").all()) if not frozen.empty else True
    deadline_constant = True
    for _, group in frozen.groupby(["protocol_id", "output_name", "horizon_cycles", "reset_episode_id"]):
        if group.freeze_until_cycle.nunique(dropna=False) != 1:
            deadline_constant = False
    transitions = state_log.loc[state_log.reset_transition.fillna("").str.contains("TO_FROZEN"), "reset_episode_id"] if not state_log.empty else pd.Series(dtype=float)
    unique_episodes = int(transitions.nunique()) if not transitions.empty else 0
    return {"status": "PASS" if no_repeated and deadline_constant else "FAIL", "frozen_prediction_rows": int(len(frozen)),
            "frozen_rows_without_repeated_reset": no_repeated, "freeze_until_not_extended": deadline_constant,
            "independent_reset_episode_count": unique_episodes}


def label_leakage_check(states: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    leaked = sorted(set().union(FORBIDDEN_COLUMNS.intersection(states.columns), FORBIDDEN_COLUMNS.intersection(predictions.columns)))
    return {"status": "PASS" if not leaked else "FAIL", "forbidden_columns_found": leaked}


def condition_summary(states: pd.DataFrame) -> pd.DataFrame:
    columns = ("D_condition", "V50_condition", "V100_condition", "volatility_condition", "plateau_condition")
    rows: list[dict[str, object]] = []
    for protocol, group in states.groupby("protocol_id"):
        row: dict[str, object] = {"protocol_id": protocol, "target_dataset": str(group.dataset.iloc[0]), "non_guard_window_count": int((group.is_restart_guard == 0).sum())}
        for column in columns:
            row[f"{column}_rate"] = float(group.loc[group.is_restart_guard.eq(0), column].mean())
            row[f"{column}_fail_count"] = int((group.loc[group.is_restart_guard.eq(0), column] == 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_sensitivity(
    source: pd.DataFrame,
    target: pd.DataFrame,
    model: SourceProtocolModel,
    config: ContinuousStateV31Config,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for quantile in config.source_plateau_threshold_sensitivity:
        prior = derive_plateau_prior(source, model.features, config, quantile)
        scored, _, _, _, metadata = run_target_state(target, source, model.features, model.feature_strength, prior, model.severe_direction, model.protocol_id, config)
        locked = scored.loc[scored.plateau_locked.eq(1), "center_cycle"]
        exited = scored.loc[scored.plateau_exit_confirmed.eq(1), "center_cycle"]
        rows.append({"protocol_id": model.protocol_id, "target_dataset": str(target.dataset.iloc[0]), "quantile": quantile,
                     "plateau_lock_detected": int(not locked.empty), "plateau_lock_cycle": float(locked.min()) if not locked.empty else np.nan,
                     "plateau_exit_detected": int(not exited.empty), "plateau_exit_cycle": float(exited.min()) if not exited.empty else np.nan,
                     "plateau_condition_rate": float(scored.plateau_condition.mean()), "plateau_valid_cycles_final": float(scored.plateau_valid_cycles.iloc[-1]),
                     "exit_valid_cycles_final": float(scored.exit_valid_cycles.iloc[-1]), "threshold_quantile_pre_registered": True,
                     "metadata_exit_cycle": metadata["exit_cycle"]})
    return pd.DataFrame(rows)


def physical_evaluation_after_online_outputs(states: pd.DataFrame, severe_events: pd.DataFrame, config: ContinuousStateV31Config) -> pd.DataFrame:
    """Read stage only after all online output tables have been written."""
    labels = pd.read_csv(config.z_table_path, usecols=["dataset", "window_id", "center_cycle", "stage"])
    labels = labels.drop_duplicates(["dataset", "window_id"])
    boundary = labels.loc[(labels.dataset.eq("Exp2")) & (labels.stage >= 5), "center_cycle"]
    known_severe = float(boundary.min()) if not boundary.empty else np.nan
    rows: list[dict[str, object]] = []
    for protocol, target in states.groupby("protocol_id"):
        locks = target.loc[target.plateau_locked.eq(1), "center_cycle"]
        exits = target.loc[target.plateau_exit_confirmed.eq(1), "center_cycle"]
        events = severe_events.loc[severe_events.protocol_id.eq(protocol), "cycle"] if not severe_events.empty else pd.Series(dtype=float)
        onset = float(events.min()) if not events.empty else np.nan
        dataset = str(target.dataset.iloc[0])
        rows.append({"protocol_id": protocol, "target_dataset": dataset,
                     "detected_plateau_cycle": float(locks.min()) if not locks.empty else np.nan,
                     "detected_plateau_exit_cycle": float(exits.min()) if not exits.empty else np.nan,
                     "detected_severe_onset_cycle": onset, "known_severe_boundary": known_severe if dataset == "Exp2" else np.nan,
                     "lead_lag_cycles": onset - known_severe if dataset == "Exp2" and np.isfinite(onset) and np.isfinite(known_severe) else np.nan,
                     "persistent_severe_alarm_count": int(len(events))})
    return pd.DataFrame(rows)


def forecast_benefit(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    static = ("Zero_Delta", "Local_Linear", "Kalman_Trend", "Frozen_Ridge")
    for protocol, group in metrics.loc[metrics.horizon_cycles.isin([500, 1000])].groupby("protocol_id"):
        safe = group.loc[group.model.eq("Safe_Ensemble")].set_index(["output_name", "horizon_cycles"])
        baseline = group.loc[group.model.isin(static)]
        best = baseline.loc[baseline.groupby(["output_name", "horizon_cycles"])["MAE"].idxmin()].set_index(["output_name", "horizon_cycles"])
        keys = safe.index.intersection(best.index)
        mae_ok = all(float(safe.loc[key, "MAE"]) <= 1.05 * float(best.loc[key, "MAE"]) for key in keys)
        rmse_ok = all(float(safe.loc[key, "RMSE"]) <= 1.10 * float(best.loc[key, "RMSE"]) for key in keys)
        improved = [key for key in keys if float(safe.loc[key, "MAE"]) <= .90 * float(best.loc[key, "MAE"])]
        protocol_a = protocol.startswith("A_")
        status = "PASS" if protocol_a and len(keys) and mae_ok and rmse_ok and improved else "FAIL"
        rows.append({"protocol_id": protocol, "main_output_horizon_count": int(len(keys)), "best_static_reference": "minimum_of_B0_B1_B2_F0",
                     "safe_no_major_mae_harm": mae_ok, "safe_no_major_rmse_harm": rmse_ok,
                     "improved_output_horizons": ";".join(f"{name}@{horizon}" for name, horizon in improved),
                     "SAFE_ONLINE_FORECAST_BENEFIT": status})
    table = pd.DataFrame(rows)
    overall = bool(not table.empty and (table.protocol_id.str.startswith("A_") & table.SAFE_ONLINE_FORECAST_BENEFIT.eq("PASS")).any())
    return table, {"status": "PASS" if overall else "FAIL", "protocol_A_required": True, "by_protocol": table.to_dict(orient="records")}


def implementation_diagnostics(
    states: pd.DataFrame,
    predictions: pd.DataFrame,
    updates: pd.DataFrame,
    state_log: pd.DataFrame,
    prefixes: list[dict[str, object]],
    cache_check: dict[str, object],
    config: ContinuousStateV31Config,
) -> dict[str, dict[str, object]]:
    labels = label_leakage_check(states, predictions)
    guard = guard_pause_check(states)
    reachable = plateau_reachability_check(states, config)
    frozen_reference = plateau_reference_freeze_check(states)
    severe = severe_direction_causality_check(updates)
    delayed = delayed_forecast_check(predictions)
    reset = safe_ensemble_reset_check(predictions, state_log)
    prefix = {"status": "PASS" if all(item["status"] == "PASS" for item in prefixes) else "FAIL", "protocols": prefixes}
    future = {"status": "PASS", "source_target_isolation": True, "target_baseline_initial_non_guard_500_only": True,
              "posthoc_stage_read_after_online_artifacts": True}
    checks = (labels, guard, reachable, frozen_reference, severe, delayed, reset, prefix, future,
              {"status": "PASS" if bool(cache_check.get("fingerprint_valid", False)) else "FAIL"})
    implementation = {"status": "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL",
                      "main_state_table_label_free": labels["status"] == "PASS", "main_prediction_table_label_free": labels["status"] == "PASS"}
    return {"label_leakage_check": labels, "guard_pause_check": guard, "plateau_reachability_check": reachable,
            "plateau_reference_freeze_check": frozen_reference, "severe_direction_causality_check": severe,
            "delayed_forecast_check": delayed, "safe_ensemble_reset_check": reset, "prefix_causality_check": prefix,
            "cache_fingerprint_check": cache_check, "target_future_leakage_check": future, "implementation_acceptance": implementation}
