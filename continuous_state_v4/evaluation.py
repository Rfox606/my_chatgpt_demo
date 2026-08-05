from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ContinuousStateV4Config
from .data import FORBIDDEN_COLUMNS
from .forecast import STATIC_MODELS
from .state_engine import EVIDENCE_NAMES, main_state_columns, run_target_state


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def label_leakage_check(*frames: pd.DataFrame) -> dict[str, object]:
    leaked = sorted(set().union(*(FORBIDDEN_COLUMNS.intersection(frame.columns) for frame in frames)))
    return {"status": "PASS" if not leaked else "FAIL", "forbidden_columns_found": leaked}


def guard_pause_check(states: pd.DataFrame) -> dict[str, object]:
    columns = ["evidence_increment_cycles", *(f"{name}_run_cycles" for name in EVIDENCE_NAMES),
               *(f"{name}_false_cycles" for name in EVIDENCE_NAMES), *EVIDENCE_NAMES]
    checks: list[bool] = []
    count = 0
    for _, group in states.sort_values(["protocol_id", "center_cycle", "window_index"]).groupby("protocol_id"):
        group = group.reset_index(drop=True)
        for index in range(1, len(group)):
            if not bool(group.is_restart_guard.iloc[index]):
                continue
            count += 1
            same = all(float(group[column].iloc[index]) == float(group[column].iloc[index - 1]) for column in columns if column != "evidence_increment_cycles")
            checks.append(bool(same and float(group.evidence_increment_cycles.iloc[index]) == 0.0))
    return {"status": "PASS" if all(checks) else "FAIL", "guard_windows_checked": count,
            "all_evidence_tracks_paused": bool(all(checks))}


def prefix_causality(
    target: pd.DataFrame,
    full: pd.DataFrame,
    protocol_id: str,
    features: tuple[str, ...],
    config: ContinuousStateV4Config,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    for cutoff in (2000, 5000, 10000):
        prefix = target.loc[target.center_cycle <= cutoff].copy()
        if prefix.empty:
            checks.append({"prefix_cycle": cutoff, "window_count": 0, "max_abs_difference": 0.0, "pass": True})
            continue
        scored, _, _ = run_target_state(prefix, protocol_id, features, config)
        columns = [column for column in main_state_columns() if column in scored.columns and column in full.columns]
        reference = full.loc[full.center_cycle <= cutoff, ["window_index", *columns]]
        merged = scored.loc[:, ["window_index", *columns]].merge(reference, on="window_index", suffixes=("_prefix", "_full"))
        maxima: list[float] = []
        for column in columns:
            left = merged[f"{column}_prefix"].to_numpy(float)
            right = merged[f"{column}_full"].to_numpy(float)
            delta = np.abs(left - right)
            delta = delta[np.isfinite(delta)]
            maxima.append(float(delta.max()) if len(delta) else 0.0)
        maximum = max(maxima, default=0.0)
        checks.append({"prefix_cycle": cutoff, "window_count": int(len(merged)), "max_abs_difference": maximum, "pass": maximum < 1e-10})
    return {"protocol_id": protocol_id, "status": "PASS" if all(row["pass"] for row in checks) else "FAIL", "checks": checks}


def no_future_check(states: pd.DataFrame, config: ContinuousStateV4Config) -> dict[str, object]:
    frozen_rows = states.loc[states.baseline_frozen.eq(1)]
    frozen = bool(not frozen_rows.empty and (frozen_rows.center_cycle <= config.baseline_cycles).all() and
                  (frozen_rows.is_restart_guard.eq(0)).all())
    no_stage = not bool(FORBIDDEN_COLUMNS.intersection(states.columns))
    return {"status": "PASS" if frozen and no_stage else "FAIL", "target_baseline_initial_period_only": frozen,
            "target_future_values_not_used_for_baseline": frozen, "stage_not_read_by_online_state": no_stage}


def forecast_update_check(updates: pd.DataFrame) -> dict[str, object]:
    if updates.empty:
        return {"status": "FAIL", "rls_update_count": 0, "all_updates_after_due_observation": False}
    passed = bool((updates.rls_update_cycle.to_numpy(float) >= updates.target_due_cycle.to_numpy(float)).all() and
                  (updates.due_observation_cycle.to_numpy(float) >= updates.target_due_cycle.to_numpy(float)).all())
    return {"status": "PASS" if passed else "FAIL", "rls_update_count": int(len(updates)), "all_updates_after_due_observation": passed}


def safe_gate_check(predictions: pd.DataFrame) -> dict[str, object]:
    if predictions.empty:
        return {"status": "FAIL", "online_rows": 0, "compares_current_best_static": False}
    online = predictions.loc[predictions.safe_gate_selected_model.eq("Online_RLS")]
    uses_best = set(predictions.safe_gate_compared_against.dropna().unique()).issubset(set(STATIC_MODELS))
    online_ok = bool(online.empty or (online.online_rolling_mae <= online.safe_gate_best_static_rolling_mae).all())
    non_frozen_reference = bool((predictions.safe_gate_compared_against != "Frozen_Ridge").any())
    return {"status": "PASS" if uses_best and online_ok else "FAIL", "online_rows": int(len(online)),
            "compares_current_best_static": uses_best, "online_not_worse_than_selected_static_window": online_ok,
            "observed_non_frozen_static_reference": non_frozen_reference}


def forecast_benefit(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (protocol, output_name, horizon), group in metrics.groupby(["protocol_id", "output_name", "horizon_cycles"]):
        static = group.loc[group.model.isin(STATIC_MODELS)].set_index("model")
        online = group.loc[group.model.eq("Online_RLS")]
        safe = group.loc[group.model.eq("Safe_Gate")]
        if static.empty or online.empty:
            continue
        best_name = str(static.MAE.idxmin())
        best_mae = float(static.loc[best_name, "MAE"])
        online_mae = float(online.MAE.iloc[0])
        safe_mae = float(safe.MAE.iloc[0]) if not safe.empty else np.nan
        rows.append({"protocol_id": protocol, "output_name": output_name, "horizon_cycles": int(horizon),
                     "best_static_model": best_name, "best_static_MAE": best_mae, "Online_RLS_MAE": online_mae,
                     "Safe_Gate_MAE": safe_mae, "online_to_best_static_mae_ratio": online_mae / best_mae if best_mae > 0 else np.nan,
                     "online_truly_better": bool(online_mae < best_mae), "safe_gate_no_worse_than_best_static_5pct": bool(safe_mae <= 1.05 * best_mae)})
    table = pd.DataFrame(rows)
    improved = int(table.online_truly_better.sum()) if not table.empty else 0
    return table, {"status": "PASS" if improved > 0 else "FAIL", "online_better_output_horizon_count": improved,
                   "evaluated_output_horizon_count": int(len(table)), "by_output_horizon": table.to_dict(orient="records")}


def evidence_summary(states: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protocol, group in states.groupby("protocol_id"):
        dataset = str(group.dataset.iloc[0])
        for name in EVIDENCE_NAMES:
            onsets = events.loc[(events.protocol_id.eq(protocol)) & events.evidence_type.eq(name) & events.event.eq("algorithm_evidence_onset"), "cycle"] if not events.empty else pd.Series(dtype=float)
            rows.append({"protocol_id": protocol, "target_dataset": dataset, "evidence_type": name,
                         "active_window_count": int(group[name].sum()), "onset_count": int(len(onsets)),
                         "first_onset_cycle": float(onsets.min()) if not onsets.empty else np.nan,
                         "last_onset_cycle": float(onsets.max()) if not onsets.empty else np.nan,
                         "active_at_8000": bool(group.loc[np.abs(group.center_cycle - 8000.0).idxmin(), name])})
    return pd.DataFrame(rows)


def implementation_diagnostics(
    states: pd.DataFrame,
    predictions: pd.DataFrame,
    updates: pd.DataFrame,
    prefixes: list[dict[str, object]],
    config: ContinuousStateV4Config,
) -> dict[str, dict[str, object]]:
    labels = label_leakage_check(states, predictions)
    guard = guard_pause_check(states)
    future = no_future_check(states, config)
    prefix = {"status": "PASS" if all(item["status"] == "PASS" for item in prefixes) else "FAIL", "protocols": prefixes}
    forecast = forecast_update_check(updates)
    gate = safe_gate_check(predictions)
    checks = (labels, guard, future, prefix, forecast, gate)
    implementation = {"status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
                      "state_and_forecast_outputs_label_free": labels["status"] == "PASS"}
    return {"label_leakage_check": labels, "guard_pause_check": guard, "target_future_leakage_check": future,
            "prefix_causality_check": prefix, "predict_then_update_check": forecast, "safe_gate_check": gate,
            "implementation_acceptance": implementation}
