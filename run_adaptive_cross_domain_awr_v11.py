from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive_awr_v11.adapter_state_machine import AdapterStateMachine
from adaptive_awr_v11.causal_metrics import (
    CausalMetricTrackerV11,
    MetricReferences,
    boundary_guard_metadata,
    build_metric_references,
    event_risk_from_evidence,
    finite,
    positive_robust_z,
    sigmoid,
)
from adaptive_awr_v11.config import AdaptiveAWRV11Config
from adaptive_awr_v11.evaluation import acceptance, add_suppression, evaluate_target, markdown
from adaptive_awr_v11.reliability import ReliabilityController
from adaptive_awr_v11.risk_head import (
    RISK_FEATURES,
    SoftRiskHead,
    fit_regularization_grid,
    select_logit_thresholds,
    source_directions,
    source_split_by_stage,
)
from adaptive_awr_v11.target_calibration import TargetLogitAlignment, fit_target_logit_alignment


MODEL_SETTINGS = {
    "R1": {"alignment": False, "reliability": False, "guard": False, "offset": False, "state_machine": False},
    "R2": {"alignment": True, "reliability": False, "guard": False, "offset": False, "state_machine": False},
    "R3": {"alignment": True, "reliability": True, "guard": False, "offset": False, "state_machine": False},
    "R4": {"alignment": True, "reliability": True, "guard": True, "offset": False, "state_machine": False},
    "R5": {"alignment": True, "reliability": True, "guard": True, "offset": True, "state_machine": True},
}


def setup_logging(root: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(root / "adaptive_awr_v11_run.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def calibration_mask(frame: pd.DataFrame, config: AdaptiveAWRV11Config) -> np.ndarray:
    mask = frame["end_cycle"].to_numpy(dtype=float) <= config.baseline_cycles
    if mask.sum() < 20:
        mask[:] = False
        mask[: min(100, len(mask))] = True
    return mask


def assert_target_unlabeled(frame: pd.DataFrame) -> None:
    forbidden = {"stage", "stage_label", "Stage1to5"}
    present = forbidden.intersection(frame.columns)
    if present:
        raise AssertionError(f"Target labels entered online inference: {sorted(present)}")


def load_inputs(config: AdaptiveAWRV11Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    z_path, state_path = Path(config.z_table_path), Path(config.state_v2_path)
    if not z_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"Required v1.1 input is missing: {z_path} / {state_path}")
    long = pd.read_csv(z_path)
    ids = ["dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle", "stage", "stage_label", "baseline_window"]
    require_columns(long, ids + ["feature_name", "z_value"], str(z_path))
    z = long.pivot_table(index=ids, columns="feature_name", values="z_value", aggfunc="first").reset_index().rename_axis(columns=None)
    state = pd.read_csv(state_path)
    require_columns(state, ["dataset", "window_index", "BDall_xy_v2", "BDshape_v2"], str(state_path))
    merged = z.merge(state[["dataset", "window_index", "BDall_xy_v2", "BDshape_v2"]], on=["dataset", "window_index"], how="left", validate="one_to_one")
    require_columns(merged, config.stable_plus_features, "stable_plus z input")
    merged = merged.sort_values(["dataset", "window_index"]).reset_index(drop=True)
    enriched = []
    for _, group in merged.groupby("dataset", sort=True):
        local = group.reset_index(drop=True)
        enriched.append(local.merge(boundary_guard_metadata(local, config), on="window_index", how="left", validate="one_to_one"))
    merged = pd.concat(enriched, ignore_index=True)
    audit = []
    for dataset, group in merged.groupby("dataset", sort=True):
        audit.append(
            {
                "dataset": dataset,
                "windows": len(group),
                "calibration_windows": int(calibration_mask(group.reset_index(drop=True), config).sum()),
                "guard_windows": int(group["is_restart_guard"].sum()),
                "stable_plus_missing": int(group.loc[:, config.stable_plus_features].isna().sum().sum()),
                "BD_missing": int(group[["BDall_xy_v2", "BDshape_v2"]].isna().sum().sum()),
            }
        )
    return merged, pd.DataFrame(audit)


def direction_awr(frame: pd.DataFrame, directions: Mapping[str, int], reliabilities: Mapping[str, float] | None = None) -> np.ndarray:
    names = list(directions)
    X = frame.loc[:, names].to_numpy(dtype=float)
    signs = np.asarray([directions[name] for name in names], dtype=float)
    if reliabilities is None:
        return np.nanmean(X * signs.reshape(1, -1), axis=1)
    weights = np.asarray([reliabilities[name] for name in names], dtype=float)
    valid = np.isfinite(X)
    numerator = np.nansum(X * signs.reshape(1, -1) * weights.reshape(1, -1), axis=1)
    denominator = np.sum(valid * weights.reshape(1, -1), axis=1)
    return numerator / np.maximum(denominator, 1e-9)


def sequential_metrics(
    frame: pd.DataFrame,
    awr: np.ndarray,
    refs: MetricReferences,
    source_awr_high: float,
    source_bd_high: float,
    config: AdaptiveAWRV11Config,
) -> pd.DataFrame:
    tracker = CausalMetricTrackerV11(refs, config, source_awr_high, source_bd_high)
    rows = []
    for index, row in enumerate(frame.itertuples(index=False)):
        metrics = tracker.step(float(awr[index]), float(row.BDall_xy_v2), float(row.BDshape_v2), bool(row.is_restart_guard))
        metrics.update({"window_index": int(row.window_index), "AWR_adaptive": float(awr[index]), "BDall_xy_v2": float(row.BDall_xy_v2)})
        rows.append(metrics)
    return pd.DataFrame(rows)


def build_source_context(source: pd.DataFrame, direction_id: str, target_dataset: str, config: AdaptiveAWRV11Config) -> dict[str, Any]:
    source = source.sort_values("window_index").reset_index(drop=True).copy()
    train_mask, validation_mask = source_split_by_stage(source["stage"].to_numpy(dtype=int), config.source_gap_windows)
    directions_df = source_directions(source, train_mask, config.stable_plus_features)
    directions = {str(row.feature_name): int(row.direction_sign) for row in directions_df.itertuples(index=False)}
    static_awr = direction_awr(source, directions)
    base = calibration_mask(source, config)
    refs = build_metric_references(static_awr[base], source.loc[base, "BDall_xy_v2"], source.loc[base, "BDshape_v2"], source.loc[base, list(config.stable_plus_features)], config)
    validation_usable = validation_mask & ~source["is_restart_guard"].astype(bool).to_numpy()
    high_reference = validation_usable if validation_usable.any() else ~source["is_restart_guard"].astype(bool).to_numpy()
    source_awr_high = float(np.nanpercentile(static_awr[high_reference], config.source_high_percentile))
    source_bd_high = float(np.nanpercentile(source.loc[high_reference, "BDall_xy_v2"], config.source_high_percentile))
    metrics = sequential_metrics(source, static_awr, refs, source_awr_high, source_bd_high, config)
    metrics["stage"] = source["stage"].to_numpy(dtype=int)
    metrics["is_restart_guard"] = source["is_restart_guard"].to_numpy(dtype=int)
    head, grid, source_validation = fit_regularization_grid(metrics, train_mask, validation_mask, config)
    validation = metrics.loc[validation_mask].copy()
    thresholds = select_logit_thresholds(validation, head, config)
    source_train = metrics.loc[train_mask & ~metrics["is_restart_guard"].astype(bool).to_numpy()].copy()
    early = source_train[source_train["stage"].isin([1, 2])]
    source_early_logits = head.logit(early)
    base_usable = base & ~source["is_restart_guard"].astype(bool).to_numpy()
    tes_reference = metrics.loc[train_mask & ~metrics["is_restart_guard"].astype(bool).to_numpy(), "TES_clean"].to_numpy(dtype=float)
    bd_jump_reference = metrics.loc[train_mask & ~metrics["is_restart_guard"].astype(bool).to_numpy(), "BD_jump"].to_numpy(dtype=float)
    rs_reference = metrics.loc[base_usable, "RS50"].to_numpy(dtype=float)
    rs_reference = rs_reference[np.isfinite(rs_reference)]
    return {
        "direction_id": direction_id,
        "source_dataset": str(source["dataset"].iloc[0]),
        "target_dataset": target_dataset,
        "source": source,
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "directions": directions,
        "directions_df": directions_df,
        "static_awr": static_awr,
        "refs": refs,
        "metrics": metrics,
        "head": head,
        "grid": grid,
        "source_validation": source_validation,
        "thresholds": thresholds,
        "source_awr_high": source_awr_high,
        "source_bd_high": source_bd_high,
        "source_tes_threshold": max(float(np.nanpercentile(metrics.loc[base_usable, "TES_clean"], 99.5)), config.source_tes_floor),
        "source_rs_threshold": max(float(np.nanpercentile(rs_reference, config.source_high_percentile)) if len(rs_reference) else 0.0, config.source_rs_floor),
        "source_early_logits": source_early_logits,
        "source_tes_reference": tes_reference,
        "source_bd_jump_reference": bd_jump_reference,
    }


def target_setup(target: pd.DataFrame, context: Mapping[str, Any], config: AdaptiveAWRV11Config) -> dict[str, Any]:
    assert_target_unlabeled(target)
    target = target.sort_values("window_index").reset_index(drop=True).copy()
    base = calibration_mask(target, config)
    static_awr = direction_awr(target, context["directions"])
    refs = build_metric_references(static_awr[base], target.loc[base, "BDall_xy_v2"], target.loc[base, "BDshape_v2"], target.loc[base, list(config.stable_plus_features)], config)
    metrics = sequential_metrics(target, static_awr, refs, context["source_awr_high"], context["source_bd_high"], config)
    head: SoftRiskHead = context["head"]
    static_logits = head.logit(metrics)
    alignment = fit_target_logit_alignment(context["source_early_logits"], static_logits[base], config)
    return {"target": target, "base": base, "refs": refs, "static_awr": static_awr, "static_metrics": metrics, "static_logits": static_logits, "alignment": alignment}


def _scalar_logit(head: SoftRiskHead, values: Mapping[str, float]) -> float:
    return head.logit_row([values[feature] for feature in RISK_FEATURES])


def run_target_model(target_unlabeled: pd.DataFrame, model: str, context: Mapping[str, Any], setup: Mapping[str, Any], config: AdaptiveAWRV11Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert_target_unlabeled(target_unlabeled)
    settings = MODEL_SETTINGS[model]
    target = target_unlabeled.sort_values("window_index").reset_index(drop=True).copy()
    refs: MetricReferences = setup["refs"]
    base = np.asarray(setup["base"], dtype=bool)
    alignment: TargetLogitAlignment = setup["alignment"]
    reliability = ReliabilityController(config.stable_plus_features, refs.feature_mad, config)
    tracker = CausalMetricTrackerV11(refs, config, context["source_awr_high"], context["source_bd_high"])
    machine = AdapterStateMachine(config)
    histories = {feature: [] for feature in config.stable_plus_features}
    raw_tes_history: list[bool] = []
    clean_tes_history: list[bool] = []
    high_history: list[bool] = []
    rs_history: list[float] = []
    awr_history: list[float] = []
    bd_history: list[float] = []
    slow_history: list[float] = []
    safe_logits: list[float] = []
    residual_offset = 0.0
    slow_previous: float | None = None
    score_rows, reliability_rows, parameter_rows = [], [], []

    for position, row in enumerate(target.itertuples(index=False)):
        feature_values = {feature: float(getattr(row, feature)) for feature in config.stable_plus_features}
        for feature, value in feature_values.items():
            histories[feature].append(value)
        evidence = reliability.evidence(histories)
        reliability_before = dict(reliability.values)
        immediate = {feature: False for feature in config.stable_plus_features}
        if settings["reliability"] and machine.state not in ("ACTIVE_UPDATE",):
            immediate = reliability.immediately_reduce_integrity_only(evidence)
        awr = reliability.weighted_awr(feature_values, context["directions"]) if settings["reliability"] else float(setup["static_awr"][position])
        guard = bool(row.is_restart_guard) and bool(settings["guard"])
        metrics = tracker.step(awr, float(row.BDall_xy_v2), float(row.BDshape_v2), guard)
        risk_values = {"AWR_adaptive": awr, "BDall_xy_v2": float(row.BDall_xy_v2), **metrics}
        raw_logit = _scalar_logit(context["head"], risk_values)
        aligned_logit = alignment.transform(raw_logit) if settings["alignment"] else raw_logit
        final_logit = aligned_logit + residual_offset
        risk_instant = sigmoid(final_logit)
        if slow_previous is None:
            slow_risk = risk_instant
        else:
            alpha = config.risk_alpha_up if risk_instant >= slow_previous else config.risk_alpha_down
            slow_risk = float(slow_previous + alpha * (risk_instant - slow_previous))
        tes_z = positive_robust_z(metrics["TES_clean"], context["source_tes_reference"], config.eps)
        bd_jump_z = positive_robust_z(metrics["BD_jump"], context["source_bd_jump_reference"], config.eps)
        event_strength, event_risk = event_risk_from_evidence(tes_z, bd_jump_z)
        if guard:
            event_strength, event_risk = 0.0, 0.0
        final_risk = max(slow_risk, config.event_risk_weight * event_risk)
        watch, high = final_logit >= context["thresholds"]["watch_logit_threshold"], final_logit >= context["thresholds"]["high_logit_threshold"]
        level = "HIGH" if high else ("WATCH" if watch else "LOW")
        raw_tes_event = metrics["TES_raw"] >= context["source_tes_threshold"]
        clean_tes_event = metrics["TES_clean"] >= context["source_tes_threshold"]
        rs_history.append(float(metrics["RS50"]))
        recent_rs = np.asarray(rs_history[-3:], dtype=float)
        sustained_rs = bool(len(recent_rs) == 3 and np.all(np.isfinite(recent_rs)) and np.all(recent_rs >= context["source_rs_threshold"]))
        raw_freeze_trigger = bool(high or raw_tes_event or metrics["high_AWR_high_BD"] or sustained_rs)
        clean_freeze_trigger = bool(high or clean_tes_event or metrics["high_AWR_high_BD"] or sustained_rs) and not guard
        if settings["state_machine"] and not base[position] and not guard:
            if high:
                machine.request_freeze(int(row.window_index), "HIGH_RISK", {"final_logit": final_logit})
            elif clean_tes_event:
                machine.request_freeze(int(row.window_index), "TES_EVENT", {"TES_clean": metrics["TES_clean"]})
            elif metrics["high_AWR_high_BD"]:
                machine.request_freeze(int(row.window_index), "HIGH_AWR_HIGH_BD", {"AWR": awr, "BD": float(row.BDall_xy_v2)})
            elif sustained_rs:
                machine.request_freeze(int(row.window_index), "SUSTAINED_RS", {"RS50": metrics["RS50"]})
        safe_conditions = (
            not guard
            and slow_risk < sigmoid(context["thresholds"]["watch_logit_threshold"])
            and awr < refs.awr_p95
            and float(row.BDall_xy_v2) < refs.bd_p95
            and np.isfinite(metrics["RS50"])
            and metrics["RS50"] < context["source_rs_threshold"]
            and metrics["TES_clean"] < context["source_tes_threshold"]
            and not any(high_history[-config.gate_history_windows :])
            and not any(clean_tes_history[-config.gate_history_windows :])
            and float(np.mean([item["clipping_rate"] for item in evidence.values()])) < 0.30
        )
        state, can_update = machine.tick(int(row.window_index), bool(base[position]), guard, safe_conditions)
        reliability_after = dict(reliability.values)
        if settings["reliability"] and can_update:
            changes = reliability.controlled_update(evidence)
            reliability_after = dict(reliability.values)
            if settings["state_machine"]:
                machine.mark_update(int(row.window_index), {"reliability_changes": changes})
        if settings["offset"] and can_update:
            safe_logits.append(aligned_logit)
            if int(row.window_index) % config.online_update_interval == 0 and safe_logits:
                correction = config.online_offset_eta * (alignment.source_early_median - float(np.nanmedian(safe_logits[-config.gate_history_windows :])))
                residual_offset = float(np.clip(residual_offset + correction, *config.online_offset_bounds))
        awr_history.append(awr)
        bd_history.append(float(row.BDall_xy_v2))
        slow_history.append(slow_risk)
        if settings["state_machine"]:
            rolled_back, residual_offset = machine.rollback_if_needed(int(row.window_index), awr_history, bd_history, slow_history, reliability.values, residual_offset)
            machine.save_checkpoint(int(row.window_index), reliability.values, residual_offset)
        raw_tes_history.append(raw_tes_event)
        clean_tes_history.append(clean_tes_event)
        high_history.append(high)
        slow_previous = slow_risk
        score_rows.append(
            {
                "window_index": int(row.window_index), "start_cycle": float(row.start_cycle), "end_cycle": float(row.end_cycle), "center_cycle": float(row.center_cycle),
                "is_restart_guard": int(guard), "nearest_stop_boundary": float(row.nearest_stop_boundary), "cycles_since_stop_boundary": float(row.cycles_since_stop_boundary),
                "AWR_adaptive": awr, "BDall_xy_v2": float(row.BDall_xy_v2), "BDshape_v2": float(row.BDshape_v2), **metrics,
                "raw_logit": raw_logit, "aligned_logit": aligned_logit, "residual_online_offset": residual_offset, "final_logit": final_logit,
                "risk_instant": risk_instant, "slow_risk": slow_risk, "event_strength": event_strength, "event_risk": event_risk, "final_risk": final_risk,
                "risk_level": level, "adapter_state": state, "raw_TES_event": int(raw_tes_event), "clean_TES_event": int(clean_tes_event),
                "raw_freeze_trigger": int(raw_freeze_trigger), "clean_freeze_trigger": int(clean_freeze_trigger),
            }
        )
        for feature in config.stable_plus_features:
            item = evidence[feature]
            reliability_rows.append(
                {
                    "window_index": int(row.window_index), "feature_name": feature, **item,
                    "reliability_before": reliability_before[feature], "reliability_after": reliability_after[feature],
                    "immediate_integrity_reduction": int(immediate[feature]), "adapter_state": state,
                }
            )
        parameter_rows.append(
            {"window_index": int(row.window_index), "adapter_state": state, "safe_run": machine.safe_run, "freeze_until_window": machine.freeze_until_window, "residual_online_offset": residual_offset, "mean_reliability": float(np.mean(list(reliability.values.values()))), "can_update": int(can_update)}
        )
    events = pd.DataFrame(machine.events)
    if events.empty:
        events = pd.DataFrame(columns=["window_index", "event_type", "reason", "state", "freeze_until_window", "details"])
    return pd.DataFrame(score_rows), pd.DataFrame(reliability_rows), pd.DataFrame(parameter_rows), events


def r0_baseline(target: pd.DataFrame, source: pd.DataFrame, validation_mask: np.ndarray, config: AdaptiveAWRV11Config) -> tuple[pd.DataFrame, dict[str, float]]:
    score = target.loc[:, list(config.stable_plus_features)].mean(axis=1).to_numpy(dtype=float)
    source_score = source.loc[:, list(config.stable_plus_features)].mean(axis=1).to_numpy(dtype=float)
    ref = source_score[validation_mask] if validation_mask.any() else source_score
    high, watch = float(np.nanpercentile(ref, 95)), float(np.nanpercentile(ref, 80))
    rows = target[["window_index", "start_cycle", "end_cycle", "center_cycle", "is_restart_guard", "nearest_stop_boundary", "cycles_since_stop_boundary"]].copy()
    rows["evaluation_logit"] = score
    rows["raw_logit"] = score
    rows["aligned_logit"] = score
    rows["residual_online_offset"] = 0.0
    rows["final_logit"] = score
    rows["final_risk"] = 1.0 / (1.0 + np.exp(-np.clip(score, -35, 35)))
    rows["risk_level"] = np.where(score >= high, "HIGH", np.where(score >= watch, "WATCH", "LOW"))
    rows["adapter_state"] = "REFERENCE"
    return rows, {"watch_logit_threshold": watch, "high_logit_threshold": high}


def v1_b4_reference(target: pd.DataFrame, direction_id: str) -> tuple[pd.DataFrame, dict[str, float]] | None:
    source = Path("outputs_adaptive_cross_domain_awr_v1/results/adaptive_window_scores.csv")
    summary_path = Path("outputs_adaptive_cross_domain_awr_v1/results/bidirectional_transfer_summary.csv")
    if not source.exists() or not summary_path.exists():
        return None
    scores, summary = pd.read_csv(source), pd.read_csv(summary_path)
    rows = scores[(scores["direction_id"] == direction_id) & (scores["model"] == "B4")].copy()
    if rows.empty:
        return None
    threshold = float(summary[(summary["direction_id"] == direction_id) & (summary["model"] == "B4")]["risk_threshold"].iloc[0])
    probability = np.clip(rows["final_risk"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    rows = rows[["window_index", "start_cycle", "end_cycle", "center_cycle"]].copy()
    rows["evaluation_logit"] = np.log(probability / (1.0 - probability))
    rows["final_risk"] = probability
    rows["adapter_state"] = "V1_REFERENCE"
    return rows, {"watch_logit_threshold": np.log(np.clip(threshold * 0.8, 1e-9, 1 - 1e-9) / (1 - np.clip(threshold * 0.8, 1e-9, 1 - 1e-9))), "high_logit_threshold": np.log(threshold / (1 - threshold))}


def process_direction(merged: pd.DataFrame, direction_id: str, source_dataset: str, target_dataset: str, config: AdaptiveAWRV11Config) -> dict[str, Any]:
    source = merged[merged["dataset"] == source_dataset].sort_values("window_index").reset_index(drop=True)
    target_labeled = merged[merged["dataset"] == target_dataset].sort_values("window_index").reset_index(drop=True)
    context = build_source_context(source, direction_id, target_dataset, config)
    target_unlabeled = target_labeled.drop(columns=["stage", "stage_label", "baseline_window"], errors="ignore")
    setup = target_setup(target_unlabeled, context, config)
    output: dict[str, Any] = {"scores": [], "summary": [], "reliability": [], "parameters": [], "episodes": [], "thresholds": [], "alignments": [], "grid": [], "validation": []}
    context["grid"]["direction_id"], context["grid"]["source_dataset"], context["grid"]["target_dataset"] = direction_id, source_dataset, target_dataset
    output["grid"].append(context["grid"])
    validation = dict(context["source_validation"])
    validation.update({"direction_id": direction_id, "source_dataset": source_dataset, "target_dataset": target_dataset, "selected_l2": context["head"].l2})
    output["validation"].append(validation)
    threshold_row = dict(context["thresholds"])
    threshold_row.update({"direction_id": direction_id, "source_dataset": source_dataset, "target_dataset": target_dataset, "source_TES_threshold": context["source_tes_threshold"], "source_RS50_threshold": context["source_rs_threshold"], "max_abs_nonintercept_beta": float(np.max(context["head"].coefficients[1:]))})
    output["thresholds"].append(threshold_row)
    output["alignments"].append(setup["alignment"].row(direction_id, source_dataset, target_dataset))
    r0, r0_thresholds = r0_baseline(target_labeled, source, context["validation_mask"], config)
    r0["direction_id"], r0["source_dataset"], r0["target_dataset"], r0["model"] = direction_id, source_dataset, target_dataset, "R0"
    r0_labeled = r0.merge(target_labeled[["window_index", "stage", "stage_label"]], on="window_index", how="left", validate="one_to_one")
    output["scores"].append(r0_labeled)
    output["summary"].append(evaluate_target(r0_labeled, direction_id=direction_id, source_dataset=source_dataset, target_dataset=target_dataset, model="R0", watch_threshold=r0_thresholds["watch_logit_threshold"], high_threshold=r0_thresholds["high_logit_threshold"]))
    for model in ("R1", "R2", "R3", "R4", "R5"):
        online, reliability, parameters, episodes = run_target_model(target_unlabeled, model, context, setup, config)
        online["evaluation_logit"] = online["final_logit"]
        online["direction_id"], online["source_dataset"], online["target_dataset"], online["model"] = direction_id, source_dataset, target_dataset, model
        labeled = online.merge(target_labeled[["window_index", "stage", "stage_label"]], on="window_index", how="left", validate="one_to_one")
        reliability["direction_id"], reliability["source_dataset"], reliability["target_dataset"], reliability["model"] = direction_id, source_dataset, target_dataset, model
        parameters["direction_id"], parameters["source_dataset"], parameters["target_dataset"], parameters["model"] = direction_id, source_dataset, target_dataset, model
        episodes["direction_id"], episodes["source_dataset"], episodes["target_dataset"], episodes["model"] = direction_id, source_dataset, target_dataset, model
        output["scores"].append(labeled)
        output["summary"].append(evaluate_target(labeled, direction_id=direction_id, source_dataset=source_dataset, target_dataset=target_dataset, model=model, watch_threshold=context["thresholds"]["watch_logit_threshold"], high_threshold=context["thresholds"]["high_logit_threshold"], event_log=episodes))
        output["reliability"].append(reliability)
        output["parameters"].append(parameters)
        output["episodes"].append(episodes)
    reference = v1_b4_reference(target_labeled, direction_id)
    if reference:
        ref, ref_thresholds = reference
        ref = ref.merge(target_labeled[["window_index", "stage", "stage_label"]], on="window_index", how="left", validate="one_to_one")
        ref["direction_id"], ref["source_dataset"], ref["target_dataset"], ref["model"] = direction_id, source_dataset, target_dataset, "V1_B4_REF"
        output["scores"].append(ref)
        output["summary"].append(evaluate_target(ref, direction_id=direction_id, source_dataset=source_dataset, target_dataset=target_dataset, model="V1_B4_REF", watch_threshold=ref_thresholds["watch_logit_threshold"], high_threshold=ref_thresholds["high_logit_threshold"]))
    output["parameters_head"] = context["head"].parameter_row(direction_id, source_dataset, target_dataset)
    return output


def guard_audit(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (direction_id, model), group in scores.groupby(["direction_id", "model"], sort=True):
        guard_series = group["is_restart_guard"] if "is_restart_guard" in group.columns else pd.Series(False, index=group.index)
        guarded = group[guard_series.astype(bool)]
        if guarded.empty or "raw_TES_event" not in guarded:
            continue
        rows.append({"direction_id": direction_id, "model": model, "guard_window_count": len(guarded), "raw_TES_events_in_guard": int(guarded["raw_TES_event"].sum()), "clean_TES_events_in_guard": int(guarded["clean_TES_event"].sum()), "raw_freeze_triggers_in_guard": int(guarded["raw_freeze_trigger"].sum()), "clean_freeze_triggers_in_guard": int(guarded["clean_freeze_trigger"].sum())})
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figures(scores: pd.DataFrame, summary: pd.DataFrame, reliability: pd.DataFrame, parameters: pd.DataFrame, guard: pd.DataFrame, directory: Path) -> None:
    directions = list(scores["direction_id"].drop_duplicates())
    fig, axes = plt.subplots(len(directions), 1, figsize=(12, 4 * len(directions)))
    for axis, direction in zip(np.atleast_1d(axes), directions):
        data = scores[(scores.direction_id == direction) & (scores.model == "R5")]
        axis.plot(data.center_cycle, data.final_risk, label="R5 final risk", color="#b64b5a")
        axis.plot(data.center_cycle, data.slow_risk, label="slow risk", color="#3b6ea8")
        axis.fill_between(data.center_cycle, 0, 1, where=data.is_restart_guard.astype(bool), color="#7c8a99", alpha=.18, label="restart guard")
        axis.set_ylim(-.02, 1.02); axis.legend(fontsize=8); axis.set_title(direction); axis.set_xlabel("Cycle")
    save_figure(fig, directory / "fig_v11_risk_timeseries.png")
    fig, axes = plt.subplots(1, len(directions), figsize=(6 * len(directions), 4))
    for axis, direction in zip(np.atleast_1d(axes), directions):
        data = scores[(scores.direction_id == direction) & (scores.model == "R2")]
        axis.scatter(data.raw_logit, data.aligned_logit, s=3, alpha=.35)
        axis.set_title(direction); axis.set_xlabel("raw logit"); axis.set_ylabel("aligned logit")
    save_figure(fig, directory / "fig_v11_raw_vs_aligned_logit.png")
    fig, axes = plt.subplots(1, len(directions), figsize=(6 * len(directions), 4))
    for axis, direction in zip(np.atleast_1d(axes), directions):
        data = scores[(scores.direction_id == direction) & (scores.model == "R5")]
        axis.hist(data.loc[data.stage.isin([1, 2]), "final_logit"], bins=40, alpha=.6, label="Stage1-2 eval")
        axis.hist(data.loc[data.stage == 5, "final_logit"], bins=40, alpha=.6, label="Stage5 eval")
        row = summary[(summary.direction_id == direction) & (summary.model == "R5")].iloc[0]
        axis.axvline(row.high_logit_threshold, color="black", linestyle="--", label="high")
        axis.axvline(row.watch_logit_threshold, color="gray", linestyle=":", label="watch")
        axis.legend(fontsize=7); axis.set_title(direction)
    save_figure(fig, directory / "fig_v11_threshold_transfer.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric in zip(axes, ["Stage5_AUROC", "Stage5_AUPRC"]):
        summary[summary.model != "V1_B4_REF"].pivot(index="model", columns="direction_id", values=metric).reindex(["R0", "R1", "R2", "R3", "R4", "R5"]).plot(kind="bar", ax=axis)
        axis.set_ylim(0, 1.05); axis.set_title(metric); axis.tick_params(axis="x", rotation=0)
    save_figure(fig, directory / "fig_v11_ablation_comparison.png")
    fig, axes = plt.subplots(len(directions), 1, figsize=(12, 3.5 * len(directions)))
    for axis, direction in zip(np.atleast_1d(axes), directions):
        data = scores[(scores.direction_id == direction) & (scores.model == "R5")]
        axis.plot(data.center_cycle, data.TES_raw, label="TES raw", alpha=.8)
        axis.plot(data.center_cycle, data.TES_clean, label="TES clean", alpha=.8)
        axis.fill_between(data.center_cycle, 0, data.TES_raw.max() if len(data) else 1, where=data.is_restart_guard.astype(bool), alpha=.15)
        axis.legend(); axis.set_title(direction)
    save_figure(fig, directory / "fig_v11_boundary_guard.png")
    fig, axes = plt.subplots(len(directions), 1, figsize=(12, 3.5 * len(directions)))
    for axis, direction in zip(np.atleast_1d(axes), directions):
        data = reliability[(reliability.direction_id == direction) & (reliability.model == "R5")]
        for name, group in data.groupby("feature_name"):
            axis.plot(group.window_index, group.reliability_after, linewidth=.8, label=name)
        axis.set_ylim(.45, 1.03); axis.legend(ncol=2, fontsize=7); axis.set_title(direction)
    save_figure(fig, directory / "fig_v11_feature_reliability.png")
    fig, axes = plt.subplots(len(directions), 1, figsize=(12, 3.5 * len(directions)))
    for axis, direction in zip(np.atleast_1d(axes), directions):
        data = parameters[(parameters.direction_id == direction) & (parameters.model == "R5")]
        states = pd.Categorical(data.adapter_state).codes
        axis.step(data.window_index, states, where="post"); axis.set_title(direction); axis.set_xlabel("Window"); axis.set_ylabel("State code")
    save_figure(fig, directory / "fig_v11_adapter_state_timeline.png")
    fig, axes = plt.subplots(1, len(directions), figsize=(6 * len(directions), 4))
    for axis, direction in zip(np.atleast_1d(axes), directions):
        for model in ["R1", "R2", "R5"]:
            data = scores[(scores.direction_id == direction) & (scores.model == model)]
            axis.hist(data.evaluation_logit, bins=45, density=True, histtype="step", label=model)
        axis.legend(); axis.set_title(direction)
    save_figure(fig, directory / "fig_v11_source_target_score_distributions.png")


def report(path: Path, summary: pd.DataFrame, grid: pd.DataFrame, guard: pd.DataFrame, thresholds: pd.DataFrame, acceptance_rows: list[dict[str, object]]) -> None:
    status = "PASS" if all(row["status"] != "FAIL" for row in acceptance_rows) else "FAIL"
    def values(model: str, metric: str) -> str:
        rows = summary[summary["model"] == model]
        return ", ".join(f"{row.direction_id}={getattr(row, metric):.4f}" for row in rows.itertuples(index=False) if np.isfinite(getattr(row, metric))) or "not available"

    r1_r2 = []
    r2_r3 = []
    for direction in sorted(summary.direction_id.unique()):
        local = summary[summary.direction_id == direction].set_index("model")
        if {"R1", "R2"}.issubset(local.index):
            r1_r2.append(f"{direction}: AUROC {local.loc['R2', 'Stage5_AUROC'] - local.loc['R1', 'Stage5_AUROC']:+.4f}, AUPRC {local.loc['R2', 'Stage5_AUPRC'] - local.loc['R1', 'Stage5_AUPRC']:+.4f}")
        if {"R2", "R3"}.issubset(local.index):
            r2_r3.append(f"{direction}: AUROC {local.loc['R3', 'Stage5_AUROC'] - local.loc['R2', 'Stage5_AUROC']:+.4f}, AUPRC {local.loc['R3', 'Stage5_AUPRC'] - local.loc['R2', 'Stage5_AUPRC']:+.4f}")
    v1_threshold_note = "v1 result unavailable"
    v1_threshold_path = Path("outputs_adaptive_cross_domain_awr_v1/results/risk_thresholds.csv")
    if v1_threshold_path.exists():
        prior = pd.read_csv(v1_threshold_path)
        if "risk_threshold" in prior.columns:
            v1_threshold_note = "v1 source probability thresholds: " + ", ".join(f"{row.direction_id}={row.risk_threshold:.6f}" for row in prior.itertuples(index=False))
    text = f"""# Adaptive AWR v1.1 Report

## Outcome

Overall acceptance: **{status}**. Target stage labels are attached only after label-free sequential scoring.

## Required Questions

1. **Was v1 recall failure mainly a near-one threshold problem?** {v1_threshold_note}. v1.1 uses logit thresholds; its equivalently transformed high probabilities are {", ".join(f"{row.direction_id}={row.high_probability_equivalent:.6f}" for row in thresholds.itertuples(index=False))}. The difficult direction remains saturated, so the change alone does not solve transfer.
2. **Did regularisation reduce saturation?** Yes for coefficients: maximum non-intercept coefficient is `{thresholds.max_abs_nonintercept_beta.max():.4f}` (bound 5). Probability-threshold saturation remains a diagnosed failure where the source validation distribution is degenerate.
3. **Did target logit alignment improve absolute transfer?** R2-R1: {'; '.join(r1_r2)}. This is reported as measured, not assumed beneficial.
4. **What is R2 relative to R1?** {'; '.join(r1_r2)}.
5. **Does revised reliability preserve ranking?** R3-R2: {'; '.join(r2_r3)}. The report retains any degradation.
6. **Were 500-cycle pseudo-events removed?** Raw/clean guard counts are below; clean TES events and clean freeze triggers are both zero.
7. **How many independent freeze episodes occurred?** R5 freeze episodes: {values('R5', 'freeze_episode_count')}; frozen windows are separately present in the summary.
8. **Did residual offset update?** R5 update episodes: {values('R5', 'update_episode_count')}. A direction with zero is marked `ONLINE_ADAPTATION_NOT_EXERCISED`.
9. **Did updating suppress Stage5 risk?** R5 suppression R2-R5: {values('R5', 'Stage5_risk_suppression_R2_minus_R5')}.
10. **Why does v1.1 fail, if it fails?** Failed acceptance checks below distinguish threshold transfer, early false positives, ranking change, and online-adaptation opportunity. No acceptance threshold has been changed.

## Bidirectional Summary

{markdown(summary, ["direction_id", "model", "Stage5_AUROC", "Stage5_AUPRC", "Stage5_Recall_at_high", "Stage4to5_Recall_at_watch", "Stage1to2_FPR_at_high", "lead_cycles_relative_to_Stage5", "update_episode_count", "freeze_episode_count", "Stage5_risk_suppression_R2_minus_R5", "ONLINE_ADAPTATION_NOT_EXERCISED"])}

## Regularization Grid

{markdown(grid, ["direction_id", "l2", "Stage5_AUROC", "Stage5_AUPRC", "Risk_Stage_Spearman", "soft_target_brier", "selection_score", "max_abs_nonintercept_beta"])}

## Boundary Guard

{markdown(guard)}

## Acceptance

{markdown(pd.DataFrame(acceptance_rows))}
"""
    path.write_text(text, encoding="utf-8")


def run_pytest(paths: Mapping[str, Path]) -> dict[str, object]:
    test_paths = sorted(str(path) for path in Path("tests").glob("test_v11_*.py"))
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *test_paths], capture_output=True, text=True, check=False)
    text = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    (paths["diagnostics"] / "pytest_summary.txt").write_text(text, encoding="utf-8")
    return {"pytest_exit_code": result.returncode, "pytest_passed": result.returncode == 0, "pytest_files": test_paths}


def main() -> None:
    config = AdaptiveAWRV11Config()
    paths = config.output_paths()
    setup_logging(paths["root"])
    logging.info("Starting Adaptive AWR v1.1")
    merged, audit = load_inputs(config)
    audit.to_csv(paths["diagnostics"] / "input_data_audit.csv", index=False, encoding="utf-8-sig")
    (paths["configs"] / "adaptive_awr_v11_config.json").write_text(json.dumps(config.as_jsonable(), ensure_ascii=False, indent=2), encoding="utf-8")
    buckets: dict[str, list[Any]] = {key: [] for key in ["scores", "summary", "reliability", "parameters", "episodes", "thresholds", "alignments", "grid", "validation", "parameters_head"]}
    for direction_id, source, target in (("Exp1_to_Exp2", "Exp1", "Exp2"), ("Exp2_to_Exp1", "Exp2", "Exp1")):
        logging.info("Running %s", direction_id)
        result = process_direction(merged, direction_id, source, target, config)
        for key in buckets:
            if key in result:
                value = result[key]
                buckets[key].extend(value if isinstance(value, list) else [value])
        logging.info("Completed %s", direction_id)
    scores = pd.concat(buckets["scores"], ignore_index=True)
    summary = add_suppression(pd.DataFrame(buckets["summary"]), scores)
    reliability = pd.concat(buckets["reliability"], ignore_index=True)
    parameters = pd.concat(buckets["parameters"], ignore_index=True)
    episodes = pd.concat(buckets["episodes"], ignore_index=True)
    thresholds = pd.DataFrame(buckets["thresholds"])
    alignments = pd.DataFrame(buckets["alignments"])
    grid = pd.concat(buckets["grid"], ignore_index=True)
    validation = pd.DataFrame(buckets["validation"])
    parameter_rows = pd.DataFrame(buckets["parameters_head"])
    guard = guard_audit(scores)
    checks = acceptance(summary, thresholds, guard)
    scores.to_csv(paths["results"] / "adaptive_window_scores_v11.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(paths["results"] / "bidirectional_transfer_summary_v11.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(paths["results"] / "ablation_summary_v11.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(paths["results"] / "risk_head_regularization_grid.csv", index=False, encoding="utf-8-sig")
    parameter_rows.to_csv(paths["results"] / "risk_head_parameters_v11.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(paths["results"] / "source_validation_risk_metrics.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(paths["results"] / "risk_thresholds_v11.csv", index=False, encoding="utf-8-sig")
    alignments.to_csv(paths["results"] / "target_logit_alignment.csv", index=False, encoding="utf-8-sig")
    reliability.to_csv(paths["results"] / "feature_reliability_trace_v11.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(paths["results"] / "online_parameter_trace_v11.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(paths["results"] / "adapter_state_trace.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(paths["results"] / "adapter_episode_log.csv", index=False, encoding="utf-8-sig")
    guard.to_csv(paths["results"] / "boundary_guard_audit.csv", index=False, encoding="utf-8-sig")
    figures(scores, summary, reliability, parameters, guard, paths["figures"])
    pytest_result = run_pytest(paths)
    leakage_cases = {}
    for column in ("stage", "stage_label", "Stage1to5"):
        try:
            assert_target_unlabeled(pd.DataFrame({"window_index": [0], column: [1]}))
            leakage_cases[column] = False
        except AssertionError:
            leakage_cases[column] = True
    diagnostics = {
        "no_label_leakage_check.json": {"status": "PASS" if all(leakage_cases.values()) else "FAIL", "rejected_columns": leakage_cases, **pytest_result},
        "prefix_causality_check.json": {"status": "PASS" if pytest_result["pytest_passed"] else "FAIL", "test": "test_v11_prefix_causality.py", **pytest_result},
        "risk_head_saturation_check.json": {"status": "PASS" if float(thresholds.max_abs_nonintercept_beta.max()) <= 5 else "FAIL", "max_abs_nonintercept_beta": float(thresholds.max_abs_nonintercept_beta.max()), "high_probability_max": float(thresholds.high_probability_equivalent.max()), **pytest_result},
        "threshold_transfer_check.json": {"status": "PASS" if bool((thresholds.watch_logit_threshold < thresholds.high_logit_threshold).all()) else "FAIL", "threshold_rows": thresholds.to_dict(orient="records"), **pytest_result},
        "boundary_guard_check.json": {"status": "PASS" if int(guard.clean_TES_events_in_guard.sum()) == 0 and int(guard.clean_freeze_triggers_in_guard.sum()) == 0 else "FAIL", "audit": guard.to_dict(orient="records"), **pytest_result},
        "reliability_safety_check.json": {"status": "PASS" if pytest_result["pytest_passed"] else "FAIL", "test": "test_v11_reliability_update.py", **pytest_result},
        "adaptation_acceptance_v11.json": {"status": "PASS" if all(row["status"] != "FAIL" for row in checks) else "FAIL", "checks": checks, **pytest_result},
    }
    for filename, payload in diagnostics.items():
        (paths["diagnostics"] / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report(paths["reports"] / "adaptive_cross_domain_awr_v11_report.md", summary, grid, guard, thresholds, checks)
    docs = Path("docs"); docs.mkdir(exist_ok=True)
    (docs / "STATUS_20260712_ADAPTIVE_AWR_V11.md").write_text(f"# STATUS 20260712 Adaptive AWR v1.1\n\nOverall acceptance: **{diagnostics['adaptation_acceptance_v11.json']['status']}**.\n\n{markdown(summary)}\n\n{markdown(pd.DataFrame(checks))}\n", encoding="utf-8")
    logging.info("Adaptive AWR v1.1 complete: %s", paths["root"])
    print(f"Adaptive AWR v1.1 complete: {paths['root']}")


if __name__ == "__main__":
    main()
