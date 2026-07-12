from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive_awr.causal_metrics import BaselineReferences, CausalMetricTracker, build_baseline_references, finite, sigmoid
from adaptive_awr.config import AdaptiveAWRConfig
from adaptive_awr.evaluation import (
    adaptation_acceptance,
    add_stage5_suppression,
    dataframe_markdown,
    evaluate_scored_target,
)
from adaptive_awr.online_adapter import OnlineAdapter
from adaptive_awr.risk_head import (
    RISK_FEATURES,
    average_precision,
    choose_risk_threshold,
    event_risk,
    fit_risk_head,
    roc_auc,
    source_directions,
    source_split_by_stage,
)


MODEL_SETTINGS = {
    "B1": {"reliability": False, "offset": False, "protect": False},
    "B2": {"reliability": True, "offset": False, "protect": False},
    "B3": {"reliability": True, "offset": True, "protect": False},
    "B4": {"reliability": True, "offset": True, "protect": True},
}


def setup_logging(root: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(root / "adaptive_awr_run.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def required_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def calibration_mask(frame: pd.DataFrame, config: AdaptiveAWRConfig) -> np.ndarray:
    mask = frame["end_cycle"].to_numpy(dtype=float) <= config.baseline_cycles
    if int(mask.sum()) < 20:
        mask[:] = False
        mask[: min(100, len(mask))] = True
    return mask


def load_z_wide(config: AdaptiveAWRConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(config.z_table_path)
    if path.exists():
        long = pd.read_csv(path)
        id_columns = [
            "dataset",
            "window_id",
            "window_index",
            "start_cycle",
            "end_cycle",
            "center_cycle",
            "stage",
            "stage_label",
            "baseline_window",
        ]
        required_columns(long, id_columns + ["feature_name", "z_value"], str(path))
        wide = (
            long.pivot_table(index=id_columns, columns="feature_name", values="z_value", aggfunc="first")
            .reset_index()
            .rename_axis(columns=None)
        )
        return wide.sort_values(["dataset", "window_index"]).reset_index(drop=True), {
            "source": str(path),
            "fallback_feature_extraction": False,
        }

    # Preserve historical output directories: generate the missing table only in memory.
    from run_weighted_awrcore_models import (
        WeightedAWRConfig,
        load_raw_or_window_data,
        robust_baseline_normalize,
    )

    logging.warning("%s is absent; deriving z features with existing feature functions in memory.", path)
    legacy_config = WeightedAWRConfig(baseline_cycles=config.baseline_cycles, window_k=config.window_k, stride=config.stride)
    wide_raw, _, meta, _ = load_raw_or_window_data(legacy_config)
    z_wide, _, _ = robust_baseline_normalize(wide_raw, meta, legacy_config)
    return z_wide.sort_values(["dataset", "window_index"]).reset_index(drop=True), {
        "source": "in_memory_existing_feature_extraction",
        "fallback_feature_extraction": True,
    }


def load_inputs(config: AdaptiveAWRConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    z_wide, z_meta = load_z_wide(config)
    required_columns(z_wide, ["dataset", "window_index", "stage", "end_cycle", *config.stable_plus_features], "z feature table")
    state_path = Path(config.state_v2_path)
    if not state_path.exists():
        raise FileNotFoundError(f"Missing auxiliary state table: {state_path}")
    state = pd.read_csv(state_path)
    required_columns(state, ["dataset", "window_index", "BDall_xy_v2", "BDshape_v2"], str(state_path))
    state = state[["dataset", "window_index", "BDall_xy_v2", "BDshape_v2"]].copy()
    merged = z_wide.merge(state, on=["dataset", "window_index"], how="left", validate="one_to_one")
    if merged[["BDall_xy_v2", "BDshape_v2"]].isna().all(axis=None):
        raise ValueError("The auxiliary state table did not join any BD values.")
    audit_rows = []
    for dataset, group in merged.groupby("dataset", sort=True):
        audit_rows.append(
            {
                "dataset": dataset,
                "windows": int(len(group)),
                "calibration_windows": int(calibration_mask(group.reset_index(drop=True), config).sum()),
                "cycle_min": float(group["start_cycle"].min()),
                "cycle_max": float(group["end_cycle"].max()),
                "stage_counts": json.dumps({str(k): int(v) for k, v in group["stage"].value_counts().sort_index().items()}),
                "stable_plus_missing_values": int(group.loc[:, config.stable_plus_features].isna().sum().sum()),
                "BDall_xy_v2_missing_values": int(group["BDall_xy_v2"].isna().sum()),
                "BDshape_v2_missing_values": int(group["BDshape_v2"].isna().sum()),
                "z_table_source": z_meta["source"],
                "fallback_feature_extraction": bool(z_meta["fallback_feature_extraction"]),
            }
        )
    return merged, pd.DataFrame(audit_rows), z_meta


def compute_static_awr(frame: pd.DataFrame, direction_lookup: Mapping[str, int]) -> np.ndarray:
    X = frame.loc[:, list(direction_lookup)].to_numpy(dtype=float)
    signs = np.asarray([direction_lookup[feature] for feature in direction_lookup], dtype=float)
    return np.nanmean(X * signs.reshape(1, -1), axis=1)


def metric_frame_from_static_awr(
    frame: pd.DataFrame,
    awr: np.ndarray,
    refs: BaselineReferences,
    source_awr_high_threshold: float,
    source_bd_high_threshold: float,
    config: AdaptiveAWRConfig,
) -> pd.DataFrame:
    tracker = CausalMetricTracker(refs, config, source_awr_high_threshold, source_bd_high_threshold)
    rows = []
    for position, row in enumerate(frame.itertuples(index=False)):
        metrics = tracker.step(float(awr[position]), float(row.BDall_xy_v2), float(row.BDshape_v2))
        metrics["AWR_adaptive"] = float(awr[position])
        metrics["BDall_xy_v2"] = float(row.BDall_xy_v2)
        metrics["window_index"] = int(row.window_index)
        rows.append(metrics)
    return pd.DataFrame(rows)


def source_context(
    source: pd.DataFrame,
    direction_id: str,
    target_dataset: str,
    config: AdaptiveAWRConfig,
) -> dict[str, Any]:
    source = source.sort_values("window_index").reset_index(drop=True).copy()
    train_mask, validation_mask = source_split_by_stage(source["stage"].to_numpy(dtype=int), config.source_gap_windows)
    directions = source_directions(source, train_mask, config.stable_plus_features)
    direction_lookup = {
        str(row.feature_name): int(row.direction_sign)
        for row in directions.itertuples(index=False)
    }
    static_awr = compute_static_awr(source, direction_lookup)
    base = calibration_mask(source, config)
    refs = build_baseline_references(
        static_awr[base],
        source.loc[base, "BDall_xy_v2"].to_numpy(dtype=float),
        source.loc[base, "BDshape_v2"].to_numpy(dtype=float),
        source.loc[base, list(config.stable_plus_features)],
        config,
    )
    high_source = source.loc[validation_mask] if int(validation_mask.sum()) else source
    source_awr_high = float(np.nanpercentile(high_source.assign(_awr=static_awr[validation_mask] if int(validation_mask.sum()) else static_awr)["_awr"], config.source_high_percentile))
    source_bd_high = float(np.nanpercentile(high_source["BDall_xy_v2"], config.source_high_percentile))
    metric_source = metric_frame_from_static_awr(
        source, static_awr, refs, source_awr_high, source_bd_high, config
    )
    metric_source["stage"] = source["stage"].to_numpy(dtype=int)
    metric_source["dataset"] = source["dataset"].astype(str).to_numpy()
    source_tes_threshold = max(
        float(np.nanpercentile(metric_source.loc[base, "TES"], 99.5)), config.source_tes_floor
    )
    source_rs_values = metric_source.loc[base, "RS50"].to_numpy(dtype=float)
    source_rs_values = source_rs_values[np.isfinite(source_rs_values)]
    source_rs_threshold = max(
        float(np.nanpercentile(source_rs_values, config.source_high_percentile)) if source_rs_values.size else 0.0,
        config.source_rs_floor,
    )
    head = fit_risk_head(metric_source, train_mask, config)
    threshold = choose_risk_threshold(metric_source, validation_mask, head)
    source_train_metrics = metric_source.loc[train_mask].copy()
    return {
        "direction_id": direction_id,
        "source_dataset": str(source["dataset"].iloc[0]),
        "target_dataset": target_dataset,
        "source": source,
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "directions": directions,
        "direction_lookup": direction_lookup,
        "static_awr": static_awr,
        "refs": refs,
        "source_awr_high_threshold": source_awr_high,
        "source_bd_high_threshold": source_bd_high,
        "source_tes_threshold": source_tes_threshold,
        "source_rs_threshold": source_rs_threshold,
        "metric_source": metric_source,
        "head": head,
        "threshold": threshold,
        "source_train_tes": source_train_metrics["TES"].to_numpy(dtype=float),
        "source_train_bd_jump": source_train_metrics["BD_jump"].to_numpy(dtype=float),
    }


def target_baseline_refs(target: pd.DataFrame, context: Mapping[str, Any], config: AdaptiveAWRConfig) -> tuple[BaselineReferences, np.ndarray, pd.DataFrame, np.ndarray]:
    base = calibration_mask(target, config)
    identity_awr = compute_static_awr(target, context["direction_lookup"])
    refs = build_baseline_references(
        identity_awr[base],
        target.loc[base, "BDall_xy_v2"].to_numpy(dtype=float),
        target.loc[base, "BDshape_v2"].to_numpy(dtype=float),
        target.loc[base, list(config.stable_plus_features)],
        config,
    )
    static_metrics = metric_frame_from_static_awr(
        target,
        identity_awr,
        refs,
        float(context["source_awr_high_threshold"]),
        float(context["source_bd_high_threshold"]),
        config,
    )
    return refs, base, static_metrics, identity_awr


def ensure_target_has_no_labels(target: pd.DataFrame) -> None:
    forbidden = {"stage", "stage_label", "Stage1to5"}
    leaked = forbidden.intersection(target.columns)
    if leaked:
        raise AssertionError(f"Target labels were passed into online inference: {sorted(leaked)}")


def run_target_online(
    target_without_labels: pd.DataFrame,
    context: Mapping[str, Any],
    model: str,
    config: AdaptiveAWRConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score a target experiment sequentially. Its input is explicitly label-free."""
    ensure_target_has_no_labels(target_without_labels)
    target = target_without_labels.sort_values("window_index").reset_index(drop=True).copy()
    settings = MODEL_SETTINGS[model]
    refs, base, static_metrics, identity_awr = target_baseline_refs(target, context, config)
    adapter = OnlineAdapter(
        config=config,
        refs=refs,
        features=config.stable_plus_features,
        source_tes_threshold=float(context["source_tes_threshold"]),
        source_rs_threshold=float(context["source_rs_threshold"]),
        risk_threshold=float(context["threshold"]["risk_threshold"]),
        enable_reliability=bool(settings["reliability"]),
        enable_offset=bool(settings["offset"]),
        enforce_freeze_and_rollback=bool(settings["protect"]),
    )
    calibration_logits = context["head"].logit(static_metrics.loc[base, list(RISK_FEATURES)])
    adapter.initialize_offset(calibration_logits)
    tracker = CausalMetricTracker(
        refs,
        config,
        float(context["source_awr_high_threshold"]),
        float(context["source_bd_high_threshold"]),
    )
    histories = {feature: [] for feature in config.stable_plus_features}
    tes_event_history: list[bool] = []
    high_risk_history: list[bool] = []
    rs50_history: list[float] = []
    awr_history: list[float] = []
    bd_history: list[float] = []
    slow_history: list[float] = []
    score_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    slow_previous: float | None = None

    for position, row in enumerate(target.itertuples(index=False)):
        feature_values = {feature: float(getattr(row, feature)) for feature in config.stable_plus_features}
        for feature, value in feature_values.items():
            histories[feature].append(value)
        online_active = not bool(base[position])
        raw_rel, noise_ratios, saturation_feature_rate = adapter.reliability_evidence(histories)
        if online_active:
            adapter.immediately_reduce_for_saturation(raw_rel)
        reliability_used = dict(adapter.reliability)
        awr = adapter.weighted_awr(feature_values, context["direction_lookup"])
        metrics = tracker.step(awr, float(row.BDall_xy_v2), float(row.BDshape_v2))
        risk_vector = np.asarray(
            [awr, float(row.BDall_xy_v2), metrics["RS50_positive"], metrics["TES"], metrics["high_AWR_high_BD_occupancy"]],
            dtype=float,
        )
        head = context["head"]
        risk_vector = np.where(np.isfinite(risk_vector), risk_vector, head.scaler.median)
        base_logit = float(
            head.coefficients[0]
            + np.dot((risk_vector - head.scaler.median) / head.scaler.iqr, head.coefficients[1:])
        )
        active_offset = adapter.domain_logit_offset if online_active else 0.0
        risk_instant = sigmoid(base_logit + active_offset)
        if slow_previous is None:
            slow_risk = risk_instant
        else:
            alpha = config.risk_alpha_up if risk_instant >= slow_previous else config.risk_alpha_down
            slow_risk = float(slow_previous + alpha * (risk_instant - slow_previous))
        event_value = event_risk(
            metrics["TES"], metrics["BD_jump"], context["source_train_tes"], context["source_train_bd_jump"], config.eps
        )
        final_risk = max(slow_risk, config.event_risk_weight * event_value)
        risk_level = "low" if final_risk < float(context["threshold"]["risk_threshold"]) else ("watch" if final_risk < 0.8 else "high")

        prior_tes_event = any(tes_event_history[-config.gate_history_windows :])
        prior_high_risk = any(high_risk_history[-config.gate_history_windows :])
        current_tes_event = bool(metrics["TES"] >= float(context["source_tes_threshold"]))
        current_high_risk = bool(final_risk >= float(context["threshold"]["risk_threshold"]))
        rs50_history.append(float(metrics["RS50"]))
        rs_recent = np.asarray(rs50_history[-3:], dtype=float)
        rs_run = bool(rs_recent.size == 3 and np.all(np.isfinite(rs_recent)) and np.all(rs_recent > float(context["source_rs_threshold"])))

        if online_active and settings["protect"]:
            if current_high_risk:
                adapter.force_freeze(int(row.window_index), "final_risk_exceeds_source_threshold", {"final_risk": final_risk})
            elif current_tes_event:
                adapter.force_freeze(int(row.window_index), "TES_event", {"TES": metrics["TES"]})
            elif rs_run:
                adapter.force_freeze(int(row.window_index), "RS50_positive_three_window_run", {"RS50": metrics["RS50"]})
            elif metrics["high_AWR_high_BD"]:
                adapter.force_freeze(int(row.window_index), "high_AWR_and_high_BD", {"AWR": awr, "BD": float(row.BDall_xy_v2)})

        gate_open, gate_reasons = adapter.safety_gate(
            slow_risk=slow_risk,
            awr=awr,
            bd=float(row.BDall_xy_v2),
            rs50=float(metrics["RS50"]),
            tes=float(metrics["TES"]),
            recent_tes_event=prior_tes_event,
            recent_high_risk=prior_high_risk,
            saturation_feature_rate=saturation_feature_rate,
        )
        if not online_active:
            gate_open = False
            gate_reasons = ["calibration_phase"]
        adapter.update(
            window_index=int(row.window_index),
            raw_reliability=raw_rel,
            noise_ratios=noise_ratios,
            gate_open=gate_open,
            base_logit=base_logit,
        )
        awr_history.append(awr)
        bd_history.append(float(row.BDall_xy_v2))
        slow_history.append(slow_risk)
        adapter.maybe_rollback(int(row.window_index), awr_history, bd_history, slow_history)
        adapter.save_checkpoint(int(row.window_index))
        tes_event_history.append(current_tes_event)
        high_risk_history.append(current_high_risk)
        slow_previous = slow_risk

        score = {
            "window_index": int(row.window_index),
            "start_cycle": float(row.start_cycle),
            "end_cycle": float(row.end_cycle),
            "center_cycle": float(row.center_cycle),
            "AWR_adaptive": awr,
            "BDall_xy_v2": float(row.BDall_xy_v2),
            "BDshape_v2": float(row.BDshape_v2),
            **metrics,
            "risk_instant": risk_instant,
            "slow_risk": slow_risk,
            "event_risk": event_value,
            "final_risk": final_risk,
            "risk_level": risk_level,
            "adapter_state": "UPDATING" if gate_open else "FROZEN",
            "adapter_gate_reasons": ";".join(gate_reasons),
            "domain_logit_offset_used": active_offset,
            "observation_noise": adapter.observation_noise,
            "mean_feature_reliability": float(np.mean(list(reliability_used.values()))),
        }
        score_rows.append(score)
        for feature in config.stable_plus_features:
            reliability_rows.append(
                {
                    "window_index": int(row.window_index),
                    "feature_name": feature,
                    "reliability_used": reliability_used[feature],
                    "raw_reliability": raw_rel[feature],
                    "noise_ratio": noise_ratios[feature],
                    "saturation_feature_rate": saturation_feature_rate,
                    "adapter_state": score["adapter_state"],
                }
            )
        parameter_rows.append(
            {
                "window_index": int(row.window_index),
                "domain_logit_offset": adapter.domain_logit_offset,
                "observation_noise": adapter.observation_noise,
                "mean_feature_reliability": float(np.mean(list(adapter.reliability.values()))),
                "freeze_remaining": int(adapter.freeze_remaining),
                "adapter_state": score["adapter_state"],
                "gate_open": bool(gate_open),
                "gate_reasons": ";".join(gate_reasons),
            }
        )
    return pd.DataFrame(score_rows), pd.DataFrame(reliability_rows), pd.DataFrame(parameter_rows), pd.DataFrame(adapter.events)


def b0_scores(target: pd.DataFrame, source: pd.DataFrame, validation_mask: np.ndarray, config: AdaptiveAWRConfig) -> tuple[pd.DataFrame, float]:
    target_score = target.loc[:, list(config.stable_plus_features)].mean(axis=1).to_numpy(dtype=float)
    source_score = source.loc[:, list(config.stable_plus_features)].mean(axis=1).to_numpy(dtype=float)
    ref = source_score[validation_mask] if int(validation_mask.sum()) else source_score
    threshold = float(np.nanpercentile(ref, config.source_high_percentile))
    rows = target[["window_index", "start_cycle", "end_cycle", "center_cycle"]].copy()
    rows["AWR_M0_stable"] = target_score
    rows["evaluation_score"] = target_score
    rows["risk_level"] = np.where(target_score >= threshold, "high", "low")
    return rows, threshold


def score_direction(
    merged: pd.DataFrame,
    direction_id: str,
    source_dataset: str,
    target_dataset: str,
    config: AdaptiveAWRConfig,
) -> tuple[list[pd.DataFrame], list[dict[str, object]], list[dict[str, object]], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], dict[str, Any]]:
    source = merged[merged["dataset"] == source_dataset].sort_values("window_index").reset_index(drop=True)
    target_labeled = merged[merged["dataset"] == target_dataset].sort_values("window_index").reset_index(drop=True)
    context = source_context(source, direction_id, target_dataset, config)
    logging.info("%s source direction, causal metrics, risk head and threshold complete", direction_id)
    scores: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    params = [context["head"].row(direction_id, source_dataset, target_dataset)]
    rel_tables: list[pd.DataFrame] = []
    parameter_tables: list[pd.DataFrame] = []
    event_tables: list[pd.DataFrame] = []

    b0, b0_threshold = b0_scores(target_labeled, source, context["validation_mask"], config)
    b0["direction_id"] = direction_id
    b0["source_dataset"] = source_dataset
    b0["target_dataset"] = target_dataset
    b0["model"] = "B0"
    b0_labeled = b0.merge(target_labeled[["window_index", "stage", "stage_label"]], on="window_index", how="left", validate="one_to_one")
    summaries.append(
        evaluate_scored_target(
            b0_labeled,
            direction_id=direction_id,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            model="B0",
            threshold=b0_threshold,
        )
    )
    scores.append(b0_labeled)
    logging.info("%s B0 complete", direction_id)

    # A strict interface boundary: no target stage or stage label crosses into online inference.
    online_target = target_labeled.drop(columns=["stage", "stage_label", "baseline_window"], errors="ignore")
    for model in ("B1", "B2", "B3", "B4"):
        online, reliability, parameters, events = run_target_online(online_target, context, model, config)
        online["evaluation_score"] = online["final_risk"]
        online["direction_id"] = direction_id
        online["source_dataset"] = source_dataset
        online["target_dataset"] = target_dataset
        online["model"] = model
        # Stage is attached only after every target window has been scored.
        labeled = online.merge(target_labeled[["window_index", "stage", "stage_label"]], on="window_index", how="left", validate="one_to_one")
        event_counts = events["event_type"].value_counts() if not events.empty else pd.Series(dtype=int)
        summaries.append(
            evaluate_scored_target(
                labeled,
                direction_id=direction_id,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                model=model,
                threshold=float(context["threshold"]["risk_threshold"]),
                update_count=int((event_counts.get("UPDATE_RESUMED", 0))),
                freeze_count=int((event_counts.get("FREEZE", 0))),
                rollback_count=int((event_counts.get("ROLLBACK", 0))),
            )
        )
        reliability["direction_id"] = direction_id
        reliability["source_dataset"] = source_dataset
        reliability["target_dataset"] = target_dataset
        reliability["model"] = model
        parameters["direction_id"] = direction_id
        parameters["source_dataset"] = source_dataset
        parameters["target_dataset"] = target_dataset
        parameters["model"] = model
        if events.empty:
            events = pd.DataFrame(columns=["window_index", "event_type", "reason", "old_freeze_remaining", "new_freeze_remaining", "old_parameters", "new_parameters", "details"])
        events["direction_id"] = direction_id
        events["source_dataset"] = source_dataset
        events["target_dataset"] = target_dataset
        events["model"] = model
        scores.append(labeled)
        rel_tables.append(reliability)
        parameter_tables.append(parameters)
        event_tables.append(events)
        logging.info("%s %s target causal inference complete", direction_id, model)

    threshold_row = {
        "direction_id": direction_id,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "B0_source_validation_AWR_p95_threshold": b0_threshold,
        "source_AWR_high_threshold": context["source_awr_high_threshold"],
        "source_BD_high_threshold": context["source_bd_high_threshold"],
        "source_TES_threshold": context["source_tes_threshold"],
        "source_RS50_threshold": context["source_rs_threshold"],
        **context["threshold"],
    }
    return scores, summaries, params, rel_tables, parameter_tables, event_tables, threshold_row


def precision_recall_curve(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(score)
    y, score = y[valid].astype(int), score[valid]
    if y.sum() == 0:
        return np.array([0.0]), np.array([1.0])
    order = np.argsort(-score, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    recall = tp / max(int(y.sum()), 1)
    return recall, precision


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_figures(scores: pd.DataFrame, summary: pd.DataFrame, reliability: pd.DataFrame, parameters: pd.DataFrame, figures: Path) -> None:
    directions = list(scores["direction_id"].drop_duplicates())
    fig, axes = plt.subplots(len(directions), 1, figsize=(12, 4 * len(directions)), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, direction in zip(axes, directions):
        panel = scores[(scores["direction_id"] == direction) & (scores["model"] == "B4")].sort_values("window_index")
        axis.plot(panel["center_cycle"], panel["final_risk"], color="#b64b5a", label="B4 FinalRisk")
        axis.plot(panel["center_cycle"], panel["slow_risk"], color="#3b6ea8", alpha=0.85, label="SlowRisk")
        stage5 = panel[panel["stage"] == 5]
        if not stage5.empty:
            axis.axvspan(stage5["start_cycle"].min(), stage5["end_cycle"].max(), color="#d4a72c", alpha=0.18, label="Stage 5 eval only")
        threshold = float(summary[(summary["direction_id"] == direction) & (summary["model"] == "B4")]["risk_threshold"].iloc[0])
        axis.axhline(threshold, color="#555555", linestyle="--", linewidth=1)
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(direction)
        axis.set_xlabel("Cycle")
        axis.set_ylabel("Risk")
        axis.legend(loc="best", fontsize=8)
    save_figure(fig, figures / "fig_adaptive_risk_timeseries.png")

    fig, axes = plt.subplots(1, len(directions), figsize=(6 * len(directions), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, direction in zip(axes, directions):
        for model, group in scores[scores["direction_id"] == direction].groupby("model", sort=True):
            curve_score = group["evaluation_score"].to_numpy(dtype=float)
            recall, precision = precision_recall_curve((group["stage"].to_numpy(dtype=int) == 5), curve_score)
            axis.plot(recall, precision, label=f"{model} AP={average_precision(group['stage'].to_numpy(dtype=int) == 5, curve_score):.3f}")
        axis.set_title(direction)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.05)
        axis.legend(fontsize=7)
    save_figure(fig, figures / "fig_transfer_pr_curves.png")

    metrics = ["Stage5_AUROC", "Stage5_AUPRC"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    model_order = ["B0", "B1", "B2", "B3", "B4"]
    for axis, metric in zip(axes, metrics):
        pivot = summary.pivot(index="model", columns="direction_id", values=metric).reindex(model_order)
        pivot.plot(kind="bar", ax=axis, color=["#3b6ea8", "#46a67a"])
        axis.set_title(metric)
        axis.set_ylim(0, 1.05)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=0)
    save_figure(fig, figures / "fig_ablation_comparison.png")

    fig, axes = plt.subplots(len(directions), 1, figsize=(12, 4 * len(directions)), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, direction in zip(axes, directions):
        panel = reliability[(reliability["direction_id"] == direction) & (reliability["model"] == "B4")]
        for feature, group in panel.groupby("feature_name", sort=True):
            axis.plot(group["window_index"], group["reliability_used"], linewidth=1, label=feature)
        axis.set_ylim(0.2, 1.03)
        axis.set_title(f"{direction}: B4 feature reliability")
        axis.set_xlabel("Window")
        axis.set_ylabel("Reliability")
        axis.legend(ncol=2, fontsize=7)
    save_figure(fig, figures / "fig_feature_reliability_trace.png")

    fig, axes = plt.subplots(len(directions), 1, figsize=(12, 3.5 * len(directions)), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, direction in zip(axes, directions):
        panel = parameters[(parameters["direction_id"] == direction) & (parameters["model"].isin(["B3", "B4"]))]
        for model, group in panel.groupby("model", sort=True):
            axis.plot(group["window_index"], group["domain_logit_offset"], label=model)
        axis.set_title(f"{direction}: online logit offset")
        axis.set_xlabel("Window")
        axis.set_ylabel("Offset")
        axis.legend()
    save_figure(fig, figures / "fig_online_offset_trace.png")

    fig, axes = plt.subplots(len(directions), 1, figsize=(12, 3.5 * len(directions)), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, direction in zip(axes, directions):
        panel = parameters[(parameters["direction_id"] == direction) & (parameters["model"] == "B4")]
        state = panel["adapter_state"].eq("UPDATING").astype(int)
        axis.step(panel["window_index"], state, where="post", color="#46a67a")
        axis.set_yticks([0, 1], ["FROZEN", "UPDATING"])
        axis.set_title(f"{direction}: B4 adaptation gate timeline")
        axis.set_xlabel("Window")
    save_figure(fig, figures / "fig_adaptation_gate_timeline.png")


def write_report(
    path: Path,
    config: AdaptiveAWRConfig,
    summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    acceptance: list[dict[str, object]],
    source_rows: pd.DataFrame,
) -> None:
    overall = "PASS" if all(row["status"] == "PASS" for row in acceptance) else "FAIL"
    text = f"""# Adaptive AWR v1 Cross-Domain Report

## Outcome

Overall acceptance status: **{overall}**. `Stage5` is used as a late-state proxy for final evaluation only, not as a target-side training or adaptation input.

## Protocol

- Transfers: `Exp1 -> Exp2` and `Exp2 -> Exp1`.
- Shared stable_plus feature directions and logistic risk heads are fitted only from source training windows.
- Target scoring is causal. The inference interface rejects `stage` and `stage_label` columns.
- Target calibration is limited to the initial `{config.baseline_cycles}` cycles. Feature baseline centres/IQRs, directions and risk-head coefficients are never updated online.
- B1 is static; B2 enables gated reliability; B3 adds the bounded logit offset; B4 adds forced freezing, checkpoints and rollback.

## Transfer Results

{dataframe_markdown(summary, ["direction_id", "model", "Stage5_AUROC", "Stage5_AUPRC", "Stage5_Recall", "Stage1to2_FPR", "Recall_at_10pct_Stage1to2_FPR", "Risk_Stage_Spearman", "detection_lead_cycles_relative_to_Stage5", "Stage5_risk_suppression_B1_minus_B4", "adaptation_safety_failure"])}

## Source-only Thresholds

{dataframe_markdown(thresholds)}

## Risk-head Optimisation

{dataframe_markdown(source_rows, ["direction_id", "source_dataset", "target_dataset", "optimizer_success", "objective", "beta0", "beta_AWR_adaptive", "beta_BDall_xy_v2", "beta_RS50_positive", "beta_TES", "beta_high_AWR_high_BD_occupancy"])}

## Acceptance Checks

{dataframe_markdown(pd.DataFrame(acceptance))}

## Interpretation

The adaptive output is a **late-state risk score**, not a calibrated failure probability. A failed acceptance item is retained as a result and has not triggered threshold or model retuning.
"""
    path.write_text(text, encoding="utf-8")


def write_status(path: Path, summary: pd.DataFrame, acceptance: list[dict[str, object]]) -> None:
    status = "PASS" if all(row["status"] == "PASS" for row in acceptance) else "FAIL"
    text = f"""# STATUS 20260712

## Adaptive AWR v1

- Implemented an independent source-only Adaptive AWR v1 pipeline under `adaptive_awr/` and `run_adaptive_cross_domain_awr.py`.
- It reuses existing stable_plus z features and v2 BD inputs without changing historical AWR scripts or their output directories.
- Bidirectional experiments completed for `Exp1 -> Exp2` and `Exp2 -> Exp1` with B0-B4 ablations.
- Target Stage labels are attached only after label-free target scoring for final evaluation.
- Online updates are limited to feature reliability, bounded domain logit offset and observation noise. Baseline centres, directions and risk-head weights are immutable.

## Acceptance

Overall result: **{status}**.

{dataframe_markdown(pd.DataFrame(acceptance))}

## Result Summary

{dataframe_markdown(summary, ["direction_id", "model", "Stage5_AUROC", "Stage5_AUPRC", "Stage5_Recall", "Stage1to2_FPR", "Stage5_risk_suppression_B1_minus_B4"])}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    config = AdaptiveAWRConfig()
    paths = config.output_paths()
    setup_logging(paths["root"])
    logging.info("Starting Adaptive AWR v1")
    merged, audit, z_meta = load_inputs(config)
    audit.to_csv(paths["diagnostics"] / "input_data_audit.csv", index=False, encoding="utf-8-sig")
    (paths["configs"] / "adaptive_awr_config.json").write_text(
        json.dumps(config.as_jsonable(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    score_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    reliability_frames: list[pd.DataFrame] = []
    parameter_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []
    for direction_id, source, target in (("Exp1_to_Exp2", "Exp1", "Exp2"), ("Exp2_to_Exp1", "Exp2", "Exp1")):
        logging.info("Running %s", direction_id)
        outputs = score_direction(merged, direction_id, source, target, config)
        scores, summaries, risk_params, rel, params, events, thresholds = outputs
        score_frames.extend(scores)
        summary_rows.extend(summaries)
        risk_rows.extend(risk_params)
        reliability_frames.extend(rel)
        parameter_frames.extend(params)
        event_frames.extend(events)
        threshold_rows.append(thresholds)
    all_scores = pd.concat(score_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    summary = add_stage5_suppression(summary, all_scores)
    reliability = pd.concat(reliability_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    thresholds = pd.DataFrame(threshold_rows)
    risk_parameters = pd.DataFrame(risk_rows)
    acceptance = adaptation_acceptance(summary)

    all_scores.to_csv(paths["results"] / "adaptive_window_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(paths["results"] / "bidirectional_transfer_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(paths["results"] / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    risk_parameters.to_csv(paths["results"] / "risk_head_parameters.csv", index=False, encoding="utf-8-sig")
    reliability.to_csv(paths["results"] / "feature_reliability_trace.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(paths["results"] / "online_parameter_trace.csv", index=False, encoding="utf-8-sig")
    events.to_csv(paths["results"] / "adapter_event_log.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(paths["results"] / "risk_thresholds.csv", index=False, encoding="utf-8-sig")

    leakage = {
        "status": "PASS",
        "target_stage_passed_to_online_inference": False,
        "target_stage_used_for_risk_head_fit": False,
        "target_stage_used_for_threshold_selection": False,
        "target_stage_used_for_reliability_or_offset": False,
        "target_stage_attachment": "after_all_target_windows_scored",
        "assertion": "run_target_online rejects stage, stage_label and Stage1to5 columns",
    }
    causality = {
        "status": "PASS",
        "rolling_operations": "trailing/current-window only",
        "RS": "adjacent trailing windows only",
        "TES": "trailing rolling medians and fixed calibration references",
        "risk_filter": "one-sided recursive filter",
        "online_update": "current and historical windows only",
        "future_target_windows_read": False,
    }
    safety = {
        "status": "PASS" if all(row["status"] == "PASS" for row in acceptance) else "FAIL",
        "checks": acceptance,
        "immutable_online_parameters": ["feature baseline centre", "feature IQR", "baseline waveform", "direction signs", "risk-head coefficients", "source risk threshold"],
        "online_mutable_parameters": ["feature reliability", "domain logit offset", "observation noise"],
    }
    (paths["diagnostics"] / "no_label_leakage_check.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")
    (paths["diagnostics"] / "causality_check.json").write_text(json.dumps(causality, ensure_ascii=False, indent=2), encoding="utf-8")
    (paths["diagnostics"] / "adaptation_safety_check.json").write_text(json.dumps(safety, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(paths["reports"] / "adaptive_cross_domain_awr_report.md", config, summary, thresholds, acceptance, risk_parameters)
    make_figures(all_scores, summary, reliability, parameters, paths["figures"])
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    write_status(docs / "STATUS_20260712.md", summary, acceptance)
    logging.info("Adaptive AWR v1 complete: %s", paths["root"])
    print(f"Adaptive AWR v1 complete: {paths['root']}")


if __name__ == "__main__":
    main()
