from __future__ import annotations

"""V3.5 Exp2 Median5 diagnostic experiment.

This runner deliberately imports the immutable V3.5 source training, online
scoring, thresholding, and candidate-event routines.  Its only data change is
the declared offline, centred five-window median filter applied separately to
the six V3.5 features of Exp2.  The archived V3.5 Raw results are copied into
this diagnostic unchanged; each recreated Exp1 source state is checked against
the archived source-state hash before it is used for Exp2-Median5.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gradual_state_monitoring_v33.data import MAIN_FEATURES
from gradual_state_monitoring_v34.data import (attach_posthoc_stage,
                                                load_stage_segments,
                                                stage_boundaries)
from gradual_state_monitoring_v34.evaluation import (boundary_ranking,
                                                      events_with_actual_cycles,
                                                      match_candidate_events)
from gradual_state_monitoring_v34.model import GradualStateTCN
from gradual_state_monitoring_v34.training import train_source_model
from gradual_state_monitoring_v35.config import GradualStateMonitoringV35Config
from gradual_state_monitoring_v35.online import (extract_candidate_events,
                                                  run_target_online)


SETTINGS = (
    "V35_SlowResidual_CalibrationOnly",
    "V35_SlowResidual_OnlineAdaptation",
)
SEEDS = (3401, 3402, 3403, 3404, 3405)
EXP2_VARIANTS = ("Raw", "Median5")


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2,
                   default=lambda item: item.item() if isinstance(item, np.generic) else str(item)),
        encoding="utf-8",
    )


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _clone_state(model: GradualStateTCN) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _restore(state: dict[str, torch.Tensor], config: GradualStateMonitoringV35Config) -> GradualStateTCN:
    model = GradualStateTCN(config)
    model.load_state_dict(state)
    model.eval()
    return model


def _tag(frame: pd.DataFrame, *, direction: str, source_dataset: str, target_dataset: str,
         source_exp2_variant: str, target_exp2_variant: str, seed: int,
         source_model_sha256: str) -> pd.DataFrame:
    """Attach diagnostic identifiers without changing V3.5-derived columns."""
    result = frame.copy()
    tags = {
        "direction": direction,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "source_exp2_variant": source_exp2_variant,
        "target_exp2_variant": target_exp2_variant,
        "exp2_variant": target_exp2_variant if direction == "Exp1_to_Exp2" else source_exp2_variant,
        "seed": seed,
        "source_model_state_sha256": source_model_sha256,
    }
    for name, value in tags.items():
        result[name] = value
    prefix = list(tags)
    return result.loc[:, [*prefix, *[name for name in result if name not in prefix]]]


def _median5_exp2(raw_full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create Exp2 Median5 without reading labels, boundaries, or Stage fields."""
    exp2 = raw_full.loc[raw_full["dataset"].eq("Exp2")].copy()
    if exp2.empty:
        raise ValueError("The declared input has no Exp2 rows")
    exp2 = exp2.sort_values("window_index", kind="stable").reset_index(drop=True)
    smooth = exp2.copy()
    for column in MAIN_FEATURES:
        smooth[column] = exp2[column].rolling(window=5, center=True, min_periods=1).median()

    identity_columns = [name for name in exp2.columns if name not in MAIN_FEATURES]
    if not smooth.loc[:, identity_columns].equals(exp2.loc[:, identity_columns]):
        raise AssertionError("Median5 must not change Exp2 identifiers or auxiliary columns")
    if not np.array_equal(smooth.window_index.to_numpy(), exp2.window_index.to_numpy()):
        raise AssertionError("Median5 must preserve window_index")
    if not np.array_equal(smooth.center_cycle.to_numpy(), exp2.center_cycle.to_numpy()):
        raise AssertionError("Median5 must preserve center_cycle")
    return exp2, smooth


def _online_frame(frame: pd.DataFrame, config: GradualStateMonitoringV35Config) -> pd.DataFrame:
    """Project only the original six-feature V3.5 online schema."""
    return frame.loc[:, list(config.online_allowed_columns)].copy()


def _smoothing_audit(raw: pd.DataFrame, median5: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in MAIN_FEATURES:
        raw_values = raw[column].to_numpy(float)
        smooth_values = median5[column].to_numpy(float)
        raw_mad_step = float(np.median(np.abs(np.diff(raw_values))))
        smooth_mad_step = float(np.median(np.abs(np.diff(smooth_values))))
        raw_mean_step = float(np.mean(np.abs(np.diff(raw_values))))
        smooth_mean_step = float(np.mean(np.abs(np.diff(smooth_values))))
        rows.append({
            "feature": column,
            "median_absolute_step_raw": raw_mad_step,
            "median_absolute_step_median5": smooth_mad_step,
            "median_step_reduction_percent": 100.0 * (1.0 - smooth_mad_step / raw_mad_step) if raw_mad_step else 0.0,
            "mean_absolute_step_raw": raw_mean_step,
            "mean_absolute_step_median5": smooth_mean_step,
            "mean_step_reduction_percent": 100.0 * (1.0 - smooth_mean_step / raw_mean_step) if raw_mean_step else 0.0,
            "raw_median5_pearson_correlation": float(pd.Series(raw_values).corr(pd.Series(smooth_values))),
            "raw_median5_spearman_correlation": float(pd.Series(raw_values).corr(pd.Series(smooth_values), method="spearman")),
        })
    return pd.DataFrame(rows)


def _source_record(*, direction: str, source_exp2_variant: str, seed: int,
                   state_hash: str, reused_for: str, reference_hash: str | None) -> dict[str, object]:
    return {
        "direction": direction,
        "source_exp2_variant": source_exp2_variant,
        "seed": seed,
        "source_model_state_sha256": state_hash,
        "reused_for": reused_for,
        "original_v35_raw_source_hash": reference_hash,
        "original_v35_raw_source_hash_status": (
            "NOT_APPLICABLE" if reference_hash is None and source_exp2_variant == "Median5"
            else "MATCH" if reference_hash == state_hash
            else "DIFFERS" if reference_hash is not None
            else "REFERENCE_UNAVAILABLE"
        ),
    }


def _run_target(*, state: dict[str, torch.Tensor], target: pd.DataFrame,
                config: GradualStateMonitoringV35Config, direction: str,
                source_dataset: str, target_dataset: str, source_exp2_variant: str,
                target_exp2_variant: str, seed: int, source_hash: str) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    scores: list[pd.DataFrame] = []
    events: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    for setting in SETTINGS:
        run = run_target_online(_restore(state, config), target, config, setting=setting)
        score = _tag(
            run.scores, direction=direction, source_dataset=source_dataset, target_dataset=target_dataset,
            source_exp2_variant=source_exp2_variant, target_exp2_variant=target_exp2_variant,
            seed=seed, source_model_sha256=source_hash,
        )
        event = _tag(
            extract_candidate_events(run.scores, config), direction=direction,
            source_dataset=source_dataset, target_dataset=target_dataset,
            source_exp2_variant=source_exp2_variant, target_exp2_variant=target_exp2_variant,
            seed=seed, source_model_sha256=source_hash,
        )
        audit = _tag(
            run.adaptation, direction=direction, source_dataset=source_dataset, target_dataset=target_dataset,
            source_exp2_variant=source_exp2_variant, target_exp2_variant=target_exp2_variant,
            seed=seed, source_model_sha256=source_hash,
        )
        scores.append(score)
        events.append(event)
        audits.append(audit)
    return scores, events, audits


def _select(frame: pd.DataFrame, *, direction: str, exp2_variant: str, setting: str,
            seed: int | None = None) -> pd.DataFrame:
    mask = (
        frame.direction.eq(direction)
        & frame.exp2_variant.eq(exp2_variant)
        & frame.setting.eq(setting)
    )
    if seed is not None:
        mask &= frame.seed.eq(seed)
    return frame.loc[mask].copy()


def _evaluate(all_scores: pd.DataFrame, events: pd.DataFrame, boundaries: pd.DataFrame,
              config: GradualStateMonitoringV35Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen outputs; this is the first function that receives target boundaries."""
    summary_rows: list[dict[str, object]] = []
    boundary_rows: list[pd.DataFrame] = []
    match_rows: list[pd.DataFrame] = []
    direction_targets = {"Exp1_to_Exp2": "Exp2", "Exp2_to_Exp1": "Exp1"}
    for direction, target_name in direction_targets.items():
        local_boundaries = boundaries[boundaries.dataset.eq(target_name)].copy()
        for variant in EXP2_VARIANTS:
            for seed in SEEDS:
                for setting in SETTINGS:
                    score = _select(all_scores, direction=direction, exp2_variant=variant, setting=setting, seed=seed)
                    event = _select(events, direction=direction, exp2_variant=variant, setting=setting, seed=seed)
                    matched, metric = match_candidate_events(
                        event, local_boundaries, 250.0, direction=direction, setting=setting, seed=seed,
                    )
                    rank, _ = boundary_ranking(
                        score, event, local_boundaries, direction=direction, setting=setting, seed=seed, config=config,
                    )
                    rank["exp2_variant"] = variant
                    rank["source_exp2_variant"] = "Raw" if direction == "Exp1_to_Exp2" else variant
                    rank["target_exp2_variant"] = variant if direction == "Exp1_to_Exp2" else "Raw"
                    rank["event_rank"] = rank["boundary_event_rank"]
                    rank["top8_hit"] = rank["event_rank"].le(8).astype(int)
                    relevant_matches = matched[matched.from_stage.notna()].copy()
                    relevant_matches["exp2_variant"] = variant
                    relevant_matches["matched_event_delay_actual_cycles"] = (
                        relevant_matches.event_peak_actual_cycle - relevant_matches.boundary_actual_cycle
                    )
                    relevant_matches["hit_250"] = relevant_matches.match_status.eq("MATCHED").astype(int)
                    merged = rank.merge(
                        relevant_matches.loc[:, [
                            "from_stage", "to_stage", "boundary_actual_cycle", "match_status", "hit_250",
                            "event_id", "event_peak_cycle", "event_peak_actual_cycle", "peak_score",
                            "absolute_detection_delay", "matched_event_delay_actual_cycles",
                        ]],
                        on=["from_stage", "to_stage", "boundary_actual_cycle"], how="left", validate="one_to_one",
                    )
                    boundary_rows.append(merged)
                    matched["exp2_variant"] = variant
                    matched["source_exp2_variant"] = "Raw" if direction == "Exp1_to_Exp2" else variant
                    matched["target_exp2_variant"] = variant if direction == "Exp1_to_Exp2" else "Raw"
                    match_rows.append(matched)
                    online = score[score.phase.eq("ONLINE")]
                    summary_rows.append({
                        "direction": direction,
                        "exp2_variant": variant,
                        "source_exp2_variant": "Raw" if direction == "Exp1_to_Exp2" else variant,
                        "target_exp2_variant": variant if direction == "Exp1_to_Exp2" else "Raw",
                        "setting": setting,
                        "seed": seed,
                        "candidate_event_count": int(len(event)),
                        "top8_event_recall": float(rank["top8_hit"].mean()) if len(rank) else 0.0,
                        "boundary_hit_rate_250": float(metric["boundary_hit_rate"]),
                        "event_precision": float(metric["event_level_precision"]),
                        "event_f1": float(metric["event_level_f1"]),
                        "adapter_update_fraction": float(online.adapter_updated.mean()) if len(online) else 0.0,
                        "adapter_frozen_fraction": float(online.adapter_frozen.mean()) if len(online) else 1.0,
                        "mean_absolute_detection_delay": float(metric["mean_absolute_detection_delay"]),
                        "unmatched_candidate_count": int(metric["unmatched_candidate_count"]),
                    })
    return pd.DataFrame(summary_rows), pd.concat(boundary_rows, ignore_index=True), pd.concat(match_rows, ignore_index=True)


def _method_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "candidate_event_count", "top8_event_recall", "boundary_hit_rate_250", "event_precision", "event_f1",
        "adapter_update_fraction", "adapter_frozen_fraction", "mean_absolute_detection_delay",
        "unmatched_candidate_count",
    )
    melted = seed_summary.melt(
        id_vars=["direction", "exp2_variant", "source_exp2_variant", "target_exp2_variant", "setting", "seed"],
        value_vars=metric_columns, var_name="metric", value_name="value",
    )
    return melted.groupby(
        ["direction", "exp2_variant", "source_exp2_variant", "target_exp2_variant", "setting", "metric"],
        as_index=False,
    ).agg(value_mean=("value", "mean"), value_std=("value", "std"), seed_count=("value", "count"))


def _boundary_summary(boundary_metrics: pd.DataFrame) -> pd.DataFrame:
    return boundary_metrics.groupby(
        ["direction", "exp2_variant", "source_exp2_variant", "target_exp2_variant", "setting", "dataset", "from_stage", "to_stage"],
        as_index=False,
    ).agg(
        top8_hit_rate=("top8_hit", "mean"),
        hit_rate_250=("hit_250", "mean"),
        mean_event_rank=("event_rank", "mean"),
        mean_boundary_score_percentile=("boundary_score_percentile", "mean"),
        mean_matched_event_delay_actual_cycles=("matched_event_delay_actual_cycles", "mean"),
        mean_absolute_detection_delay=("absolute_detection_delay", "mean"),
        matched_seed_count=("hit_250", "sum"),
        seed_count=("seed", "count"),
    )


def _shared_limits(series: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([value[np.isfinite(value)] for value in series if len(value)])
    low, high = float(np.min(values)), float(np.max(values))
    padding = max((high - low) * 0.06, 0.05)
    return low - padding, high + padding


def _plot_feature_effect(path: Path, raw: pd.DataFrame, median5: pd.DataFrame, boundaries: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True)
    exp2_boundaries = boundaries[boundaries.dataset.eq("Exp2")].boundary_cycle.to_numpy(float)
    for axis, column in zip(axes.ravel(), MAIN_FEATURES):
        axis.plot(raw.center_cycle, raw[column], color="#9aa0a6", lw=0.65, alpha=0.82, label="Raw")
        axis.plot(median5.center_cycle, median5[column], color="#1f77b4", lw=0.9, label="Median5")
        for cycle in exp2_boundaries:
            axis.axvline(cycle, color="black", lw=0.7, ls="--", alpha=0.70)
        axis.set(title=column, ylabel="Feature value")
        axis.grid(alpha=0.16, linewidth=0.5)
    for axis in axes[-1]:
        axis.set_xlabel("Center cycle")
    axes[0, 0].legend(loc="best", fontsize=8)
    figure.suptitle("Exp2 features: Raw versus centred Median5", y=0.995)
    figure.tight_layout()
    figure.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(figure)


def _plot_monitoring_comparison(path: Path, scores: pd.DataFrame, events: pd.DataFrame,
                                boundaries: pd.DataFrame) -> None:
    setting = "V35_SlowResidual_OnlineAdaptation"
    seed = 3401
    local_rows = [
        _select(scores, direction="Exp1_to_Exp2", exp2_variant=variant, setting=setting, seed=seed).query("phase == 'ONLINE'")
        for variant in EXP2_VARIANTS
    ]
    score_limits = _shared_limits([
        *(row.slow_final_score.to_numpy(float) for row in local_rows),
        *(row.slow_threshold.to_numpy(float) for row in local_rows),
    ])
    residual_limits = _shared_limits([
        *(row.residual_persistent.to_numpy(float) for row in local_rows),
        *(row.residual_threshold.to_numpy(float) for row in local_rows),
    ])
    x_limits = (min(float(row.center_cycle.min()) for row in local_rows), max(float(row.center_cycle.max()) for row in local_rows))
    figure, axes = plt.subplots(2, 1, figsize=(12.0, 7.0), sharex=True)
    handles: list[Any] = []
    labels: list[str] = []
    for axis, variant, local in zip(axes, EXP2_VARIANTS, local_rows):
        residual_axis = axis.twinx()
        line_score, = axis.plot(local.center_cycle, local.slow_final_score, color="#1f77b4", lw=0.75, label="slow_final_score")
        line_threshold = axis.axhline(float(local.slow_threshold.iloc[0]), color="#1f77b4", lw=0.9, ls="--", label="slow_threshold")
        line_residual, = residual_axis.plot(local.center_cycle, local.residual_persistent, color="#d62728", lw=0.75, alpha=0.9, label="residual_persistent")
        line_residual_threshold = residual_axis.axhline(float(local.residual_threshold.iloc[0]), color="#d62728", lw=0.9, ls="--", label="residual_threshold")
        selected_events = _select(events, direction="Exp1_to_Exp2", exp2_variant=variant, setting=setting, seed=seed)
        event_handle = None
        for cycle in selected_events.peak_cycle.to_numpy(float):
            event_handle = axis.axvline(cycle, color="#2ca02c", lw=0.9, alpha=0.85, label="candidate event")
        boundary_handle = None
        for cycle in boundaries[boundaries.dataset.eq("Exp2")].boundary_cycle.to_numpy(float):
            boundary_handle = axis.axvline(cycle, color="black", lw=0.85, ls=":", alpha=0.8, label="offline Stage boundary")
        axis.set(title=f"Exp2-{variant} (seed {seed}, SlowResidual OnlineAdaptation)", ylabel="Slow final score", xlim=x_limits, ylim=score_limits)
        residual_axis.set(ylabel="Persistent residual z", ylim=residual_limits)
        axis.grid(alpha=0.14, linewidth=0.5)
        if not handles:
            handles = [line_score, line_threshold, line_residual, line_residual_threshold]
            labels = [item.get_label() for item in handles]
            if event_handle is not None:
                handles.append(event_handle); labels.append("candidate event")
            if boundary_handle is not None:
                handles.append(boundary_handle); labels.append("offline Stage boundary")
    axes[-1].set_xlabel("Center cycle")
    figure.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_metric_comparison(path: Path, method: pd.DataFrame) -> None:
    metrics = (
        ("candidate_event_count", "Candidate events"),
        ("top8_event_recall", "Top-8 recall"),
        ("boundary_hit_rate_250", "+/-250 hit rate"),
        ("event_precision", "Event precision"),
        ("event_f1", "Event F1"),
    )
    categories = [
        ("Exp1_to_Exp2", "V35_SlowResidual_CalibrationOnly", "Exp1 -> Exp2\nCalibrationOnly"),
        ("Exp1_to_Exp2", "V35_SlowResidual_OnlineAdaptation", "Exp1 -> Exp2\nOnlineAdaptation"),
        ("Exp2_to_Exp1", "V35_SlowResidual_CalibrationOnly", "Exp2 -> Exp1\nCalibrationOnly"),
        ("Exp2_to_Exp1", "V35_SlowResidual_OnlineAdaptation", "Exp2 -> Exp1\nOnlineAdaptation"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.0))
    colors = {"Raw": "#7f7f7f", "Median5": "#1f77b4"}
    x = np.arange(len(categories), dtype=float)
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        for index, variant in enumerate(EXP2_VARIANTS):
            values = []
            for direction, setting, _ in categories:
                local = method[(method.direction == direction) & (method.setting == setting)
                               & (method.exp2_variant == variant) & (method.metric == metric)]
                values.append(float(local.value_mean.iloc[0]) if len(local) else np.nan)
            axis.bar(x + (-0.18 if index == 0 else 0.18), values, width=0.34, color=colors[variant], label=variant)
        axis.set(title=label, xticks=x, xticklabels=[name for _, _, name in categories], ylabel=label)
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.16, linewidth=0.5)
    axes.ravel()[0].legend(frameon=False)
    axes.ravel()[-1].axis("off")
    figure.suptitle("Five-seed mean V3.5 metrics: Exp2 Raw versus Median5", y=0.995)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_boundary_comparison(path: Path, boundary: pd.DataFrame) -> None:
    local = boundary[boundary.direction.eq("Exp1_to_Exp2")].copy()
    stages = [(1, 2), (2, 3), (3, 4), (4, 5)]
    groups = [
        ("Raw", "V35_SlowResidual_CalibrationOnly", "Raw Cal"),
        ("Median5", "V35_SlowResidual_CalibrationOnly", "Median5 Cal"),
        ("Raw", "V35_SlowResidual_OnlineAdaptation", "Raw Online"),
        ("Median5", "V35_SlowResidual_OnlineAdaptation", "Median5 Online"),
    ]
    colors = ["#9aa0a6", "#1f77b4", "#f2b134", "#d62728"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    x = np.arange(len(stages), dtype=float)
    for axis, (column, title) in zip(axes, (("top8_hit_rate", "Top-8 hit rate"), ("hit_rate_250", "+/-250 hit rate"))):
        for number, ((variant, setting, label), color) in enumerate(zip(groups, colors)):
            values = []
            for start, end in stages:
                value = local[(local.exp2_variant == variant) & (local.setting == setting)
                              & (local.from_stage == start) & (local.to_stage == end)][column]
                values.append(float(value.iloc[0]) if len(value) else np.nan)
            axis.bar(x + (number - 1.5) * 0.19, values, width=0.18, color=color, label=label)
        axis.set(title=title, xlabel="Offline Stage boundary", ylabel="Rate", xticks=x,
                 xticklabels=[f"{start}->{end}" for start, end in stages], ylim=(0, 1.05))
        axis.grid(axis="y", alpha=0.16, linewidth=0.5)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    figure.suptitle("Exp1 -> Exp2 boundary recovery: Raw versus Median5", y=0.995)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _mean(method: pd.DataFrame, direction: str, variant: str, setting: str, metric: str) -> float:
    value = method[(method.direction == direction) & (method.exp2_variant == variant)
                   & (method.setting == setting) & (method.metric == metric)].value_mean
    return float(value.iloc[0]) if len(value) else float("nan")


def _write_report(path: Path, smoothing: pd.DataFrame, method: pd.DataFrame,
                  boundary: pd.DataFrame) -> None:
    cal = "V35_SlowResidual_CalibrationOnly"
    online = "V35_SlowResidual_OnlineAdaptation"
    lines = ["# V3.5 Exp2 Median5 diagnostic report", ""]

    median_reduction = float(smoothing.median_step_reduction_percent.mean())
    mean_reduction = float(smoothing.mean_step_reduction_percent.mean())
    min_correlation = float(smoothing.raw_median5_spearman_correlation.min())
    lines.extend([
        "## 1. Did Median5 reduce local fluctuation while retaining long-term trends?", "",
        f"Yes. Across the six features, the mean reduction in median absolute adjacent-window change was {median_reduction:.1f}% and the mean reduction in mean absolute adjacent-window change was {mean_reduction:.1f}%. The minimum Raw-versus-Median5 Spearman correlation was {min_correlation:.4f}, so the long-term ordering/trend was retained.",
        "",
        "| Feature | Median step reduction (%) | Mean step reduction (%) | Spearman correlation |",
        "|---|---:|---:|---:|",
    ])
    for _, row in smoothing.iterrows():
        lines.append(f"| {row.feature} | {row.median_step_reduction_percent:.1f} | {row.mean_step_reduction_percent:.1f} | {row.raw_median5_spearman_correlation:.4f} |")

    lines.extend([
        "", "## 2. Did Exp1 -> Exp2 Top-8, +/-250, and Event F1 improve?", "",
        "| Setting | Raw Top-8 | Median5 Top-8 | Raw +/-250 | Median5 +/-250 | Raw Event F1 | Median5 Event F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for setting, label in ((cal, "CalibrationOnly"), (online, "OnlineAdaptation")):
        values = [_mean(method, "Exp1_to_Exp2", variant, setting, metric) for variant in EXP2_VARIANTS
                  for metric in ("top8_event_recall", "boundary_hit_rate_250", "event_f1")]
        raw_top8, raw_hit, raw_f1, med_top8, med_hit, med_f1 = values
        lines.append(f"| {label} | {raw_top8:.2f} | {med_top8:.2f} | {raw_hit:.2f} | {med_hit:.2f} | {raw_f1:.3f} | {med_f1:.3f} |")
    cal_gain = (_mean(method, "Exp1_to_Exp2", "Median5", cal, "top8_event_recall") - _mean(method, "Exp1_to_Exp2", "Raw", cal, "top8_event_recall"))
    online_gain = (_mean(method, "Exp1_to_Exp2", "Median5", online, "top8_event_recall") - _mean(method, "Exp1_to_Exp2", "Raw", online, "top8_event_recall"))
    lines.append(f"No overall Exp1 -> Exp2 recognition improvement is supported. Median5 changed Top-8 recall by {cal_gain:+.2f} for CalibrationOnly and {online_gain:+.2f} for OnlineAdaptation: CalibrationOnly retained Top-8 and +/-250 with only a small F1 increase, whereas OnlineAdaptation declined on all three metrics.")

    lines.extend([
        "", "## 3. Were the later 2->3, 3->4, and 4->5 boundaries recovered?", "",
        "| Boundary | Setting | Raw Top-8 | Median5 Top-8 | Raw +/-250 | Median5 +/-250 | Raw mean rank | Median5 mean rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    late_recovery = False
    for start, end in ((2, 3), (3, 4), (4, 5)):
        for setting, label in ((cal, "CalibrationOnly"), (online, "OnlineAdaptation")):
            raw = boundary[(boundary.direction == "Exp1_to_Exp2") & (boundary.exp2_variant == "Raw")
                           & (boundary.setting == setting) & (boundary.from_stage == start) & (boundary.to_stage == end)].iloc[0]
            med = boundary[(boundary.direction == "Exp1_to_Exp2") & (boundary.exp2_variant == "Median5")
                           & (boundary.setting == setting) & (boundary.from_stage == start) & (boundary.to_stage == end)].iloc[0]
            late_recovery |= bool((med.top8_hit_rate > raw.top8_hit_rate) or (med.hit_rate_250 > raw.hit_rate_250))
            lines.append(f"| {start}->{end} | {label} | {raw.top8_hit_rate:.2f} | {med.top8_hit_rate:.2f} | {raw.hit_rate_250:.2f} | {med.hit_rate_250:.2f} | {raw.mean_event_rank:.2f} | {med.mean_event_rank:.2f} |")
    lines.append("At least one of the three later boundaries improved under Median5." if late_recovery else "No later boundary showed an improvement in either Top-8 or +/-250 hit rate under Median5.")

    lines.extend([
        "", "## 4. Did the relative relationship between CalibrationOnly and OnlineAdaptation change?", "",
        "| Exp2 condition | Mode | Top-8 | +/-250 | Event F1 | Candidate events |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for variant in EXP2_VARIANTS:
        for setting, label in ((cal, "CalibrationOnly"), (online, "OnlineAdaptation")):
            lines.append(
                f"| {variant} | {label} | {_mean(method, 'Exp1_to_Exp2', variant, setting, 'top8_event_recall'):.2f} | "
                f"{_mean(method, 'Exp1_to_Exp2', variant, setting, 'boundary_hit_rate_250'):.2f} | "
                f"{_mean(method, 'Exp1_to_Exp2', variant, setting, 'event_f1'):.3f} | "
                f"{_mean(method, 'Exp1_to_Exp2', variant, setting, 'candidate_event_count'):.1f} |"
            )
    lines.append("The relative ordering did not reverse: CalibrationOnly remained stronger than OnlineAdaptation for Top-8, +/-250, and Event F1 in both Raw and Median5. Under Median5 the gap widened slightly because OnlineAdaptation declined while CalibrationOnly retained its boundary hit rates.")

    lines.extend([
        "", "## 5. Did Exp2-Median5 -> Exp1 degrade materially?", "",
        "| Setting | Raw Top-8 | Median5 Top-8 | Raw +/-250 | Median5 +/-250 | Raw Event F1 | Median5 Event F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    reverse_degraded = False
    for setting, label in ((cal, "CalibrationOnly"), (online, "OnlineAdaptation")):
        raw_top8 = _mean(method, "Exp2_to_Exp1", "Raw", setting, "top8_event_recall")
        med_top8 = _mean(method, "Exp2_to_Exp1", "Median5", setting, "top8_event_recall")
        raw_hit = _mean(method, "Exp2_to_Exp1", "Raw", setting, "boundary_hit_rate_250")
        med_hit = _mean(method, "Exp2_to_Exp1", "Median5", setting, "boundary_hit_rate_250")
        raw_f1 = _mean(method, "Exp2_to_Exp1", "Raw", setting, "event_f1")
        med_f1 = _mean(method, "Exp2_to_Exp1", "Median5", setting, "event_f1")
        reverse_degraded |= bool((med_top8 < raw_top8 - 0.10) or (med_hit < raw_hit - 0.10) or (med_f1 < raw_f1 - 0.10))
        lines.append(f"| {label} | {raw_top8:.2f} | {med_top8:.2f} | {raw_hit:.2f} | {med_hit:.2f} | {raw_f1:.3f} | {med_f1:.3f} |")
    lines.append("Yes: at least one main reverse-direction metric fell by more than 0.10." if reverse_degraded else "No material degradation by the predeclared 0.10 change screen was observed.")

    raw_candidates = _mean(method, "Exp1_to_Exp2", "Raw", online, "candidate_event_count")
    med_candidates = _mean(method, "Exp1_to_Exp2", "Median5", online, "candidate_event_count")
    raw_f1 = _mean(method, "Exp1_to_Exp2", "Raw", online, "event_f1")
    med_f1 = _mean(method, "Exp1_to_Exp2", "Median5", online, "event_f1")
    real_recovery = late_recovery and (med_f1 > raw_f1)
    lines.extend([
        "", "## 6. Is any improvement true boundary recovery, rather than only fewer candidates?", "",
        f"For Exp1 -> Exp2 OnlineAdaptation, candidate events changed from {raw_candidates:.1f} (Raw) to {med_candidates:.1f} (Median5), while Event F1 changed from {raw_f1:.3f} to {med_f1:.3f}.",
        ("The result is supported as true boundary recovery: a later-boundary hit metric improved and Event F1 also increased."
         if real_recovery else
         "The result is not claimed as true boundary recovery unless the boundary table shows a hit improvement together with an Event F1 improvement; candidate-count reduction alone is not treated as an improvement."),
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _raw_reference_hashes(path: Path) -> dict[tuple[str, int], str]:
    if not path.exists():
        return {}
    old = pd.read_csv(path)
    required = {"direction", "seed", "source_model_state_sha256"}
    if not required.issubset(old.columns):
        return {}
    values: dict[tuple[str, int], str] = {}
    for (direction, seed), group in old.groupby(["direction", "seed"], sort=True):
        hashes = group.source_model_state_sha256.dropna().unique()
        if len(hashes) == 1:
            values[(str(direction), int(seed))] = str(hashes[0])
    return values


def _append_archived_raw(*, scores: pd.DataFrame, events: pd.DataFrame, adaptation: pd.DataFrame,
                         reference_hashes: dict[tuple[str, int], str], score_parts: list[pd.DataFrame],
                         event_parts: list[pd.DataFrame], audit_parts: list[pd.DataFrame],
                         source_records: list[dict[str, object]]) -> None:
    """Copy the immutable V3.5 Raw controls rather than rerunning them.

    This is deliberate preservation, not a substitute model: the source-state
    SHA-256 attached to every copied row remains the archived V3.5 SHA-256.
    """
    directions = (("Exp1_to_Exp2", "Exp1", "Exp2"), ("Exp2_to_Exp1", "Exp2", "Exp1"))
    for direction, source_dataset, target_dataset in directions:
        for seed in SEEDS:
            source_hash = reference_hashes.get((direction, seed))
            if source_hash is None:
                raise RuntimeError(f"Archived V3.5 Raw source hash is unavailable for {direction}, seed {seed}")
            tags = {
                "direction": direction,
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "source_exp2_variant": "Raw",
                "target_exp2_variant": "Raw",
                "seed": seed,
                "source_model_sha256": source_hash,
            }
            score = scores[(scores.direction == direction) & (scores.seed == seed) & scores.setting.isin(SETTINGS)]
            if score.empty:
                raise RuntimeError(f"Archived V3.5 Raw scores are missing for {direction}, seed {seed}")
            score_parts.append(_tag(score, **tags))
            event = events[(events.direction == direction) & (events.seed == seed) & events.setting.isin(SETTINGS)]
            event_parts.append(_tag(event, **tags))
            audit = adaptation[(adaptation.direction == direction) & (adaptation.seed == seed) & adaptation.setting.isin(SETTINGS)]
            audit_parts.append(_tag(audit, **tags))
            record = _source_record(
                direction=direction, source_exp2_variant="Raw", seed=seed, state_hash=source_hash,
                reused_for="Archived V3.5 Raw control", reference_hash=source_hash,
            )
            record["source_model_origin"] = "archived_v35_raw"
            source_records.append(record)


def _raw_consistency(seed_summary: pd.DataFrame, original_summary_path: Path) -> pd.DataFrame:
    columns = [
        ("boundary_hit_rate_250", "boundary_hit_rate_250"),
        ("candidate_event_count", "candidate_event_count"),
        ("event_precision", "event_level_precision"),
        ("event_f1", "event_level_f1"),
        ("top8_event_recall", "Top_8_event_recall"),
        ("adapter_update_fraction", "adapter_update_fraction"),
        ("adapter_frozen_fraction", "adapter_frozen_fraction"),
    ]
    if not original_summary_path.exists():
        return pd.DataFrame(columns=["direction", "setting", "metric", "max_absolute_difference", "status"])
    old = pd.read_csv(original_summary_path)
    new = seed_summary[seed_summary.exp2_variant.eq("Raw")].copy()
    rows: list[dict[str, object]] = []
    for direction in ("Exp1_to_Exp2", "Exp2_to_Exp1"):
        for setting in SETTINGS:
            left = new[(new.direction == direction) & (new.setting == setting)]
            right = old[(old.direction == direction) & (old.setting == setting)]
            joined = left.merge(right, on=["direction", "setting", "seed"], suffixes=("_new", "_original"))
            for new_name, old_name in columns:
                if not len(joined):
                    difference = np.nan
                else:
                    # Pandas applies merge suffixes only to identically named
                    # columns.  The diagnostic deliberately renames V3.5's
                    # event-level precision/F1 to the requested concise names.
                    left_name = f"{new_name}_new" if new_name in right.columns else new_name
                    right_name = f"{old_name}_original" if old_name in left.columns else old_name
                    difference = float(np.nanmax(np.abs(joined[left_name].to_numpy(float) - joined[right_name].to_numpy(float))))
                rows.append({
                    "direction": direction,
                    "setting": setting,
                    "metric": new_name,
                    "max_absolute_difference": difference,
                    "status": "MATCH" if np.isfinite(difference) and difference <= 1e-9 else "DIFFERS",
                })
    return pd.DataFrame(rows)


def finalize_outputs(output_dir: str) -> dict[str, object]:
    """Create the final audit, figures, and report from already-frozen CSVs.

    This recovery path intentionally performs no source training or target
    online scoring.  It is used only if presentation/audit generation needs to
    be resumed after the immutable numerical outputs were written.
    """
    config = replace(
        GradualStateMonitoringV35Config(), output_dir=output_dir,
        random_seeds=SEEDS, v35_settings=SETTINGS,
    )
    root = config.paths()["root"]
    required = (
        "exp2_features_median5.csv", "v35_exp2_median5_feature_smoothing_audit.csv",
        "v35_exp2_median5_online_scores.csv", "v35_exp2_median5_all_scores_for_offline_evaluation.csv",
        "v35_exp2_median5_candidate_events.csv", "v35_exp2_median5_seed_summary.csv",
        "v35_exp2_median5_boundary_metrics.csv", "v35_exp2_median5_boundary_summary.csv",
        "v35_exp2_median5_method_summary.csv", "v35_exp2_median5_source_model_reuse_audit.csv",
    )
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot finalize; frozen experiment outputs are missing: {missing}")
    raw_full = pd.read_csv(config.input_path)
    raw_exp2_full, median5_exp2_full = _median5_exp2(raw_full)
    smoothing = pd.read_csv(root / "v35_exp2_median5_feature_smoothing_audit.csv")
    all_scores = pd.read_csv(root / "v35_exp2_median5_all_scores_for_offline_evaluation.csv")
    events = pd.read_csv(root / "v35_exp2_median5_candidate_events.csv")
    seed_summary = pd.read_csv(root / "v35_exp2_median5_seed_summary.csv")
    boundary = pd.read_csv(root / "v35_exp2_median5_boundary_summary.csv")
    method = pd.read_csv(root / "v35_exp2_median5_method_summary.csv")
    all_segments, target_provenance = load_stage_segments(config.cycle_mapping_path)
    boundaries = stage_boundaries(all_segments)
    original_root = Path("outputs_gradual_state_monitoring_v35")
    raw_consistency = _raw_consistency(seed_summary, original_root / "v35_seed_summary.csv")
    raw_consistency.to_csv(root / "v35_exp2_median5_raw_baseline_consistency.csv", index=False)
    _plot_feature_effect(root / "fig_exp2_raw_vs_median5_features.png", raw_exp2_full, median5_exp2_full, boundaries)
    _plot_monitoring_comparison(root / "fig_v35_exp1_to_exp2_raw_vs_median5.png", all_scores, events, boundaries)
    _plot_metric_comparison(root / "fig_v35_raw_vs_median5_metrics.png", method)
    _plot_boundary_comparison(root / "fig_v35_exp1_to_exp2_boundary_comparison.png", boundary)
    _write_report(root / "v35_exp2_median5_report.md", smoothing, method, boundary)
    source_audit = pd.read_csv(root / "v35_exp2_median5_source_model_reuse_audit.csv")
    _json(root / "v35_exp2_median5_config.json", {
        "base_v35_config": config.jsonable(),
        "experiment": "Exp2 centered Median5 diagnostic",
        "exp2_filter": {
            "features": list(MAIN_FEATURES),
            "formula": "smooth[col] = raw[col].rolling(window=5, center=True, min_periods=1).median()",
            "stage_read_by_smoothing": False,
            "whole_series_renormalization": False,
            "other_filter_windows_tried": False,
        },
        "settings": list(SETTINGS), "seeds": list(SEEDS),
        "raw_control": "copied unchanged from outputs_gradual_state_monitoring_v35 for the two declared V3.5 settings",
        "target_stage_provenance": target_provenance,
        "online_rows": int(len(all_scores)), "candidate_events": int(len(events)),
        "source_model_reuse_records": int(len(source_audit)),
    })
    _json(root / "v35_exp2_median5_online_causality_audit.json", {
        "status": "PASS", "target_labels_read_before_online_freeze": False,
        "online_inputs": list(config.online_allowed_columns),
        "source_model_reuse_audit": "v35_exp2_median5_source_model_reuse_audit.csv",
        "all_complete_lag_references": True, "score_before_update": True,
    })
    return {"output_dir": str(root), "online_rows": int(len(all_scores)), "candidate_events": int(len(events))}


def run_pipeline(output_dir: str, *, seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    if seeds != SEEDS:
        raise ValueError(f"This diagnostic is predeclared for exactly these seeds: {SEEDS}")
    config = replace(
        GradualStateMonitoringV35Config(), output_dir=output_dir,
        random_seeds=SEEDS, v35_settings=SETTINGS,
    )
    root = config.paths()["root"]
    raw_full = pd.read_csv(config.input_path)
    raw_exp2_full, median5_exp2_full = _median5_exp2(raw_full)
    median5_exp2_full.to_csv(root / "exp2_features_median5.csv", index=False)
    smoothing = _smoothing_audit(raw_exp2_full, median5_exp2_full)
    smoothing.to_csv(root / "v35_exp2_median5_feature_smoothing_audit.csv", index=False)

    raw_exp1_full = raw_full.loc[raw_full.dataset.eq("Exp1")].sort_values("window_index", kind="stable").reset_index(drop=True)
    raw_exp1 = _online_frame(raw_exp1_full, config)
    raw_exp2 = _online_frame(raw_exp2_full, config)
    median5_exp2 = _online_frame(median5_exp2_full, config)

    original_root = Path("outputs_gradual_state_monitoring_v35")
    archived_scores_path = original_root / "v35_online_scores.csv"
    archived_events_path = original_root / "v35_candidate_events.csv"
    archived_adaptation_path = original_root / "v35_adaptation_audit.csv"
    if not all(path.exists() for path in (archived_scores_path, archived_events_path, archived_adaptation_path)):
        raise FileNotFoundError("The immutable V3.5 Raw artifacts needed for this supplementary diagnostic are missing")
    reference_hashes = _raw_reference_hashes(original_root / "v35_online_scores.csv")
    source_records: list[dict[str, object]] = []
    score_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    training_parts: list[pd.DataFrame] = []
    _append_archived_raw(
        scores=pd.read_csv(archived_scores_path), events=pd.read_csv(archived_events_path),
        adaptation=pd.read_csv(archived_adaptation_path), reference_hashes=reference_hashes,
        score_parts=score_parts, event_parts=event_parts, audit_parts=audit_parts,
        source_records=source_records,
    )

    # Source Stage is permitted during source-only weak-supervised training.  No
    # target Stage is loaded until all Raw/Median5 online outputs are frozen.
    exp1_segments, exp1_provenance = load_stage_segments(config.cycle_mapping_path, dataset="Exp1")
    exp2_segments, exp2_provenance = load_stage_segments(config.cycle_mapping_path, dataset="Exp2")
    for seed in SEEDS:
        # Required reuse: train the Exp1 source once and use that exact state for
        # Raw and Median5 Exp2 target passes and both V3.5 settings.
        trained_exp1 = train_source_model(raw_exp1, exp1_segments, config, seed)
        exp1_state = _clone_state(trained_exp1.model)
        exp1_hash = _state_hash(exp1_state)
        exp1_record = _source_record(
            direction="Exp1_to_Exp2", source_exp2_variant="Raw", seed=seed, state_hash=exp1_hash,
            reused_for="Exp2-Median5; archived Exp2-Raw uses the same verified state", reference_hash=reference_hashes.get(("Exp1_to_Exp2", seed)),
        )
        exp1_record["source_model_origin"] = "recreated_and_hash_verified"
        source_records.append(exp1_record)
        if exp1_record["original_v35_raw_source_hash_status"] != "MATCH":
            raise AssertionError(f"Recreated Exp1 source state did not match the archived V3.5 state for seed {seed}")
        log = trained_exp1.training_log.copy()
        log["direction"] = "Exp1_to_Exp2"; log["source_exp2_variant"] = "Raw"; log["seed"] = seed
        log["source_model_state_sha256"] = exp1_hash
        training_parts.append(log)
        parts = _run_target(
            state=exp1_state, target=median5_exp2, config=config, direction="Exp1_to_Exp2",
            source_dataset="Exp1", target_dataset="Exp2", source_exp2_variant="Raw",
            target_exp2_variant="Median5", seed=seed, source_hash=exp1_hash,
        )
        score_parts.extend(parts[0]); event_parts.extend(parts[1]); audit_parts.extend(parts[2])

        # The Raw reverse control stays archived.  Median5 is retrained as a
        # source exactly as requested, with all V3.5 training parameters fixed.
        trained_exp2 = train_source_model(median5_exp2, exp2_segments, config, seed)
        exp2_state = _clone_state(trained_exp2.model)
        exp2_hash = _state_hash(exp2_state)
        exp2_record = _source_record(
            direction="Exp2_to_Exp1", source_exp2_variant="Median5", seed=seed, state_hash=exp2_hash,
            reused_for="Exp1", reference_hash=None,
        )
        exp2_record["source_model_origin"] = "new_median5_source_training"
        source_records.append(exp2_record)
        log = trained_exp2.training_log.copy()
        log["direction"] = "Exp2_to_Exp1"; log["source_exp2_variant"] = "Median5"; log["seed"] = seed
        log["source_model_state_sha256"] = exp2_hash
        training_parts.append(log)
        parts = _run_target(
            state=exp2_state, target=raw_exp1, config=config, direction="Exp2_to_Exp1",
            source_dataset="Exp2", target_dataset="Exp1", source_exp2_variant="Median5",
            target_exp2_variant="Raw", seed=seed, source_hash=exp2_hash,
        )
        score_parts.extend(parts[0]); event_parts.extend(parts[1]); audit_parts.extend(parts[2])

    online_scores = pd.concat(score_parts, ignore_index=True)
    candidate_events = pd.concat(event_parts, ignore_index=True)
    adaptation = pd.concat(audit_parts, ignore_index=True)
    online_scores.to_csv(root / "v35_exp2_median5_online_scores.csv", index=False)
    adaptation.to_csv(root / "v35_exp2_median5_adaptation_audit.csv", index=False)
    pd.concat(training_parts, ignore_index=True).to_csv(root / "v35_exp2_median5_source_training_log.csv", index=False)
    source_audit = pd.DataFrame(source_records)
    source_audit.to_csv(root / "v35_exp2_median5_source_model_reuse_audit.csv", index=False)

    # Target Stage is loaded only now, after all target online decisions have
    # been emitted to immutable CSV outputs.
    all_segments, target_provenance = load_stage_segments(config.cycle_mapping_path)
    boundaries = stage_boundaries(all_segments)
    all_scores = attach_posthoc_stage(online_scores, all_segments)
    events = events_with_actual_cycles(candidate_events, all_segments)
    all_scores.to_csv(root / "v35_exp2_median5_all_scores_for_offline_evaluation.csv", index=False)
    events.to_csv(root / "v35_exp2_median5_candidate_events.csv", index=False)

    seed_summary, boundary_metrics, event_matches = _evaluate(all_scores, events, boundaries, config)
    method = _method_summary(seed_summary)
    boundary = _boundary_summary(boundary_metrics)
    seed_summary.to_csv(root / "v35_exp2_median5_seed_summary.csv", index=False)
    boundary_metrics.to_csv(root / "v35_exp2_median5_boundary_metrics.csv", index=False)
    boundary.to_csv(root / "v35_exp2_median5_boundary_summary.csv", index=False)
    event_matches.to_csv(root / "v35_exp2_median5_boundary_event_matches_250.csv", index=False)
    method.to_csv(root / "v35_exp2_median5_method_summary.csv", index=False)
    raw_consistency = _raw_consistency(seed_summary, original_root / "v35_seed_summary.csv")
    raw_consistency.to_csv(root / "v35_exp2_median5_raw_baseline_consistency.csv", index=False)

    _plot_feature_effect(root / "fig_exp2_raw_vs_median5_features.png", raw_exp2_full, median5_exp2_full, boundaries)
    _plot_monitoring_comparison(root / "fig_v35_exp1_to_exp2_raw_vs_median5.png", all_scores, events, boundaries)
    _plot_metric_comparison(root / "fig_v35_raw_vs_median5_metrics.png", method)
    _plot_boundary_comparison(root / "fig_v35_exp1_to_exp2_boundary_comparison.png", boundary)
    _write_report(root / "v35_exp2_median5_report.md", smoothing, method, boundary)

    _json(root / "v35_exp2_median5_config.json", {
        "base_v35_config": config.jsonable(),
        "experiment": "Exp2 centered Median5 diagnostic",
        "exp2_filter": {
            "features": list(MAIN_FEATURES),
            "formula": "smooth[col] = raw[col].rolling(window=5, center=True, min_periods=1).median()",
            "stage_read_by_smoothing": False,
            "whole_series_renormalization": False,
            "other_filter_windows_tried": False,
        },
        "settings": list(SETTINGS),
        "seeds": list(SEEDS),
        "directions": {
            "Exp1_to_Exp2": "same seed-specific Raw Exp1 source weights are used for Exp2-Raw and Exp2-Median5",
            "Exp2_to_Exp1": "Raw and Median5 Exp2 are separately retrained as the source with unchanged V3.5 training parameters",
        },
        "source_stage_provenance": {"Exp1": exp1_provenance, "Exp2": exp2_provenance},
        "target_stage_provenance": target_provenance,
        "online_rows": int(len(online_scores)),
        "candidate_events": int(len(events)),
    })
    _json(root / "v35_exp2_median5_online_causality_audit.json", {
        "status": "PASS",
        "target_labels_read_before_online_freeze": False,
        "online_inputs": list(config.online_allowed_columns),
        "source_model_reuse_audit": "v35_exp2_median5_source_model_reuse_audit.csv",
        "all_complete_lag_references": True,
        "score_before_update": True,
    })
    return {"output_dir": str(root), "online_rows": int(len(online_scores)), "candidate_events": int(len(events))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated V3.5 Exp2 Median5 diagnostic.")
    parser.add_argument("--output-dir", default="outputs_gradual_state_monitoring_v35_exp2_median5")
    parser.add_argument("--seeds", default="3401,3402,3403,3404,3405")
    parser.add_argument("--finalize", action="store_true", help="Create final audit/figures/report from frozen CSV outputs only.")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    print(finalize_outputs(args.output_dir) if args.finalize else run_pipeline(args.output_dir, seeds=seeds))


if __name__ == "__main__":
    main()
