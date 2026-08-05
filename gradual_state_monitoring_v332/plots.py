from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATE_CODES = {"STABLE": 0, "TRANSITION": 1, "NEW_STABLE": 2, "TEMPORARY_DISTURBANCE": 3}
STATE_LABELS = ["STABLE", "TRANSITION", "NEW_STABLE", "TEMPORARY"]
STAGE_COLORS = ["#dceefb", "#e8f4e5", "#fff1cf", "#f7e1e5", "#e9ddf5"]


def _stage_spans(axis: plt.Axes, boundaries: pd.DataFrame, dataset: str, maximum: float) -> None:
    local = boundaries[boundaries.dataset == dataset].sort_values("from_stage", kind="stable")
    starts = [0.0, *local.boundary_actual_cycle.to_numpy(float).tolist()]
    ends = [*local.boundary_actual_cycle.to_numpy(float).tolist(), maximum]
    for stage, (start, end) in enumerate(zip(starts, ends), start=1):
        axis.axvspan(start, end, color=STAGE_COLORS[(stage - 1) % len(STAGE_COLORS)], alpha=0.42, linewidth=0)
    for boundary in local.boundary_actual_cycle.to_numpy(float):
        axis.axvline(boundary, color="#555555", linestyle="--", linewidth=0.8, zorder=2)


def write_alignment_figure(posthoc: pd.DataFrame, evidence: pd.DataFrame, boundaries: pd.DataFrame,
                           match_log: pd.DataFrame, dataset: str, path: Path) -> None:
    state = posthoc[posthoc.dataset == dataset].sort_values("input_index", kind="stable")
    ev = evidence[evidence.dataset == dataset].sort_values("input_index", kind="stable")
    merged = state.merge(ev.loc[:, ["input_index", "change_evidence"]], on="input_index", how="left")
    log = match_log[(match_log.dataset == dataset) & (match_log.tolerance_actual_cycles == 250.0)]
    figure, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    maximum = float(np.nanmax(merged.actual_cycle.to_numpy(float)))
    for axis in axes:
        _stage_spans(axis, boundaries, dataset, maximum)
    axes[0].step(merged.actual_cycle, [STATE_CODES.get(value, -1) for value in merged.online_state], where="post", color="#1f4e79", linewidth=1.0)
    axes[0].set_yticks(range(len(STATE_LABELS)), STATE_LABELS)
    axes[0].set_ylabel("online_state")
    axes[1].step(merged.actual_cycle, merged.local_state_id, where="post", color="#6a3d9a", linewidth=1.0)
    axes[1].set_ylabel("local_state_id")
    axes[2].plot(merged.actual_cycle, merged.change_evidence, color="#cc4c02", linewidth=0.8)
    axes[2].set_ylabel("change_evidence")
    axes[2].set_xlabel("actual cycle (post-hoc mapped)")
    for _, row in log.iterrows():
        if row.match_status == "MATCHED":
            axes[0].scatter(row.alarm_actual_cycle, 1, marker="o", s=28, color="#238b45", zorder=5, label="matched alarm")
        elif row.match_status == "FALSE_ALARM":
            axes[0].scatter(row.alarm_actual_cycle, 1, marker="x", s=36, color="#d7301f", zorder=5, label="false alarm")
        elif row.match_status == "MISSED":
            axes[0].scatter(row.boundary_actual_cycle, 1, marker="v", s=32, color="#7f0000", zorder=5, label="missed boundary")
        elif row.match_status == "DUPLICATE":
            axes[0].scatter(row.alarm_actual_cycle, 1, marker="+", s=42, color="#8856a7", zorder=5, label="duplicate alarm")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        axes[0].legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    figure.suptitle(f"V3.3.2 Stage alignment — {dataset} (Stage is post-hoc only)", fontsize=13)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_boundary_summary(metrics: pd.DataFrame, path: Path) -> None:
    selected = metrics[(metrics.model == "V331") & (metrics.setting == "v331_frozen")].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for dataset, group in selected.groupby("dataset", sort=True):
        axis = axes[0]
        axis.plot(group.tolerance_actual_cycles, group.boundary_precision, marker="o", label=f"{dataset} precision")
        axis.plot(group.tolerance_actual_cycles, group.boundary_recall, marker="s", linestyle="--", label=f"{dataset} recall")
    axes[0].set_ylim(-0.03, 1.03); axes[0].set_xlabel("tolerance (actual cycles)"); axes[0].set_ylabel("score")
    axes[0].set_title("V3.3.1 boundary detection")
    axes[0].legend(fontsize=8)
    primary = selected[selected.tolerance_actual_cycles == 250.0]
    if len(primary):
        primary.set_index("dataset")[["matched_boundary_count", "missed_boundary_count", "false_alarm_count", "duplicate_alarm_count"]].plot.bar(ax=axes[1], color=["#41ab5d", "#ef3b2c", "#756bb1", "#fd8d3c"])
    axes[1].set_title("±250-cycle event accounting")
    axes[1].set_ylabel("count"); axes[1].tick_params(axis="x", rotation=0)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_stage_stability_summary(metrics: pd.DataFrame, path: Path) -> None:
    selected = metrics[(metrics.model == "V331") & (metrics.setting == "v331_frozen")].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    labels = [f"{row.dataset}-S{row.stage}" for _, row in selected.iterrows()]
    axes[0].bar(labels, selected.stage_stable_rate, color="#3182bd")
    axes[0].axhline(.70, color="#d7301f", linestyle="--", linewidth=1, label="Gate C threshold")
    axes[0].set_ylim(0, 1.03); axes[0].set_ylabel("stage_stable_rate"); axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, selected.false_switches_per_1000_windows, color="#e6550d")
    axes[1].axhline(5.0, color="#d7301f", linestyle="--", linewidth=1, label="Gate C threshold")
    axes[1].set_ylabel("false switches / 1000 windows"); axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(fontsize=8)
    figure.suptitle("V3.3.1 stage-interior stability (±250 actual-cycle buffer)")
    figure.savefig(path, dpi=150)
    plt.close(figure)
