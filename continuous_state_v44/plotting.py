from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_METRICS = (
    ("D_state", "D state"),
    ("V500_norm", "V500 (normalised / 100 cycles)"),
    ("V1000_norm", "V1000 (normalised / 100 cycles)"),
    ("multi_scale_rate_divergence", "multi-scale rate divergence"),
    ("state_volatility", "state volatility"),
)


def consensus_trajectory_figure(consensus: pd.DataFrame, path: Path) -> None:
    """Plot configuration consensus on actual-cycle coordinates, without using actual time in state calculation."""
    fig, axes = plt.subplots(len(_METRICS), 1, figsize=(13, 14), sharex=True, constrained_layout=True)
    colors = {"Exp1": "#2676b8", "Exp2": "#dc6b2f"}
    for axis, (metric, title) in zip(axes, _METRICS):
        for dataset, group in consensus.groupby("dataset", sort=True):
            group = group.sort_values("center_cycle_actual")
            x = group.center_cycle_actual.to_numpy(float)
            q25 = group[f"{metric}_q25"].to_numpy(float)
            q50 = group[f"{metric}_q50"].to_numpy(float)
            q75 = group[f"{metric}_q75"].to_numpy(float)
            axis.fill_between(x, q25, q75, color=colors.get(dataset, "grey"), alpha=.18)
            axis.plot(x, q50, color=colors.get(dataset, "grey"), lw=1.5, label=f"{dataset} median")
        axis.set_ylabel(title)
        axis.grid(alpha=.22)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Actual machine cycle (plotting / post-hoc coordinate only)")
    fig.suptitle("v4.4 continuous-state configuration consensus", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def morphology_interval_figure(alignment: pd.DataFrame, path: Path) -> None:
    """Show post-hoc interval summaries. It is intentionally not a fitted morphology model."""
    table = alignment.loc[alignment.row_type.eq("morphology_anchor_interval")].copy()
    labels = [f"{int(row.start_cycle_actual/1000)}–{int(row.end_cycle_actual/1000)}k" for _, row in table.iterrows()]
    x = np.arange(len(table))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].bar(x, table.D_cumulative_absolute_change, width=.48, color="#2676b8", label="D cumulative |change|")
    axes[0].set_ylabel("D cumulative |change|")
    velocity_axis = axes[0].twinx()
    velocity_axis.plot(x, table.V500_mean, marker="o", color="#ef9b36", lw=1.7, label="V500 mean")
    velocity_axis.set_ylabel("V500 mean")
    first_handles, first_labels = axes[0].get_legend_handles_labels()
    second_handles, second_labels = velocity_axis.get_legend_handles_labels()
    axes[0].legend(first_handles + second_handles, first_labels + second_labels, fontsize=8, loc="upper right")
    axes[0].grid(axis="y", alpha=.22)
    axes[1].plot(x, table.rate_divergence_mean, marker="o", color="#9835a4", label="rate divergence mean")
    axes[1].plot(x, table.volatility_mean, marker="o", color="#188977", label="volatility mean")
    axes[1].set_ylabel("continuous state")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.22)
    for column, color in (("abs_delta_Sa", "#005f73"), ("abs_delta_Sq", "#ca6702"), ("abs_delta_Sz", "#9b2226"), ("abs_delta_Sku", "#5a189a")):
        axes[2].plot(x, table[column], marker="o", lw=1.3, label=column.replace("abs_delta_", "|Δ" ) + "|")
    axes[2].set_ylabel("post-hoc morphology |Δ|")
    axes[2].legend(ncol=2, fontsize=8); axes[2].grid(alpha=.22)
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Actual-cycle morphology interval")
    fig.suptitle("Exp1 post-hoc morphology interval alignment (not used to select state parameters)", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)
