from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def state_space_figure(consensus: pd.DataFrame, dataset: str, path: Path) -> None:
    """D--V1000 state-space view; all encodings are continuous and use effective-cycle calculations."""
    group = consensus.loc[consensus.dataset.eq(dataset)].sort_values("center_cycle_effective")
    divergence = group.multi_scale_rate_divergence_q50.to_numpy(float)
    scale = max(float(np.quantile(divergence, .99)), 1e-9)
    size = 12.0 + 150.0 * np.clip(divergence / scale, 0.0, 1.0)
    fig, axis = plt.subplots(figsize=(10.5, 7.5), constrained_layout=True)
    dots = axis.scatter(group.D_state_q50, group.V1000_norm_q50, c=group.center_cycle_effective, s=size, cmap="viridis", alpha=.72, linewidths=.1)
    colorbar = fig.colorbar(dots, ax=axis); colorbar.set_label("effective cycle")
    axis.set_xlabel("D_state: early-baseline deviation")
    axis.set_ylabel("V1000: long-horizon state-vector speed")
    axis.set_title(f"{dataset} v4.5 continuous state space\npoint size: multi-scale rate divergence")
    axis.grid(alpha=.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def v44_v45_figure(display: pd.DataFrame, morphology_cycles: list[float], path: Path) -> None:
    metrics = ("D_state", "V1000_norm", "multi_scale_rate_divergence", "state_volatility")
    labels = {"D_state": "D_state", "V1000_norm": "V1000", "multi_scale_rate_divergence": "rate divergence", "state_volatility": "volatility"}
    fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    colors = {"v44_display_standard": "#377eb8", "v45_display_standard": "#e6550d"}
    for axis, metric in zip(axes, metrics):
        for dataset, group in display.loc[display.metric.eq(metric)].groupby("dataset", sort=True):
            group = group.sort_values("center_cycle_actual")
            for column, color in colors.items():
                axis.plot(group.center_cycle_actual, group[column], color=color, lw=1.05, alpha=.82, label=f"{dataset} {column[0:3]}")
            if dataset == "Exp1":
                for marker in morphology_cycles[1:]:
                    axis.axvline(marker, color="0.55", lw=.55, ls=":", zorder=0)
        axis.set_ylabel(f"{labels[metric]}\nwithin-run display scale")
        axis.grid(alpha=.22)
    handles, labels_legend = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles)); axes[0].legend(by_label.values(), by_label.keys(), ncol=2, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("actual machine cycle (plot coordinate only; Exp1 dotted lines are morphology anchors)")
    fig.suptitle("v4.4 pre-normalised vs v4.5 raw-feature consensus trajectories", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)
