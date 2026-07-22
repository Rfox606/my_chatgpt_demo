from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figures(
    scores: pd.DataFrame,
    coefficients: pd.DataFrame,
    source_gaps: pd.DataFrame,
    candidates: pd.DataFrame,
    figure_dir: Path,
) -> None:
    target = scores.loc[scores["dataset_role"].eq("target")].copy()
    for direction in ("Exp1_to_Exp2", "Exp2_to_Exp1"):
        subset = target.loc[target["direction_id"].eq(direction)]
        fig, left = plt.subplots(figsize=(9, 4))
        right = left.twinx()
        left.plot(subset["center_cycle"], subset["AWR_rel"], color="#1769aa", label="AWR_rel")
        right.plot(subset["center_cycle"], subset["BD"], color="#d95f02", alpha=0.75, label="BD")
        left.set(xlabel="Cycle", ylabel="AWR_rel", title=f"{direction}: target AWR and BD")
        right.set_ylabel("BD")
        _save(fig, figure_dir / f"fig_csv1_target_awr_bd_{direction}.png")

    fig, axis = plt.subplots(figsize=(7, 5))
    for direction, subset in target.groupby("direction_id", sort=True):
        axis.scatter(subset["AWR_rel"], subset["BD"], s=8, alpha=0.45, label=direction)
    axis.set(xlabel="AWR_rel", ylabel="BD", title="Target continuous-state map")
    axis.legend()
    _save(fig, figure_dir / "fig_csv1_awr_bd_scatter.png")

    fig, axis = plt.subplots(figsize=(9, 5))
    labels = coefficients["direction_id"] + "\n" + coefficients["feature_name"]
    axis.barh(labels, coefficients["normalized_weight"], color=np.where(coefficients["normalized_weight"] >= 0, "#2c7fb8", "#d95f02"))
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(title="Normalised rank weights", xlabel="weight")
    _save(fig, figure_dir / "fig_csv1_rank_coefficients.png")

    fig, axis = plt.subplots(figsize=(7, 4))
    for direction, subset in source_gaps.groupby("direction_id", sort=True):
        axis.plot(subset["gap_bin"], subset["pair_auc"], marker="o", label=direction)
    axis.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axis.set(ylabel="Source validation pair AUC", title="Pair AUC by time gap")
    axis.legend()
    _save(fig, figure_dir / "fig_csv1_pair_auc_by_gap.png")

    fig, axis = plt.subplots(figsize=(9, 4))
    for direction, subset in target.groupby("direction_id", sort=True):
        axis.plot(subset["center_cycle"], subset["oos_fraction"], label=direction)
    axis.set(xlabel="Cycle", ylabel="Out-of-support fraction", title="Target support over time")
    axis.legend()
    _save(fig, figure_dir / "fig_csv1_oos_over_time.png")

    fig, axis = plt.subplots(figsize=(9, 4))
    if not candidates.empty:
        for kind, subset in candidates.groupby("candidate_type", sort=True):
            axis.scatter(subset["center_cycle"], subset["AWR_rel"], s=28, label=kind)
    axis.set(xlabel="Cycle", ylabel="AWR_rel", title="Offline physical-validation candidates")
    if not candidates.empty:
        axis.legend(fontsize=7, ncol=2)
    _save(fig, figure_dir / "fig_csv1_candidate_windows.png")
