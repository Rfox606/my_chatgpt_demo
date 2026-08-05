from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(path) -> None:
    plt.tight_layout(); plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()


def make_figures(consensus: pd.DataFrame, support: pd.DataFrame, episodes: pd.DataFrame, ry_audit: pd.DataFrame, metrics: pd.DataFrame, output) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for dataset, group in consensus.groupby("dataset"):
        group = group.sort_values("center_cycle")
        axes[0].plot(group.center_cycle, group.D_state_q50, label=dataset)
        axes[0].fill_between(group.center_cycle, group.D_state_q25, group.D_state_q75, alpha=.2)
        axes[1].plot(group.center_cycle, group.multi_scale_rate_divergence_q50, label=dataset)
        axes[1].fill_between(group.center_cycle, group.multi_scale_rate_divergence_q25, group.multi_scale_rate_divergence_q75, alpha=.2)
        axes[2].plot(group.center_cycle, group.combined_change_score_q50, label=dataset)
        if not episodes.empty:
            for _, episode in episodes.loc[episodes.target_dataset.eq(dataset)].iterrows(): axes[2].axvspan(episode.start_cycle, episode.end_cycle, alpha=.12, color="tab:red")
    axes[0].set_ylabel("D consensus"); axes[1].set_ylabel("Rate divergence"); axes[2].set_ylabel("Change score")
    axes[2].set_xlabel("Cycle")
    for axis in axes: axis.legend(fontsize=8)
    _save(output / "fig_v42_consensus_trajectories.png")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for dataset, group in support.groupby("dataset"):
        group = group.sort_values("center_cycle")
        axes[0].plot(group.center_cycle, group.directed_configuration_support, label=dataset)
        axes[1].plot(group.center_cycle, group.rate_divergence_configuration_support, label=dataset)
        axes[2].plot(group.center_cycle, group.abrupt_configuration_support, label=dataset)
    for axis, label in zip(axes, ("Directed support", "Rate-divergence support", "Abrupt support")):
        axis.set_ylabel(label); axis.set_ylim(-.02, 1.02); axis.legend(fontsize=8)
    axes[-1].set_xlabel("Cycle")
    _save(output / "fig_v42_configuration_support.png")

    plt.figure(figsize=(10, 4))
    if not episodes.empty:
        for index, (_, episode) in enumerate(episodes.iterrows()):
            plt.hlines(index, episode.start_cycle, episode.end_cycle, linewidth=5, color="tab:blue")
            plt.plot(episode.peak_cycle, index, "o", color="tab:red")
        plt.yticks(range(len(episodes)), [f"{row.target_dataset}:{row.dominant_evidence}" for _, row in episodes.iterrows()], fontsize=7)
    plt.xlabel("Cycle"); plt.title("Consensus change episodes")
    _save(output / "fig_v42_change_episodes.png")

    plt.figure(figsize=(8, 4))
    if not ry_audit.empty:
        plt.bar(ry_audit.dataset, ry_audit.ry_subspace_dominance_p95, label="ry dominance p95")
        plt.axhline(.60, color="tab:red", linestyle="--", label="single-subspace flag threshold")
        plt.legend()
    plt.ylabel("D_ry / group-distance sum"); plt.title("ry_p2p subspace audit")
    _save(output / "fig_v42_ry_audit.png")

    plt.figure(figsize=(11, 5))
    if not metrics.empty:
        subset = metrics.loc[metrics.horizon_cycles.isin([500, 1000])]
        for model, group in subset.groupby("model"): plt.plot(range(len(group)), group.MAE, marker="o", markersize=2, linewidth=.7, label=model)
        plt.legend(fontsize=7, ncol=3)
    plt.ylabel("MAE"); plt.xlabel("output / horizon / protocol rows"); plt.title("Forecast metrics")
    _save(output / "fig_v42_forecast_comparison.png")
