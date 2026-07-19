from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .state_engine import EVIDENCE_NAMES


def _save(path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def make_figures(states: pd.DataFrame, events: pd.DataFrame, ablations: pd.DataFrame, metrics: pd.DataFrame, output) -> None:
    output.mkdir(parents=True, exist_ok=True)
    targets = {str(dataset): frame.sort_values("center_cycle") for dataset, frame in states.groupby("dataset")}
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    for dataset, frame in targets.items():
        axes[0].plot(frame.center_cycle, frame.D_state, linewidth=.7, label=dataset)
        axes[1].plot(frame.center_cycle, frame.V100_norm, linewidth=.7, label=f"{dataset}: V100")
        axes[1].plot(frame.center_cycle, frame.V1000_norm, linewidth=.7, linestyle="--", label=f"{dataset}: V1000")
        evidence_total = frame.loc[:, list(EVIDENCE_NAMES)].sum(axis=1)
        axes[2].step(frame.center_cycle, evidence_total, linewidth=.7, where="mid", label=dataset)
    axes[0].set_ylabel("D")
    axes[1].set_ylabel("Velocity norm")
    axes[2].set_ylabel("Active evidence count")
    axes[2].set_xlabel("Cycle")
    for axis in axes:
        axis.legend(fontsize=8, ncol=2)
    _save(output / "fig_v41_state_trajectories.png")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for dataset, frame in targets.items():
        for group in ("rs", "rx", "ry"):
            axes[0].plot(frame.center_cycle, frame[f"velocity_v100_{group}_contribution"], linewidth=.6, label=f"{dataset}:{group}")
            axes[1].plot(frame.center_cycle, frame[f"velocity_v500_{group}_contribution"], linewidth=.6, label=f"{dataset}:{group}")
            axes[2].plot(frame.center_cycle, frame[f"velocity_v1000_{group}_contribution"], linewidth=.6, label=f"{dataset}:{group}")
    for axis, label in zip(axes, ("V100 subspace contribution", "V500 subspace contribution", "V1000 subspace contribution")):
        axis.set_ylabel(label)
        axis.set_ylim(-.02, 1.02)
        axis.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Cycle")
    _save(output / "fig_v41_velocity_subspace_contributions.png")

    fig, axes = plt.subplots(len(EVIDENCE_NAMES), 1, figsize=(12, 9), sharex=True)
    for axis, name in zip(axes, EVIDENCE_NAMES):
        for dataset, frame in targets.items():
            axis.step(frame.center_cycle, frame[name], where="mid", linewidth=.7, label=dataset)
        if not events.empty:
            subset = events.loc[(events.evidence_type.eq(name)) & events.event.eq("algorithm_evidence_onset")]
            for _, row in subset.iterrows():
                axis.axvline(row.cycle, color="black", alpha=.2, linewidth=.5)
        axis.set_ylabel(name.replace("_evidence", ""), fontsize=8)
        axis.legend(fontsize=7)
    axes[-1].set_xlabel("Cycle")
    _save(output / "fig_v41_evidence_tracks.png")

    plt.figure(figsize=(10, 4))
    if not ablations.empty:
        values = ablations.loc[ablations.trajectory_spearman.notna()]
        labels = values.apply(lambda row: f"b{int(row.baseline_cycles)}-{row.distance_form}-{row.removed_feature_group}", axis=1)
        plt.scatter(np.arange(len(values)), values.trajectory_spearman, s=12, c=values.event_collection_stable.map({True: "tab:blue", False: "tab:red"}))
        plt.xticks(np.arange(len(values)), labels, rotation=75, fontsize=6)
        plt.axhline(.8, color="black", linestyle="--", linewidth=.7)
    plt.ylabel("D trajectory Spearman correlation")
    plt.title("Ablation trajectory and event-position stability")
    _save(output / "fig_v41_ablation_stability.png")

    plt.figure(figsize=(11, 5))
    if not metrics.empty:
        subset = metrics.loc[metrics.horizon_cycles.isin([500, 1000])]
        for model, group in subset.groupby("model"):
            plt.plot(np.arange(len(group)), group.MAE, marker="o", markersize=2, linewidth=.7, label=model)
        plt.legend(fontsize=7, ncol=3)
    plt.ylabel("MAE")
    plt.xlabel("output / horizon / protocol rows")
    plt.title("Forecast metrics at cycle horizons")
    _save(output / "fig_v41_forecast_comparison.png")
