from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(path) -> None:
    plt.tight_layout(); plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()


def make_figures(consensus: pd.DataFrame, episodes: pd.DataFrame, deconfounding: pd.DataFrame, time_table: pd.DataFrame, ry_physical: pd.DataFrame, metrics: pd.DataFrame, output) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    for dataset, group in consensus.groupby("dataset"):
        group = group.sort_values("center_cycle_actual")
        axes[0].plot(group.center_cycle_actual, group.D_state_q50, label=dataset)
        axes[0].fill_between(group.center_cycle_actual, group.D_state_q25, group.D_state_q75, alpha=.2)
        axes[1].plot(group.center_cycle_actual, group.multi_scale_rate_divergence_q50, label=dataset)
        axes[2].plot(group.center_cycle_actual, group.change_configuration_support, label=dataset)
    for _, episode in episodes.iterrows(): axes[2].axvspan(episode.start_cycle_actual, episode.end_cycle_actual, color="tab:red", alpha=.12)
    axes[0].set_ylabel("D consensus"); axes[1].set_ylabel("Rate divergence"); axes[2].set_ylabel("Configuration support"); axes[2].set_xlabel("Actual cycle")
    for axis in axes: axis.legend(fontsize=8)
    _save(output / "fig_v43_actual_cycle_consensus.png")

    plt.figure(figsize=(11, 5))
    if not episodes.empty:
        for index, (_, episode) in enumerate(episodes.iterrows()):
            plt.hlines(index, episode.start_cycle_actual, episode.end_cycle_actual, linewidth=5, color="tab:blue")
            plt.plot(episode.peak_cycle_actual, index, "o", color="tab:red")
        plt.yticks(range(len(episodes)), [f"{row.target_dataset}:{row.dominant_evidence}" for _, row in episodes.iterrows()], fontsize=7)
    plt.xlabel("Actual cycle"); plt.title("Change episodes: effective-state / actual-cycle localization")
    _save(output / "fig_v43_change_episodes_actual.png")

    plt.figure(figsize=(11, 5))
    matches = deconfounding.loc[deconfounding.row_type.eq("original_episode_match")]
    for width, group in matches.groupby("stop_exclusion_half_width_actual"):
        plt.scatter(group.original_peak_nearest_stop_distance_actual, group.interval_iou_actual, label=f"exclude +/-{width}", alpha=.8)
    plt.xlabel("Original peak distance to nearest actual stop"); plt.ylabel("Deconfounded interval IoU"); plt.ylim(-.02, 1.02); plt.legend(); plt.title("Stop deconfounding sensitivity")
    _save(output / "fig_v43_stop_deconfounding.png")

    rows = time_table.loc[time_table.row_type.eq("window")]
    plt.figure(figsize=(7, 6))
    if not rows.empty:
        plt.scatter(rows.V500_norm_q50_effective_time, rows.V500_norm_q50_actual_time, s=3, alpha=.25)
        low = min(rows.V500_norm_q50_effective_time.min(), rows.V500_norm_q50_actual_time.min()); high = max(rows.V500_norm_q50_effective_time.max(), rows.V500_norm_q50_actual_time.max())
        plt.plot([low, high], [low, high], "k--", linewidth=1)
    plt.xlabel("V500 effective-time"); plt.ylabel("V500 actual-time"); plt.title("Actual-time sensitivity")
    _save(output / "fig_v43_effective_vs_actual_time.png")

    anchors = ry_physical.loc[ry_physical.row_type.eq("anchor")] if "row_type" in ry_physical else pd.DataFrame()
    plt.figure(figsize=(8, 4))
    if not anchors.empty:
        plt.plot(anchors.cycle_actual, anchors.ry_subspace_dominance, marker="o", label="ry subspace dominance")
        plt.plot(anchors.cycle_actual, anchors.ry_p2p, marker="x", label="ry_p2p (z input)")
        plt.legend()
    plt.xlabel("Actual cycle"); plt.title("ry physical post-hoc audit")
    _save(output / "fig_v43_ry_physical_audit.png")

    plt.figure(figsize=(11, 5))
    if not metrics.empty:
        subset = metrics.loc[metrics.horizon_cycles.isin([500, 1000])]
        for model, group in subset.groupby("model"): plt.plot(range(len(group)), group.MAE, marker="o", markersize=2, linewidth=.7, label=model)
        plt.legend(fontsize=7, ncol=3)
    plt.ylabel("MAE"); plt.xlabel("output / horizon / protocol rows"); plt.title("Forecast metrics (unchanged v4.2 module)")
    _save(output / "fig_v43_forecast_comparison.png")
