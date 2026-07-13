from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(path: Path) -> None:
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def make_figures(states: pd.DataFrame, forecasts: pd.DataFrame, metrics: pd.DataFrame, ablation: pd.DataFrame, physical: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    targets = {row.target_dataset: frame.sort_values("center_cycle") for _, frame in states.groupby("target_dataset") for row in [frame.iloc[0]]}
    for dataset, name in (("Exp1", "fig_csv3_exp1_state_trajectory.png"), ("Exp2", "fig_csv3_exp2_state_trajectory.png")):
        frame = targets.get(dataset, pd.DataFrame())
        plt.figure(figsize=(10, 4))
        if not frame.empty:
            for column in ("D_state", "V50_norm", "A_smooth_20"):
                plt.plot(frame.center_cycle, frame[column], label=column, linewidth=.9)
            plt.legend(); plt.xlabel("cycle")
        plt.title(f"{dataset} target-relative state")
        _save(output / name)
    plt.figure(figsize=(10, 4))
    for dataset, frame in targets.items():
        plt.plot(frame.center_cycle, frame.plateau_locked, label=f"{dataset} plateau locked")
    plt.legend(); plt.title("Causal plateau detection"); _save(output / "fig_csv3_plateau_detection.png")
    plt.figure(figsize=(10, 4))
    for dataset, frame in targets.items():
        plt.plot(frame.center_cycle, frame.instability_score, label=f"{dataset} instability")
    plt.legend(); plt.title("Causal plateau exit"); _save(output / "fig_csv3_plateau_exit_detection.png")
    plt.figure(figsize=(10, 4))
    for dataset, frame in targets.items():
        plt.plot(frame.center_cycle, frame.S_smooth_50, label=f"{dataset} severe candidate")
    plt.legend(); plt.title("Severe candidate score"); _save(output / "fig_csv3_severe_candidate_score.png")
    plt.figure(figsize=(6, 5))
    for dataset, frame in targets.items(): plt.scatter(frame.D_state, frame.V50_norm, s=2, label=dataset, alpha=.5)
    plt.xlabel("D_state"); plt.ylabel("V50_norm"); plt.legend(); _save(output / "fig_csv3_D_V_state_plane.png")
    plt.figure(figsize=(10, 4))
    if not metrics.empty:
        for model, group in metrics.loc[metrics.horizon_cycles.isin([500, 1000])].groupby("model"):
            plt.plot(range(len(group)), group.MAE, marker="o", label=model)
        plt.legend()
    plt.title("Frozen, robust online and safe forecast MAE"); _save(output / "fig_csv3_forecast_frozen_online_safe.png")
    plt.figure(figsize=(10, 4))
    if not forecasts.empty:
        for label, group in forecasts.groupby(["protocol_id", "output_name", "horizon_cycles"]):
            plt.plot(group.prediction_origin_cycle, group.ensemble_alpha, linewidth=.6, label=" / ".join(map(str, label)))
        plt.legend(fontsize=6, ncol=2)
    plt.title("Safe ensemble alpha"); _save(output / "fig_csv3_ensemble_alpha.png")
    plt.figure(figsize=(9, 4))
    if not ablation.empty:
        counts = ablation.groupby("ablation").size(); plt.bar(counts.index, counts.values)
    plt.title("Ablation modules executed"); _save(output / "fig_csv3_ablation_comparison.png")
    plt.figure(figsize=(8, 4))
    if not physical.empty:
        for _, row in physical.iterrows():
            for column, marker in (("detected_plateau_cycle", "o"), ("detected_plateau_exit_cycle", "x"), ("detected_severe_onset_cycle", "^")):
                if pd.notna(row[column]): plt.scatter([row[column]], [row.protocol_id], marker=marker, s=60, label=column)
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles: plt.legend(fontsize=7)
    plt.title("Post-hoc physical boundary comparison"); _save(output / "fig_csv3_physical_boundary_comparison.png")
