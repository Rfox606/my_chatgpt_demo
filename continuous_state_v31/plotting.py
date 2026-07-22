from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _targets(states: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {str(frame.dataset.iloc[0]): frame.sort_values("center_cycle") for _, frame in states.groupby("protocol_id")}


def make_figures(
    states: pd.DataFrame,
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    rolling: pd.DataFrame,
    regret: pd.DataFrame,
    episodes: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    targets = _targets(states)
    plt.figure(figsize=(10, 4))
    for name, frame in targets.items():
        plt.plot(frame.center_cycle, frame.plateau_condition.rolling(50, min_periods=1).mean(), label=name)
    plt.legend(); plt.title("Plateau-condition rate"); plt.xlabel("cycle")
    _save(output / "fig_csv31_plateau_condition_rates.png")

    plt.figure(figsize=(10, 4))
    for name, frame in targets.items():
        plt.plot(frame.center_cycle, frame.plateau_valid_cycles, label=name)
    plt.legend(); plt.title("Guard-aware valid-cycle plateau evidence"); plt.xlabel("cycle")
    _save(output / "fig_csv31_plateau_valid_cycle_accumulation.png")

    for dataset, filename in (("Exp1", "fig_csv31_exp1_plateau.png"), ("Exp2", "fig_csv31_exp2_plateau_exit.png")):
        frame = targets.get(dataset, pd.DataFrame())
        plt.figure(figsize=(10, 4))
        if not frame.empty:
            plt.plot(frame.center_cycle, frame.D_state, label="D_state", linewidth=.8)
            plt.plot(frame.center_cycle, frame.V50_norm, label="V50_norm", linewidth=.8)
            plt.step(frame.center_cycle, frame.plateau_locked, label="plateau_locked", where="mid")
            if dataset == "Exp2":
                plt.step(frame.center_cycle, frame.plateau_exit_confirmed, label="exit_confirmed", where="mid")
            plt.legend()
        plt.title(f"{dataset} causal plateau state"); plt.xlabel("cycle")
        _save(output / filename)

    plt.figure(figsize=(10, 4))
    for name, frame in targets.items():
        plt.plot(frame.center_cycle, frame.S_smooth_50, label=name)
    plt.legend(); plt.title("Severe candidate score"); plt.xlabel("cycle")
    _save(output / "fig_csv31_severe_candidate.png")

    plt.figure(figsize=(6, 5))
    for name, frame in targets.items():
        plt.scatter(frame.D_state, frame.V50_norm, s=2, alpha=.4, label=name)
    plt.legend(); plt.xlabel("D_state"); plt.ylabel("V50_norm")
    _save(output / "fig_csv31_D_V_trajectory.png")

    plt.figure(figsize=(10, 4))
    if not metrics.empty:
        subset = metrics.loc[metrics.horizon_cycles.isin([500, 1000])]
        for model, frame in subset.groupby("model"):
            plt.plot(range(len(frame)), frame.MAE, marker="o", linewidth=.8, label=model)
        plt.legend(fontsize=7, ncol=2)
    plt.title("Forecast comparison at 500/1000 cycles")
    _save(output / "fig_csv31_forecast_baseline_comparison.png")

    plt.figure(figsize=(10, 4))
    if not rolling.empty:
        subset = rolling.loc[rolling.model.eq("Safe_Ensemble")]
        for label, frame in subset.groupby(["protocol_id", "output_name", "horizon_cycles"]):
            plt.plot(frame.due_observation_cycle, frame.rolling_MAE, linewidth=.6, label="/".join(map(str, label)))
        plt.legend(fontsize=6, ncol=2)
    plt.title("Safe Ensemble rolling MAE")
    _save(output / "fig_csv31_forecast_rolling_mae.png")

    plt.figure(figsize=(10, 4))
    if not regret.empty:
        for label, frame in regret.groupby(["protocol_id", "baseline_model"]):
            plt.plot(frame.due_observation_cycle, frame.cumulative_regret, linewidth=.6, label="/".join(map(str, label)))
        plt.legend(fontsize=6, ncol=2)
    plt.title("Safe Ensemble cumulative regret (lower is better)")
    _save(output / "fig_csv31_cumulative_regret.png")

    plt.figure(figsize=(10, 4))
    if not forecasts.empty:
        for label, frame in forecasts.groupby(["protocol_id", "output_name", "horizon_cycles"]):
            plt.plot(frame.prediction_origin_cycle, frame.ensemble_alpha, linewidth=.5, label="/".join(map(str, label)))
        plt.legend(fontsize=6, ncol=2)
    plt.title("Safe Ensemble alpha")
    _save(output / "fig_csv31_ensemble_state.png")

    plt.figure(figsize=(10, 3))
    if not episodes.empty:
        for index, (_, row) in enumerate(episodes.iterrows()):
            plt.hlines(index, row.freeze_start_cycle, row.freeze_until_cycle, linewidth=4)
        plt.yticks(range(len(episodes)), [f"{row.output_name}@{row.horizon_cycles}" for _, row in episodes.iterrows()])
    plt.title("Independent Safe Ensemble reset episodes")
    _save(output / "fig_csv31_reset_episodes.png")

    plt.figure(figsize=(8, 4))
    if not sensitivity.empty:
        for label, frame in sensitivity.groupby("protocol_id"):
            plt.plot(frame["quantile"], frame.plateau_lock_cycle, marker="o", label=label)
        plt.legend()
    plt.title("Pre-registered plateau threshold sensitivity")
    _save(output / "fig_csv31_threshold_sensitivity.png")
