from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def make_figures(stability: pd.DataFrame, common: pd.DataFrame, branch: pd.DataFrame, scores: pd.DataFrame, forecasts: pd.DataFrame, metrics: pd.DataFrame, candidates: pd.DataFrame, directory: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4));
    for direction, group in stability.groupby("direction_id"): ax.errorbar(group.feature_name, group.median_weight, yerr=[group.median_weight-group.weight_p05, group.weight_p95-group.median_weight], fmt="o", label=direction)
    ax.tick_params(axis="x", rotation=40); ax.axhline(0,color="black",lw=.8); ax.legend(); ax.set_title("Source bootstrap weight stability"); _save(fig, directory/"fig_csv2_source_weight_stability.png")
    fig, ax = plt.subplots(figsize=(9,4)); x=np.arange(len(branch)); ax.bar(x-.2, branch.w_branch, .4,label="branch");
    if not common.empty: ax.bar(x+.2, common.set_index("feature_name").reindex(branch.feature_name).w_common.fillna(0),.4,label="common")
    ax.set_xticks(x, branch.feature_name, rotation=40); ax.legend(); ax.set_title("Common and terminal-branch axes"); _save(fig,directory/"fig_csv2_common_branch_axes.png")
    for dataset in ("Exp1","Exp2"):
        q=scores[(scores.dataset==dataset)&(scores.dataset_role=="target")]
        fig,ax=plt.subplots(figsize=(9,4)); ax.plot(q.center_cycle,q.P_common,label="P"); ax.plot(q.center_cycle,q.BD,label="BD"); ax.plot(q.center_cycle,q.B_terminal,label="B"); ax.legend(); ax.set_title(f"{dataset} target state trajectory"); _save(fig,directory/f"fig_csv2_{dataset.lower()}_state_trajectory.png")
    q=scores[scores.dataset_role=="target"]; fig,ax=plt.subplots(figsize=(6,5));
    for direction,g in q.groupby("direction_id"): ax.scatter(g.P_common,g.BD,s=4,label=direction)
    ax.legend(); ax.set(xlabel="P_common",ylabel="BD",title="P–BD plane"); _save(fig,directory/"fig_csv2_P_BD_plane.png")
    fig,ax=plt.subplots(figsize=(6,5));
    for direction,g in q.groupby("direction_id"): ax.scatter(g.P_common,g.B_terminal,s=4,label=direction)
    ax.legend(); ax.set(xlabel="P_common",ylabel="B_terminal",title="Common process vs branch"); _save(fig,directory/"fig_csv2_P_B_branch_3d_or_pairwise.png")
    fig,ax=plt.subplots(figsize=(9,4));
    for direction,g in q.groupby("direction_id"): ax.plot(g.center_cycle,g.weighted_oos_common,label=direction)
    ax.legend(); ax.set_title("Weighted OOS over time"); _save(fig,directory/"fig_csv2_weighted_oos_over_time.png")
    fig,ax=plt.subplots(figsize=(9,4));
    for direction,g in q.groupby("direction_id"): ax.plot(g.center_cycle,g.beta_norm,label=direction)
    ax.legend(); ax.set_title("Adapter beta norm"); _save(fig,directory/"fig_csv2_adapter_beta_norm.png")
    fig,ax=plt.subplots(figsize=(9,4));
    for direction,g in q.groupby("direction_id"): ax.plot(g.center_cycle,g.pre_update_P_common,label=direction)
    ax.legend(); ax.set_title("Predict-then-update state score"); _save(fig,directory/"fig_csv2_frozen_vs_adapted_state.png")
    fig,ax=plt.subplots(figsize=(8,4));
    if not metrics.empty:
        for model,g in metrics.groupby("model"): ax.plot(g.horizon_cycles,g.MAE_P,marker="o",label=f"{model} P MAE")
    ax.legend(); ax.set_title("Forecast MAE comparison"); _save(fig,directory/"fig_csv2_forecast_mae_comparison.png")
    fig,ax=plt.subplots(figsize=(8,4));
    if not metrics.empty:
        for model,g in metrics.groupby("model"): ax.plot(g.horizon_cycles,g.direction_accuracy_P,marker="o",label=f"{model} P direction")
    ax.legend(); ax.set_title("Forecast direction accuracy"); _save(fig,directory/"fig_csv2_forecast_direction_accuracy.png")
    fig,ax=plt.subplots(figsize=(9,4));
    if not candidates.empty:
        for kind,g in candidates.groupby("candidate_type"): ax.scatter(g.peak_cycle,g.peak_BD,s=20,label=kind)
    if not candidates.empty: ax.legend(fontsize=6,ncol=2)
    ax.set_title("Physical-validation candidate regions"); _save(fig,directory/"fig_csv2_physical_candidates.png")
