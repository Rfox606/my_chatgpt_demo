from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def transition_score_figure(scores: pd.DataFrame, events: pd.DataFrame, *, direction: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 4.5))
    selected = scores[(scores.direction == direction) & (scores.setting == "OnlineAdaptation") & (scores.seed == scores.seed.min())]
    if len(selected):
        axis.plot(selected.center_cycle, selected.final_transition_score, lw=0.8, label="final transition score")
        local_events = events[(events.direction == direction) & (events.setting == "OnlineAdaptation") & (events.seed == selected.seed.iloc[0])]
        if len(local_events):
            axis.scatter(local_events.peak_cycle, local_events.peak_score, color="crimson", s=18, label="candidate peak")
    axis.set(xlabel="effective cycle", ylabel="score", title=f"V3.4 {direction}: online adaptation")
    axis.legend(loc="best"); _save(figure, path)


def rank_comparison_figure(ranking: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if len(ranking):
        values = ranking.groupby("setting", sort=False).boundary_score_percentile.mean().sort_values(ascending=False)
        axis.bar(values.index, values.values, color="#4472c4")
        axis.tick_params(axis="x", rotation=25)
    axis.set(ylabel="mean boundary-score percentile", title="V3.4 boundary rank comparison")
    _save(figure, path)


def adaptation_ablation_figure(methods: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    selected = methods[methods.metric == "proximity_weighted_score"] if len(methods) else methods
    if len(selected):
        values = selected.groupby("setting", sort=False).value.mean()
        axis.bar(values.index, values.values, color="#70ad47")
        axis.tick_params(axis="x", rotation=25)
    axis.set(ylabel="mean proximity-weighted score", title="V3.4 adaptation ablation")
    _save(figure, path)


def topk_recall_figure(ranking_summary: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if len(ranking_summary):
        values = ranking_summary.groupby(["metric", "setting"], sort=False).value.mean().unstack(fill_value=0.0)
        values.plot(kind="bar", ax=axis)
        axis.tick_params(axis="x", rotation=0)
    axis.set(ylabel="recall", title="V3.4 Top-K candidate-event recall")
    _save(figure, path)
