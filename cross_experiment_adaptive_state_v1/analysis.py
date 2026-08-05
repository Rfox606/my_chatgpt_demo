from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .config import CrossExperimentAdaptiveConfig
from .data import temporal_pairs


COMPARATORS = {
    "Source_Static": "progression_prior",
    "Target_Local": "target_local_score",
    "Adaptive_Cross_Experiment": "progression_adapted",
    "Elapsed_Time_Since_Entry": "elapsed_time_since_entry_score",
}


def _rank(value: np.ndarray) -> np.ndarray:
    return pd.Series(value).rank(method="average").to_numpy(float)


def comparator_metrics(scores: pd.DataFrame, direction: str, config: CrossExperimentAdaptiveConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entry, entry_frame in scores.groupby("entry_cycle", sort=True):
        ordered = entry_frame.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
        time_rank = _rank(ordered.center_cycle.to_numpy(float))
        for name, column in COMPARATORS.items():
            values = ordered[column].to_numpy(float)
            for gap_number, gap in enumerate(config.source_gap_bins):
                pairs = temporal_pairs(ordered, (gap,), config.source_max_pairs_per_gap_bin, seed=config.random_seed + gap_number)
                if pairs.count:
                    delta = values[pairs.later] - values[pairs.earlier]
                    auc = float(np.mean(delta > 0) + .5 * np.mean(delta == 0))
                else:
                    auc = float("nan")
                rows.append({"direction": direction, "dataset": ordered.dataset.iloc[0], "entry_cycle": entry, "comparator": name,
                             "gap_lower": gap[0], "gap_upper": gap[1], "pair_count": pairs.count, "time_pair_auc": auc,
                             "spearman_progression_time": float(spearmanr(values, time_rank).statistic),
                             "kendall_progression_time": float(kendalltau(values, time_rank).statistic),
                             "evaluation_note": "time-order ranking only; not absolute wear"})
    return pd.DataFrame(rows)


def delayed_entry_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (direction, dataset), group in scores.groupby(["direction", "dataset"], sort=True):
        starts: list[dict[str, object]] = []
        for entry, frame in group.groupby("entry_cycle", sort=True):
            first = frame.sort_values("center_cycle").iloc[0]
            starts.append({"entry_cycle": float(entry), "initial_prior": float(first.progression_prior), "initial_adapted": float(first.progression_adapted),
                           "initial_nonzero": int(abs(float(first.progression_prior)) > 1e-12), "initial_activity": float(first.activity_score)})
            rows.append({"row_type": "entry_initialization", "direction": direction, "dataset": dataset, **starts[-1], "common_windows": np.nan,
                         "convergence_mean_std": np.nan, "convergence_mean_abs_error_vs_entry0": np.nan, "entry_prior_spearman": np.nan})
        start_frame = pd.DataFrame(starts)
        if len(start_frame) > 1:
            rank = spearmanr(start_frame.entry_cycle, start_frame.initial_prior).statistic
            rows.append({"row_type": "entry_ranking", "direction": direction, "dataset": dataset, "entry_cycle": np.nan, "initial_prior": np.nan, "initial_adapted": np.nan,
                         "initial_nonzero": int(start_frame.initial_nonzero.all()), "initial_activity": np.nan, "common_windows": np.nan,
                         "convergence_mean_std": np.nan, "convergence_mean_abs_error_vs_entry0": np.nan, "entry_prior_spearman": float(rank)})
        pivot = group.pivot_table(index="window_index", columns="entry_cycle", values="progression_adapted", aggfunc="first")
        complete = pivot.dropna()
        if complete.shape[1] >= 2:
            anchor = complete.iloc[:, 0]
            rows.append({"row_type": "common_suffix_convergence", "direction": direction, "dataset": dataset, "entry_cycle": np.nan, "initial_prior": np.nan,
                         "initial_adapted": np.nan, "initial_nonzero": np.nan, "initial_activity": np.nan, "common_windows": int(len(complete)),
                         "convergence_mean_std": float(complete.std(axis=1).mean()), "convergence_mean_abs_error_vs_entry0": float(complete.sub(anchor, axis=0).abs().mean().mean()),
                         "entry_prior_spearman": np.nan})
    return pd.DataFrame(rows)


def source_metrics(direction: str, source_name: str, models: dict[str, object]) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        coefficients = np.asarray(model.coefficients, dtype=float)
        for feature, coefficient in zip(model.feature_names, coefficients, strict=True):
            rows.append({"direction": direction, "source_dataset": source_name, "feature_configuration": name, "feature": feature,
                         "coefficient": float(coefficient), "absolute_coefficient": float(abs(coefficient)), "selected_c": model.selected_c,
                         "source_validation_time_pair_auc": model.validation_pair_auc, "source_validation_pair_count": model.validation_pair_count,
                         "source_ood_threshold": model.source_ood_threshold, "source_model_frozen_hash": model.frozen_hash})
    return pd.DataFrame(rows)


def feature_audit(config: CrossExperimentAdaptiveConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in config.feature_configs:
        for feature in config.jsonable()["feature_definitions"][name]:
            rows.append({"feature_configuration": name, "feature": feature, "formal_input": True, "source_training": True, "target_online_adaptation": True,
                         "selection_timing": "predeclared", "forbidden": False, "note": "direct force-ratio window summary"})
    for feature in ("Stage1to5", "Sa", "Sq", "Sz", "Sku", "wear_debris_count", "future target data", "cycle"):
        rows.append({"feature_configuration": "forbidden_or_index", "feature": feature, "formal_input": False, "source_training": False, "target_online_adaptation": False,
                     "selection_timing": "excluded", "forbidden": True, "note": "cycle is an ordering/audit index only" if feature == "cycle" else "forbidden formal input"})
    return pd.DataFrame(rows)


def posthoc_stage_diagnostics(target_scores: pd.DataFrame) -> pd.DataFrame:
    # The v4.5 raw-window artifact intentionally has no Stage field.  Record this
    # absence rather than silently importing labels into the formal pipeline.
    return pd.DataFrame([{"status": "NOT_AVAILABLE_INPUT_NOT_VERSIONED", "reason": "Stage is absent from the formal v4.5 raw-window input and was not fetched for CEAP v1 training/adaptation",
                          "stage_progression_spearman": np.nan, "ordinal_mae": np.nan, "confusion_matrix": "not available", "rows": len(target_scores)}])


def make_figures(scores: pd.DataFrame, comparison: pd.DataFrame, delayed: pd.DataFrame, paths: dict[str, Path]) -> None:
    figure_dir = paths["figures"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, (direction, group) in zip(axes, scores.loc[scores.entry_cycle.eq(0)].groupby("direction", sort=True), strict=False):
        axis.plot(group.center_cycle, group.progression_prior, label="prior", alpha=.75)
        axis.plot(group.center_cycle, group.progression_adapted, label="adapted", alpha=.85)
        axis.set_title(direction); axis.set_xlabel("target cycle"); axis.set_ylabel("progression score")
        axis.legend()
    fig.tight_layout(); fig.savefig(figure_dir / "prior_vs_adapted_bidirectional_v1.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, ((direction, dataset), group) in zip(axes, scores.groupby(["direction", "dataset"], sort=True), strict=False):
        for entry, frame in group.groupby("entry_cycle", sort=True):
            axis.plot(frame.center_cycle, frame.progression_adapted, label=f"entry {entry:g}", alpha=.75)
        axis.set_title(f"{direction} delayed entries"); axis.set_xlabel("target cycle"); axis.set_ylabel("adapted progression"); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figure_dir / "delayed_entry_convergence_v1.png", dpi=150); plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    for (direction, dataset), group in scores.loc[scores.entry_cycle.eq(0)].groupby(["direction", "dataset"], sort=True):
        view = group.iloc[:: max(1, len(group) // 1000)]
        axis.scatter(view.progression_adapted, view.activity_score, s=5, alpha=.45, label=f"{direction}: {dataset}")
    axis.set_xlabel("progression score"); axis.set_ylabel("activity score"); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figure_dir / "progression_activity_space_v1.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, (direction, group) in zip(axes, scores.loc[scores.entry_cycle.eq(0)].groupby("direction", sort=True), strict=False):
        axis.plot(group.center_cycle, group.state_uncertainty, color="tab:purple")
        axis.set_title(direction); axis.set_xlabel("target cycle"); axis.set_ylabel("state uncertainty")
    fig.tight_layout(); fig.savefig(figure_dir / "uncertainty_over_time_v1.png", dpi=150); plt.close(fig)

    summary = comparison.groupby(["direction", "comparator"], as_index=False).time_pair_auc.mean()
    fig, axis = plt.subplots(figsize=(8, 4))
    labels = [f"{row.direction}\n{row.comparator}" for row in summary.itertuples(index=False)]
    axis.bar(np.arange(len(summary)), summary.time_pair_auc, color="tab:blue")
    axis.set_xticks(np.arange(len(summary)), labels, rotation=35, ha="right", fontsize=8); axis.set_ylim(0, 1); axis.set_ylabel("mean target time-pair AUC")
    fig.tight_layout(); fig.savefig(figure_dir / "model_comparison_v1.png", dpi=150); plt.close(fig)
