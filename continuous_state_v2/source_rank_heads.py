from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .config import ContinuousStateV2Config
from .data import assert_label_free
from .temporal_pairs import PairBatch, build_pair_batch, split_source


@dataclass(frozen=True)
class RankHead:
    features: tuple[str, ...]
    coefficient: np.ndarray
    normalized_weight: np.ndarray
    selected_C: float
    pre_refit_metrics: dict[str, float]
    after_refit_replay_auc: float


def _fit(batch: PairBatch, C: float, config: ContinuousStateV2Config) -> LogisticRegression:
    if batch.pair_count == 0:
        raise ValueError("Rank fit has no temporal pairs")
    model = LogisticRegression(penalty="l2", C=C, fit_intercept=False, solver="liblinear", max_iter=5000, random_state=config.random_seed)
    model.fit(batch.delta_x, batch.labels)
    return model


def _metrics(coef: np.ndarray, batch: PairBatch) -> dict[str, float]:
    score = batch.delta_x @ coef
    prob = 1 / (1 + np.exp(-np.clip(score, -35, 35)))
    return {"auc": float(roc_auc_score(batch.labels, score)), "accuracy": float(accuracy_score(batch.labels, prob >= .5)), "logloss": float(log_loss(batch.labels, prob, labels=[0, 1]))}


def _gap_auc(coef: np.ndarray, batch: PairBatch) -> pd.DataFrame:
    rows = []
    for gap, pairs in batch.pairs.groupby("gap_bin", sort=True):
        ix = pairs.index.to_numpy(int)
        positive = batch.delta_x[ix]
        x = np.vstack([positive, -positive])
        y = np.r_[np.ones(len(positive), int), np.zeros(len(positive), int)]
        rows.append({"gap_bin": gap, "pair_count": len(positive), "source_validation_auc_by_gap_pre_refit": float(roc_auc_score(y, x @ coef))})
    return pd.DataFrame(rows)


def train_source_head(source: pd.DataFrame, features: tuple[str, ...], direction_id: str, config: ContinuousStateV2Config) -> tuple[RankHead, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Selection head stays train-only; deployment head is refit only after C is frozen."""
    assert_label_free(source)
    train, validation, gap = split_source(source, config)
    train_batch = build_pair_batch(train, features, config, config.pair_random_seed)
    validation_batch = build_pair_batch(validation, features, config, config.pair_random_seed + 1)
    rows = []
    selection_models: dict[float, LogisticRegression] = {}
    for C in config.rank_C_grid:
        selection_models[C] = _fit(train_batch, C, config)
        rows.append({"C": C, **_metrics(selection_models[C].coef_.reshape(-1), validation_batch)})
    grid = pd.DataFrame(rows).sort_values("C")
    best_auc = float(grid.auc.max())
    selected_C = float(grid.loc[best_auc - grid.auc < .005].sort_values("C").iloc[0].C)
    selection_coef = selection_models[selected_C].coef_.reshape(-1)
    pre = _metrics(selection_coef, validation_batch)
    flipped = pre["auc"] < .5
    if flipped:
        selection_coef = -selection_coef
        pre = _metrics(selection_coef, validation_batch)
    pre_gap = _gap_auc(selection_coef, validation_batch)
    # Only now refit a separate deployment model on all non-guard source windows.
    deploy_batch = build_pair_batch(source, features, config, config.pair_random_seed + 2)
    deployment_model = _fit(deploy_batch, selected_C, config)
    deploy_coef = deployment_model.coef_.reshape(-1)
    replay = _metrics(deploy_coef, validation_batch)
    if flipped:
        deploy_coef = -deploy_coef
    normalized = deploy_coef / (np.abs(deploy_coef).sum() + config.eps)
    head = RankHead(features, deploy_coef, normalized, selected_C,
                    {"source_validation_auc_pre_refit": pre["auc"], "source_validation_accuracy_pre_refit": pre["accuracy"], "source_validation_logloss_pre_refit": pre["logloss"]},
                    replay["auc"])
    summary = pd.DataFrame([{"direction_id": direction_id, "source_dataset": str(source.dataset.iloc[0]), "selected_C": selected_C, **head.pre_refit_metrics, "source_validation_auc_after_refit_replay": replay["auc"], "pair_count_train": train_batch.pair_count, "pair_count_validation": validation_batch.pair_count, "source_gap_windows": len(gap)}])
    pre_gap["direction_id"] = direction_id
    audit = {"train_endpoint_count": train_batch.pair_count, "validation_endpoint_count": validation_batch.pair_count, "train_window_ids": set(train.window_id), "validation_window_ids": set(validation.window_id), "train_pairs": train_batch.pairs, "validation_pairs": validation_batch.pairs}
    return head, summary, pre_gap, audit
