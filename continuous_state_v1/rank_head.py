from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .config import ContinuousStateV1Config
from .data import assert_label_free
from .pair_sampling import PairBatch, build_pair_batch


@dataclass(frozen=True)
class RankModel:
    model: LogisticRegression
    raw_coefficient: np.ndarray
    normalized_weight: np.ndarray
    selected_C: float
    validation_auc: float
    orientation_flipped: bool

    def score_deltas(self, delta_x: np.ndarray) -> np.ndarray:
        return np.asarray(delta_x, dtype=float) @ self.normalized_weight


def _fit(batch: PairBatch, C: float, config: ContinuousStateV1Config) -> LogisticRegression:
    if batch.pair_count == 0:
        raise ValueError("Cannot fit a rank model with no temporal pairs")
    model = LogisticRegression(
        penalty="l2",
        C=C,
        fit_intercept=False,
        solver=config.rank_solver,
        max_iter=5000,
        random_state=config.pair_random_seed,
    )
    model.fit(batch.delta_x, batch.labels)
    return model


def pair_metrics(coefficient: np.ndarray, batch: PairBatch) -> dict[str, float]:
    scores = batch.delta_x @ np.asarray(coefficient, dtype=float)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -35.0, 35.0)))
    return {
        "source_pair_auc": float(roc_auc_score(batch.labels, scores)),
        "source_pair_accuracy": float(accuracy_score(batch.labels, probabilities >= 0.5)),
        "source_pair_logloss": float(log_loss(batch.labels, probabilities, labels=[0, 1])),
    }


def select_rank_C(
    source_train: pd.DataFrame,
    source_validation: pd.DataFrame,
    config: ContinuousStateV1Config,
) -> tuple[float, pd.DataFrame, PairBatch, PairBatch]:
    """Choose C exclusively on held-out source time pairs."""
    assert_label_free(source_train)
    assert_label_free(source_validation)
    train_pairs = build_pair_batch(source_train, config, random_seed=config.pair_random_seed)
    validation_pairs = build_pair_batch(source_validation, config, random_seed=config.pair_random_seed + 1)
    rows: list[dict[str, float]] = []
    for C in config.rank_C_grid:
        model = _fit(train_pairs, C, config)
        metrics = pair_metrics(model.coef_.reshape(-1), validation_pairs)
        rows.append({"C": float(C), **metrics})
    grid = pd.DataFrame(rows).sort_values("C").reset_index(drop=True)
    best_auc = float(grid["source_pair_auc"].max())
    # Within the prescribed tolerance, retain the more strongly regularised model.
    eligible = grid.loc[best_auc - grid["source_pair_auc"] < 0.005]
    selected_C = float(eligible.sort_values("C").iloc[0]["C"])
    return selected_C, grid, train_pairs, validation_pairs


def fit_final_rank_model(
    source_all: pd.DataFrame,
    source_validation_pairs: PairBatch,
    selected_C: float,
    config: ContinuousStateV1Config,
) -> tuple[RankModel, PairBatch, dict[str, float]]:
    """Refit on every non-guard source window after source-only selection."""
    assert_label_free(source_all)
    all_pairs = build_pair_batch(source_all, config, random_seed=config.pair_random_seed + 2)
    final_model = _fit(all_pairs, selected_C, config)
    raw = final_model.coef_.reshape(-1).astype(float)
    validation_metrics = pair_metrics(raw, source_validation_pairs)
    flip = validation_metrics["source_pair_auc"] < 0.5
    if flip:
        raw = -raw
        validation_metrics = pair_metrics(raw, source_validation_pairs)
    normalised = raw / (np.abs(raw).sum() + config.eps)
    return (
        RankModel(
            model=final_model,
            raw_coefficient=raw,
            normalized_weight=normalised,
            selected_C=selected_C,
            validation_auc=float(validation_metrics["source_pair_auc"]),
            orientation_flipped=bool(flip),
        ),
        all_pairs,
        validation_metrics,
    )


def coefficient_table(rank_model: RankModel, features: tuple[str, ...]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "feature_name": list(features),
            "raw_coefficient": rank_model.raw_coefficient,
            "normalized_weight": rank_model.normalized_weight,
        }
    )
    table["abs_weight"] = table["normalized_weight"].abs()
    table["direction"] = np.where(table["normalized_weight"] >= 0, "positive", "negative")
    table["rank"] = table["abs_weight"].rank(method="first", ascending=False).astype(int)
    return table.sort_values("rank").reset_index(drop=True)


def pair_auc_by_gap(rank_model: RankModel, batch: PairBatch) -> pd.DataFrame:
    rows = []
    positive_count = batch.pair_count
    for name, subset in batch.positive_pairs.groupby("gap_bin", sort=True):
        positions = subset.index.to_numpy(dtype=int)
        positive_delta = batch.delta_x[positions]
        mirrored_delta = np.vstack([positive_delta, -positive_delta])
        labels = np.concatenate([np.ones(len(positive_delta), dtype=int), np.zeros(len(positive_delta), dtype=int)])
        scores = rank_model.score_deltas(mirrored_delta)
        rows.append(
            {
                "gap_bin": name,
                "pair_count": int(len(positive_delta)),
                "pair_auc": float(roc_auc_score(labels, scores)),
                "source_batch_positive_pair_count": int(positive_count),
            }
        )
    return pd.DataFrame(rows)
