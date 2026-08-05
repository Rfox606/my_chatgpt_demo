from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import CrossExperimentAdaptiveConfig
from .data import TemporalPairs, assert_formal_frame, required_features, source_train_validation_split, temporal_pairs


@dataclass(frozen=True)
class RobustReference:
    location: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.location) / self.scale


def robust_reference(values: np.ndarray, eps: float = 1e-9) -> RobustReference:
    location = np.median(values, axis=0)
    mad = np.median(np.abs(values - location), axis=0)
    iqr = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
    scale = np.maximum.reduce((1.4826 * mad, iqr / 1.349, np.full(values.shape[1], eps)))
    return RobustReference(location, scale)


def _pair_design(z: np.ndarray, pairs: TemporalPairs) -> tuple[np.ndarray, np.ndarray]:
    delta = z[pairs.later] - z[pairs.earlier]
    # Mirroring makes the rank orientation explicit and prevents a hidden intercept/time shortcut.
    design = np.vstack((delta, -delta))
    labels = np.concatenate((np.ones(len(delta), dtype=int), np.zeros(len(delta), dtype=int)))
    return design, labels


def _pair_accuracy(scores: np.ndarray, pairs: TemporalPairs) -> float:
    if pairs.count == 0:
        return float("nan")
    diff = scores[pairs.later] - scores[pairs.earlier]
    return float(np.mean(diff > 0.0) + .5 * np.mean(diff == 0.0))


@dataclass(frozen=True)
class SourceRanker:
    config_name: str
    feature_names: tuple[str, ...]
    reference: RobustReference
    coefficients: np.ndarray
    rank_knots: np.ndarray
    rank_values: np.ndarray
    selected_c: float
    validation_pair_auc: float
    validation_pair_count: int
    source_ood_threshold: float
    frozen_hash: str

    def z(self, frame: pd.DataFrame) -> np.ndarray:
        assert_formal_frame(frame)
        return self.reference.transform(frame.loc[:, list(self.feature_names)].to_numpy(float))

    def raw_score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.z(frame) @ self.coefficients

    def progression_prior(self, frame: pd.DataFrame) -> np.ndarray:
        return np.interp(self.raw_score(frame), self.rank_knots, self.rank_values, left=0.0, right=1.0)

    def ood_ratio(self, frame: pd.DataFrame) -> np.ndarray:
        distance = np.sqrt(np.mean(self.z(frame) ** 2, axis=1))
        return distance / max(self.source_ood_threshold, 1e-9)


def _fit_ranker(frame: pd.DataFrame, feature_names: tuple[str, ...], pairs: TemporalPairs, c_value: float) -> tuple[RobustReference, np.ndarray]:
    reference = robust_reference(frame.loc[:, list(feature_names)].to_numpy(float))
    z = reference.transform(frame.loc[:, list(feature_names)].to_numpy(float))
    design, labels = _pair_design(z, pairs)
    if len(np.unique(labels)) < 2 or len(design) < 8:
        raise ValueError("Insufficient source time pairs for ranking")
    classifier = LogisticRegression(C=c_value, penalty="l2", fit_intercept=False, max_iter=1000, solver="lbfgs", random_state=0)
    classifier.fit(design, labels)
    return reference, classifier.coef_.reshape(-1)


def train_source_ranker(frame: pd.DataFrame, config_name: str, config: CrossExperimentAdaptiveConfig) -> SourceRanker:
    """Select C on a source-only chronological validation segment, then refit on source."""
    assert_formal_frame(frame)
    features = required_features(config_name)
    train, validation, _ = source_train_validation_split(frame, config)
    train_pairs = temporal_pairs(train, config.source_gap_bins, config.source_max_pairs_per_gap_bin, seed=config.random_seed)
    validation_pairs = temporal_pairs(validation, config.source_gap_bins, config.source_max_pairs_per_gap_bin, seed=config.random_seed + 1)
    choices: list[tuple[float, float]] = []
    for c_value in config.rank_c_values:
        reference, coefficient = _fit_ranker(train, features, train_pairs, c_value)
        scores = reference.transform(validation.loc[:, list(features)].to_numpy(float)) @ coefficient
        choices.append((float(_pair_accuracy(scores, validation_pairs)), float(c_value)))
    best_auc, best_c = max(choices, key=lambda value: (value[0], -value[1]))
    all_pairs = temporal_pairs(frame, config.source_gap_bins, config.source_max_pairs_per_gap_bin, seed=config.random_seed + 2)
    reference, coefficient = _fit_ranker(frame, features, all_pairs, best_c)
    raw_scores = reference.transform(frame.loc[:, list(features)].to_numpy(float)) @ coefficient
    order = np.sort(raw_scores)
    # Preserve a transferred prior even for a target point outside the source-score
    # extrema.  The closed [0, 1] display scale deliberately reserves small margins
    # rather than silently resetting a new experiment's first score to exactly zero.
    values = np.linspace(0.01, 0.99, len(order), endpoint=True)
    # Coalesce tied scores so interpolation remains strictly monotone and deterministic.
    rank_knots, unique_indices = np.unique(order, return_index=True)
    rank_values = values[unique_indices]
    source_distance = np.sqrt(np.mean(reference.transform(frame.loc[:, list(features)].to_numpy(float)) ** 2, axis=1))
    frozen_hash = __import__("hashlib").sha256(np.asarray(coefficient, dtype=np.float64).tobytes()).hexdigest()
    return SourceRanker(
        config_name=config_name,
        feature_names=features,
        reference=reference,
        coefficients=coefficient,
        rank_knots=rank_knots,
        rank_values=rank_values,
        selected_c=best_c,
        validation_pair_auc=best_auc,
        validation_pair_count=validation_pairs.count,
        source_ood_threshold=float(np.quantile(source_distance, config.ood_quantile)),
        frozen_hash=frozen_hash,
    )


def train_source_models(source: pd.DataFrame, config: CrossExperimentAdaptiveConfig) -> dict[str, SourceRanker]:
    return {name: train_source_ranker(source, name, config) for name in config.feature_configs}


def pair_auc_by_gap(frame: pd.DataFrame, score_column: str, config: CrossExperimentAdaptiveConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ordered = frame.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    for gap_index, gap in enumerate(config.source_gap_bins):
        pairs = temporal_pairs(ordered, (gap,), config.source_max_pairs_per_gap_bin, seed=config.random_seed + 100 + gap_index)
        scores = ordered[score_column].to_numpy(float)
        rows.append({"gap_lower": gap[0], "gap_upper": gap[1], "pair_count": pairs.count, "time_pair_auc": _pair_accuracy(scores, pairs)})
    return rows
