from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import MultiStageTrajectoryConfig
from .segmentation import Segment


@dataclass(frozen=True)
class RegimeStructure:
    centres: np.ndarray
    variances: np.ndarray
    transition: np.ndarray
    duration_mean: np.ndarray
    novelty_threshold: float
    selected_k: int
    source_hash: str
    descriptor_columns: tuple[str, ...]


def _bic(points: np.ndarray, labels: np.ndarray, centres: np.ndarray) -> float:
    sse = float(((points - centres[labels]) ** 2).sum()); n, d = points.shape; k = len(centres)
    return float(n * d * np.log(max(sse / max(n * d, 1), 1e-12)) + (k * d + k) * np.log(max(n, 2)))


def build_source_regime_model(descriptors: pd.DataFrame, segments: list[Segment], descriptor_columns: tuple[str, ...], config: MultiStageTrajectoryConfig) -> tuple[RegimeStructure, pd.DataFrame, np.ndarray]:
    values = descriptors.loc[:, list(descriptor_columns)].to_numpy(float)
    segment_points = np.vstack([values[segment.start:segment.end].mean(axis=0) for segment in segments])
    candidates: list[tuple[float, int, KMeans, np.ndarray]] = []
    for k in config.regime_k_candidates:
        if k > len(segment_points): continue
        fitted = KMeans(n_clusters=k, n_init=20, random_state=config.random_seed).fit(segment_points); counts = np.bincount(fitted.labels_, minlength=k)
        if np.all(counts >= config.regime_min_segment_count): candidates.append((_bic(segment_points, fitted.labels_, fitted.cluster_centers_), k, fitted, counts))
    if not candidates:
        # The source has too few consensus segments for the desired minimum occupancy;
        # choose the smallest executable K and record the limitation in the outputs.
        k = min(2, len(segment_points)); fitted = KMeans(n_clusters=k, n_init=20, random_state=config.random_seed).fit(segment_points); selected_k, labels = k, fitted.labels_
    else:
        _, selected_k, fitted, _ = min(candidates, key=lambda item: item[0]); labels = fitted.labels_
    window_labels = np.empty(len(values), dtype=int); duration = np.empty(selected_k, dtype=float); centres = np.empty((selected_k, values.shape[1]), dtype=float); variances = np.empty_like(centres)
    for state in range(selected_k):
        member_segments = [segment for segment, label in zip(segments, labels) if label == state]
        indices = np.concatenate([np.arange(segment.start, segment.end) for segment in member_segments]); window_labels[indices] = state
        state_values = values[indices]; centres[state] = state_values.mean(axis=0); variances[state] = np.maximum(state_values.var(axis=0), 1e-5)
        duration[state] = float(np.mean([segment.end_cycle - segment.start_cycle for segment in member_segments]))
    counts = np.full((selected_k, selected_k), 1e-3)
    for left, right in zip(labels[:-1], labels[1:]): counts[left, right] += 1.0
    empirical = counts / counts.sum(axis=1, keepdims=True); transition = config.regime_stickiness * np.eye(selected_k) + (1 - config.regime_stickiness) * empirical; transition /= transition.sum(axis=1, keepdims=True)
    distance = np.sqrt(np.mean((values - centres[window_labels]) ** 2 / variances[window_labels], axis=1)); threshold = float(np.quantile(distance, config.regime_novelty_quantile))
    digest = hashlib.sha256();
    for item in (centres, variances, transition, duration): digest.update(np.asarray(item, dtype=np.float64).tobytes())
    structure = RegimeStructure(centres, variances, transition, duration, max(threshold, 1e-9), int(selected_k), digest.hexdigest(), descriptor_columns)
    rows: list[dict[str, object]] = []
    for segment_id, (segment, state) in enumerate(zip(segments, labels)):
        rows.append({"row_type": "source_segment", "segment_id": segment_id, "regime": f"REGIME_{int(state)}", "regime_id": int(state), "start_cycle": segment.start_cycle, "end_cycle": segment.end_cycle, "duration": segment.end_cycle - segment.start_cycle, "selected_k": int(selected_k), "source_structure_hash": structure.source_hash})
    for state in range(selected_k):
        rows.append({"row_type": "source_prototype", "regime": f"REGIME_{state}", "regime_id": state, "selected_k": int(selected_k), "source_structure_hash": structure.source_hash, "centre_json": np.array2string(centres[state], precision=8, separator=","), "variance_json": np.array2string(variances[state], precision=8, separator=","), "mean_duration": duration[state], "novelty_threshold": structure.novelty_threshold})
    return structure, pd.DataFrame(rows), window_labels


def build_target_local_model(descriptors: pd.DataFrame, descriptor_columns: tuple[str, ...], k: int, config: MultiStageTrajectoryConfig) -> RegimeStructure:
    values = descriptors.loc[:, list(descriptor_columns)].to_numpy(float); fitted = KMeans(n_clusters=min(k, len(values)), n_init=20, random_state=config.random_seed + 991).fit(values); labels = fitted.labels_; selected = len(fitted.cluster_centers_)
    centres = fitted.cluster_centers_; variances = np.vstack([np.maximum(values[labels == state].var(axis=0), 1e-5) for state in range(selected)])
    transition_counts = np.full((selected, selected), 1e-3)
    for left, right in zip(labels[:-1], labels[1:]): transition_counts[left, right] += 1
    transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    duration = np.full(selected, max(float(descriptors.center_cycle.max() - descriptors.center_cycle.min()) / max(selected, 1), 1.0))
    distance = np.sqrt(np.mean((values - centres[labels]) ** 2 / variances[labels], axis=1)); threshold = max(float(np.quantile(distance, config.regime_novelty_quantile)), 1e-9)
    digest = hashlib.sha256(np.asarray(centres, dtype=np.float64).tobytes()).hexdigest()
    return RegimeStructure(centres, variances, transition, duration, threshold, selected, digest, descriptor_columns)
