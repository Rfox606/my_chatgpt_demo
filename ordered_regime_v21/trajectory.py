from __future__ import annotations

import numpy as np


def causal_trajectory(embedding: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = len(embedding)
    d20 = np.zeros_like(embedding); d100 = np.zeros_like(embedding)
    for index in range(n):
        prior20 = embedding[max(0, index - 20):index]
        prior100 = embedding[max(0, index - 100):index]
        if len(prior20):
            d20[index] = embedding[index] - np.median(prior20, axis=0)
        if len(prior100):
            d100[index] = embedding[index] - np.median(prior100, axis=0)
    trajectory = np.concatenate([embedding, .5 * d20, .25 * d100], axis=1)
    return trajectory, {"trajectory_norm20": np.linalg.norm(d20, axis=1), "trajectory_norm100": np.linalg.norm(d100, axis=1)}


def trimmed_mean(values: np.ndarray, proportion: float = .20) -> np.ndarray:
    if len(values) <= 2:
        return values.mean(axis=0)
    low, high = np.quantile(values, [proportion, 1 - proportion], axis=0)
    kept = np.where((values >= low) & (values <= high), values, np.nan)
    result = np.nanmean(kept, axis=0)
    return np.where(np.isfinite(result), result, np.nanmean(values, axis=0))


def robust_z(value: float, reference: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    median = float(np.median(ref)); mad = float(np.median(np.abs(ref - median)))
    return max(0.0, (value - median) / max(1.4826 * mad, 1e-6))
