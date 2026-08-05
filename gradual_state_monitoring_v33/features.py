from __future__ import annotations

import numpy as np


DESCRIPTOR_NAMES = (
    "current", "delta_4", "delta_16", "delta_32", "slope_16", "slope_32",
    "std_16", "std_32", "mean_16_minus_mean_32",
)


def _slope(values: np.ndarray) -> np.ndarray:
    """Per-feature least-squares slope over an already observed local interval."""
    count = len(values)
    if count < 2:
        return np.zeros(values.shape[1], dtype=float)
    time = np.arange(count, dtype=float)
    centered = time - time.mean()
    return (centered[:, None] * (values - values.mean(axis=0))).sum(axis=0) / max(float((centered * centered).sum()), 1.0)


def _recent(values: np.ndarray, position: int, width: int) -> np.ndarray:
    return values[max(0, position - width + 1): position + 1]


def build_causal_descriptors(values: np.ndarray) -> np.ndarray:
    """Create nine descriptions per feature; every element at t sees values <= t only."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected a 2D feature matrix")
    rows, feature_count = values.shape
    output = np.zeros((rows, feature_count * len(DESCRIPTOR_NAMES)), dtype=float)
    for t in range(rows):
        local16 = _recent(values, t, 16)
        local32 = _recent(values, t, 32)
        mean16 = local16.mean(axis=0)
        mean32 = local32.mean(axis=0)
        prior4 = values[max(0, t - 4)]
        prior16 = values[max(0, t - 16)]
        prior32 = values[max(0, t - 32)]
        parts = (
            values[t], values[t] - prior4, values[t] - prior16, values[t] - prior32,
            _slope(local16), _slope(local32), local16.std(axis=0), local32.std(axis=0),
            mean16 - mean32,
        )
        output[t] = np.concatenate(parts)
    return output


class CausalDescriptorScaler:
    """Online descriptor standardisation used only by the dual RLS models."""

    def __init__(self, dimensions: int, eps: float) -> None:
        self.count = 0
        self.mean = np.zeros(dimensions, dtype=float)
        self.m2 = np.zeros(dimensions, dtype=float)
        self.eps = eps

    def transform(self, row: np.ndarray) -> np.ndarray:
        if self.count < 2:
            return np.zeros_like(row, dtype=float)
        variance = self.m2 / max(self.count - 1, 1)
        return (row - self.mean) / np.sqrt(np.maximum(variance, self.eps))

    def update(self, row: np.ndarray) -> None:
        self.count += 1
        delta = row - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (row - self.mean)


class CausalFeatureScale:
    """Freeze a robust initial scale after the configured observed history period."""

    def __init__(self, dimensions: int, freeze_after: int, eps: float) -> None:
        self.dimensions = dimensions
        self.freeze_after = freeze_after
        self.eps = eps
        self.history: list[np.ndarray] = []
        self.location = np.zeros(dimensions, dtype=float)
        self.scale = np.ones(dimensions, dtype=float)
        self.frozen = False

    def update(self, value: np.ndarray) -> None:
        self.history.append(np.asarray(value, dtype=float).copy())
        # A rolling causal scale lets a delayed observer converge once it has
        # accumulated the same recent history, while never consulting future rows.
        if len(self.history) > self.freeze_after:
            self.history.pop(0)
        observed = np.vstack(self.history)
        self.location = np.median(observed, axis=0)
        mad = np.median(np.abs(observed - self.location), axis=0)
        iqr = np.percentile(observed, 75, axis=0) - np.percentile(observed, 25, axis=0)
        self.scale = np.maximum.reduce((1.4826 * mad, iqr / 1.349, np.full(self.dimensions, self.eps)))
        if not self.frozen and len(self.history) >= self.freeze_after:
            self.frozen = True

    def distance(self, value: np.ndarray, reference: np.ndarray) -> float:
        return float(np.mean(np.abs((value - reference) / self.scale)))
