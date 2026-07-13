from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import ContinuousStateV2Config
from .data import assert_label_free, baseline_non_guard_mask


@dataclass(frozen=True)
class DistanceBaseline:
    mean: np.ndarray
    precision: np.ndarray | None
    method: str
    mask: np.ndarray


def fit_distance_baseline(frame: pd.DataFrame, features: tuple[str, ...], config: ContinuousStateV2Config) -> DistanceBaseline:
    assert_label_free(frame)
    mask = baseline_non_guard_mask(frame, config).to_numpy(bool)
    values = frame.loc[:, list(features)].to_numpy(float)
    try:
        estimator = LedoitWolf().fit(values[mask])
        precision = np.linalg.pinv(estimator.covariance_)
        if not np.isfinite(precision).all():
            raise FloatingPointError
        return DistanceBaseline(np.asarray(estimator.location_), precision, "ledoit_wolf_mahalanobis", mask)
    except Exception:
        return DistanceBaseline(np.zeros(len(features)), None, "robust_diagonal_fallback", mask)


def distance_values(values: np.ndarray, baseline: DistanceBaseline) -> tuple[np.ndarray, np.ndarray]:
    diag = np.sqrt(np.mean(values ** 2, axis=1))
    if baseline.precision is None:
        return diag.copy(), diag
    delta = values - baseline.mean
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, baseline.precision, delta), 0.0)), diag
