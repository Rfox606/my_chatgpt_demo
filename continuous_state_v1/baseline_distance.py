from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import ContinuousStateV1Config
from .data import assert_label_free
from .target_anchor import baseline_non_guard_mask


@dataclass(frozen=True)
class BaselineDistanceModel:
    mean: np.ndarray
    precision: np.ndarray | None
    method: str
    baseline_mask: np.ndarray


def fit_baseline_distance(
    frame: pd.DataFrame, config: ContinuousStateV1Config
) -> BaselineDistanceModel:
    assert_label_free(frame)
    baseline_mask = baseline_non_guard_mask(frame, config)
    values = frame.loc[:, list(config.stable_plus_features)].to_numpy(dtype=float)
    baseline = values[baseline_mask]
    try:
        if len(baseline) < 2:
            raise ValueError("Need at least two baseline rows for covariance fitting")
        estimator = LedoitWolf().fit(baseline)
        precision = np.linalg.pinv(estimator.covariance_)
        if not np.isfinite(precision).all():
            raise FloatingPointError("Non-finite precision matrix")
        return BaselineDistanceModel(
            mean=np.asarray(estimator.location_, dtype=float),
            precision=precision,
            method="ledoit_wolf_mahalanobis",
            baseline_mask=baseline_mask,
        )
    except Exception:
        return BaselineDistanceModel(
            mean=np.zeros(values.shape[1], dtype=float),
            precision=None,
            method="robust_diagonal_fallback",
            baseline_mask=baseline_mask,
        )


def score_baseline_distance(
    frame: pd.DataFrame, model: BaselineDistanceModel, config: ContinuousStateV1Config
) -> pd.DataFrame:
    assert_label_free(frame)
    result = frame.copy()
    values = result.loc[:, list(config.stable_plus_features)].to_numpy(dtype=float)
    result["BD_diag"] = np.sqrt(np.mean(values**2, axis=1))
    if model.precision is None:
        result["BD"] = result["BD_diag"]
    else:
        delta = values - model.mean
        squared = np.einsum("ij,jk,ik->i", delta, model.precision, delta)
        result["BD"] = np.sqrt(np.maximum(squared, 0.0))
    result["bd_method"] = model.method
    return result
