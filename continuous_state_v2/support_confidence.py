from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV2Config
from .data import assert_label_free


def source_support(source: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    assert_label_free(source)
    usable = source.loc[source.is_restart_guard.eq(0)]
    return pd.DataFrame([{"feature_name": f, "p01": usable[f].quantile(.01), "p99": usable[f].quantile(.99)} for f in features])


def support_scores(values: np.ndarray, features: tuple[str, ...], support: pd.DataFrame, common_features: tuple[str, ...], w_common: np.ndarray, w_branch: np.ndarray, source_features: tuple[str, ...], source_weight: np.ndarray) -> dict[str, float]:
    limits = support.set_index("feature_name") if "feature_name" in support.columns else support
    flags = np.asarray([int(value < limits.loc[f, "p01"] or value > limits.loc[f, "p99"]) for value, f in zip(values, features, strict=True)], float)
    common_full = np.zeros(len(features)); common_full[[features.index(f) for f in common_features]] = np.abs(w_common)
    source_full = np.zeros(len(features))
    for feature, weight in zip(source_features, source_weight, strict=True):
        if feature in features:
            source_full[features.index(feature)] = abs(weight)
    source_full /= source_full.sum() + 1e-9
    branch_abs = np.abs(w_branch) / (np.abs(w_branch).sum() + 1e-9)
    weighted_common = float(flags @ common_full)
    weighted_branch = float(flags @ branch_abs)
    weighted_source = float(flags @ source_full)
    return {"oos_feature_count": int(flags.sum()), "weighted_oos_common": weighted_common, "weighted_oos_branch": weighted_branch, "weighted_oos_source_head": weighted_source, "support_confidence": 1 - weighted_common, "branch_confidence": 1 - weighted_branch}
