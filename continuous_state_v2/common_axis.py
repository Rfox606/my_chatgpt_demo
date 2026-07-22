from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV2Config


def build_common_axis(stability: pd.DataFrame, retained_features: set[str], config: ContinuousStateV2Config) -> tuple[np.ndarray, tuple[str, ...], pd.DataFrame, str]:
    exp1 = stability.loc[stability.direction_id.eq("Exp1_source")].set_index("feature_name")
    exp2 = stability.loc[stability.direction_id.eq("Exp2_source")].set_index("feature_name")
    features = sorted(set(exp1.index).intersection(exp2.index).intersection(retained_features))
    rows = []
    kept: list[str] = []
    raw: list[float] = []
    for feature in features:
        one, two = exp1.loc[feature], exp2.loc[feature]
        stable = one.sign_stability >= config.common_sign_stability_min and two.sign_stability >= config.common_sign_stability_min
        same_sign = np.sign(one.median_weight) == np.sign(two.median_weight) and np.sign(one.median_weight) != 0
        substantial = max(abs(one.median_weight), abs(two.median_weight)) >= config.common_min_median_abs_weight
        accepted = bool(stable and same_sign and substantial)
        weight = float(np.sign(one.median_weight) * np.sqrt(abs(one.median_weight) * abs(two.median_weight))) if accepted else 0.0
        rows.append({"feature_name": feature, "median_weight_exp1": one.median_weight, "median_weight_exp2": two.median_weight, "sign_stability_exp1": one.sign_stability, "sign_stability_exp2": two.sign_stability, "common_weight_raw": weight, "kept_common": int(accepted), "drop_reason": "KEPT" if accepted else "FAILED_STABILITY_SIGN_OR_MAGNITUDE"})
        if accepted:
            kept.append(feature); raw.append(weight)
    if not kept:
        return np.array([]), tuple(), pd.DataFrame(rows), "FAIL_NO_SHARED_DIRECTION"
    weight = np.asarray(raw, float); weight /= np.abs(weight).sum()
    table = pd.DataFrame(rows)
    table["w_common"] = 0.0
    table.loc[table.feature_name.isin(kept), "w_common"] = weight
    return weight, tuple(kept), table, "LOW_DIMENSION_SUPPORT" if len(kept) == 1 else "SUPPORTED"


def causal_rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=1).median().to_numpy(float)
