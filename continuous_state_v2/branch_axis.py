from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV2Config
from .data import assert_label_free


def build_branch_axis(exp1: pd.DataFrame, exp2: pd.DataFrame, features: tuple[str, ...], common_features: tuple[str, ...], w_common: np.ndarray, config: ContinuousStateV2Config) -> tuple[np.ndarray, pd.DataFrame]:
    assert_label_free(exp1); assert_label_free(exp2)
    def delta(frame: pd.DataFrame) -> np.ndarray:
        usable = frame.loc[frame.is_restart_guard.eq(0)].sort_values("center_cycle")
        n = len(usable)
        middle = usable.iloc[int(n * config.middle_fraction[0]):int(n * config.middle_fraction[1])]
        terminal = usable.iloc[int(n * config.terminal_fraction[0]):int(n * config.terminal_fraction[1])]
        return terminal.loc[:, list(features)].median().to_numpy(float) - middle.loc[:, list(features)].median().to_numpy(float)
    d1, d2 = delta(exp1), delta(exp2)
    q = np.zeros(len(features), float)
    if len(common_features):
        ix = [features.index(f) for f in common_features]
        q[ix] = w_common
        q /= np.linalg.norm(q) + config.eps
    residual1 = d1 - q * np.dot(d1, q)
    residual2 = d2 - q * np.dot(d2, q)
    raw = residual2 - residual1
    weight = raw / (np.linalg.norm(raw) + config.eps)
    table = pd.DataFrame({"feature_name": features, "delta_terminal_exp1": d1, "delta_terminal_exp2": d2, "residual_exp1": residual1, "residual_exp2": residual2, "w_branch_raw": raw, "w_branch": weight})
    return weight, table
