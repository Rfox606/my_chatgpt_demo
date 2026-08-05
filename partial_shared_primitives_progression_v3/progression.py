from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PartialSharedPrimitivesConfig
from .data import robust_location_scale


def score_conditioned_continuous_progression(representation: pd.DataFrame, config: PartialSharedPrimitivesConfig) -> pd.DataFrame:
    """State-independent condition-calibrated causal evidence score.

    This module deliberately does not import or accept any state-model object.
    """
    rows: list[dict[str, object]] = []
    for dataset, group in representation.groupby("dataset", sort=True):
        ordered = group.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
        evidence = config.continuous_residual_weight * ordered.forecast_mae.to_numpy(float) + config.continuous_activity_weight * ordered.forecast_activity.to_numpy(float)
        for index, item in ordered.iterrows():
            left = max(0, index - config.continuous_calibration_window)
            history = evidence[left:index]
            if len(history) < config.continuous_minimum_history:
                score = 0.0
                calibration_size = int(len(history))
            else:
                location, scale = robust_location_scale(history[:, None]); score = float((evidence[index] - location[0]) / scale[0]); calibration_size = int(len(history))
            rows.append({
                "dataset": dataset, "window_id": int(item.window_id), "window_index": int(item.window_index), "center_cycle": float(item.center_cycle),
                "continuous_progression_score": score, "continuous_evidence": float(evidence[index]), "continuous_calibration_size": calibration_size,
                "state_model_input": False, "uses_global_time_ranking": False, "uses_absolute_wear": False,
            })
    return pd.DataFrame(rows)

