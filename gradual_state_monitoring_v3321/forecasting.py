from __future__ import annotations

"""Corrected V3.3.2 forecasting facade.

The legacy child-worker module has a frozen off-by-horizon origin generator.  This
facade leaves it untouched and runs its predictor implementation in-process with
the corrected generator for the isolated V3.3.2.1 output directory.
"""

import numpy as np
import pandas as pd

from gradual_state_monitoring_v332 import forecasting as legacy
from gradual_state_monitoring_v332.online import validate_label_free_online_frame

from .config import GradualStateMonitoringV3321Config


def legal_origins(length: int, history: int, maximum_horizon: int, train_end: int) -> np.ndarray:
    """Origins satisfy the strict requirement origin + maximum_horizon < train_end."""
    upper_exclusive = min(length - maximum_horizon, train_end - maximum_horizon)
    origins = np.arange(history - 1, max(history - 1, upper_exclusive), dtype=int)
    if len(origins) and not bool(np.all(origins + maximum_horizon < train_end)):
        raise AssertionError("V3.3.2.1 training target escaped the training interval")
    return origins


def _in_process_setting(setting: str, source: pd.DataFrame, target: pd.DataFrame, train_end: int,
                        prediction_start: int, config: GradualStateMonitoringV3321Config) -> legacy.ForecastRun:
    return legacy._forecast_one_setting(setting, source, target, train_end, prediction_start, config)


def run_label_free_forecasts(frame_by_dataset: dict[str, pd.DataFrame], config: GradualStateMonitoringV3321Config) -> list[legacy.ForecastRun]:
    exp1 = validate_label_free_online_frame(frame_by_dataset["Exp1"], config)
    exp2 = validate_label_free_online_frame(frame_by_dataset["Exp2"], config)
    # Patch only the legacy module object during this call; the committed V3.3.2
    # source file and its existing outputs remain byte-for-byte untouched.
    original_origins = legacy._origins
    original_isolated = legacy._forecast_setting_isolated
    legacy._origins = legal_origins
    legacy._forecast_setting_isolated = _in_process_setting
    try:
        runs: list[legacy.ForecastRun] = []
        for name, frame in (("Exp1", exp1), ("Exp2", exp2)):
            train_end = int(len(frame) * config.within_train_fraction)
            runs.append(legacy._forecast_setting_isolated(f"within_{name}", frame, frame, train_end, train_end, config))
        runs.append(legacy._forecast_setting_isolated("Exp1_pretrain_to_Exp2", exp1, exp2, len(exp1), config.scaler_calibration_windows, config))
        return runs
    finally:
        legacy._origins = original_origins
        legacy._forecast_setting_isolated = original_isolated
