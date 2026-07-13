from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV3Config, FEATURES


FORBIDDEN_COLUMNS = frozenset({"stage", "stage_label", "Stage1to5"})
WINDOW_COLUMNS = ("dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle", "baseline_window")


def assert_label_free(frame: pd.DataFrame) -> None:
    leaked = sorted(FORBIDDEN_COLUMNS.intersection(frame.columns))
    if leaked:
        raise AssertionError(f"Forbidden stage column(s) reached a v3 model boundary: {leaked}")


def load_window_table(config: ContinuousStateV3Config) -> pd.DataFrame:
    long = pd.read_csv(config.z_table_path)
    required = set(WINDOW_COLUMNS) | {"feature_name", "z_value"}
    missing = required.difference(long.columns)
    if missing:
        raise ValueError(f"Input z table is missing: {sorted(missing)}")
    wide = (long.pivot_table(index=list(WINDOW_COLUMNS), columns="feature_name", values="z_value", aggfunc="first")
            .reset_index().rename_axis(columns=None))
    missing_features = set(FEATURES).difference(wide.columns)
    if missing_features:
        raise ValueError(f"Input z table lacks candidate features: {sorted(missing_features)}")
    result = wide.loc[:, [*WINDOW_COLUMNS, *FEATURES]].sort_values(["dataset", "center_cycle", "window_index"]).reset_index(drop=True)
    if result.loc[:, list(FEATURES)].isna().any().any():
        raise ValueError("Candidate feature input contains missing values")
    assert_label_free(result)
    return result


def baseline_mask(frame: pd.DataFrame, config: ContinuousStateV3Config) -> np.ndarray:
    assert_label_free(frame)
    mask = ((frame.end_cycle <= config.baseline_cycles) & frame.is_restart_guard.eq(0)).to_numpy(bool)
    if int(mask.sum()) < 10:
        mask = np.zeros(len(frame), dtype=bool)
        available = np.flatnonzero(frame.is_restart_guard.to_numpy(int) == 0)
        mask[available[: min(len(available), 50)]] = True
    if not mask.any():
        raise ValueError("No non-guard target window is available for the fixed baseline")
    return mask


def robust_location_scale(values: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    iqr = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
    return median, np.maximum.reduce([1.4826 * mad, iqr / 1.349, np.full(values.shape[1], eps)])
