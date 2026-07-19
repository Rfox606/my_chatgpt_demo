from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV42Config, FEATURES


FORBIDDEN_COLUMNS = frozenset({"stage", "stage_label", "Stage1to5"})
WINDOW_COLUMNS = ("dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle", "baseline_window")


def assert_label_free(frame: pd.DataFrame) -> None:
    leaked = sorted(FORBIDDEN_COLUMNS.intersection(frame.columns))
    if leaked:
        raise AssertionError(f"Forbidden stage column(s) reached a v4.2 model boundary: {leaked}")


def load_window_table(config: ContinuousStateV42Config) -> pd.DataFrame:
    """Load only algorithm inputs. Stage columns are intentionally never read."""
    long = pd.read_csv(config.z_table_path, usecols=[*WINDOW_COLUMNS, "feature_name", "z_value"])
    wide = (long.pivot_table(index=list(WINDOW_COLUMNS), columns="feature_name", values="z_value", aggfunc="first")
            .reset_index().rename_axis(columns=None))
    missing = set(FEATURES).difference(wide.columns)
    if missing:
        raise ValueError(f"Input z table lacks candidate features: {sorted(missing)}")
    frame = wide.loc[:, [*WINDOW_COLUMNS, *FEATURES]].sort_values(["dataset", "center_cycle", "window_index"]).reset_index(drop=True)
    if frame.loc[:, list(FEATURES)].isna().any().any():
        raise ValueError("Candidate feature input contains missing values")
    assert_label_free(frame)
    return frame


def add_restart_guard(frame: pd.DataFrame, config: ContinuousStateV42Config) -> pd.DataFrame:
    """Mark complete stop-boundary and post-stop guard intersections."""
    assert_label_free(frame)
    result = frame.copy()
    start, end, center = (result[name].to_numpy(float) for name in ("start_cycle", "end_cycle", "center_cycle"))
    limit = float(end.max())
    interval = config.known_stop_interval_cycles
    boundaries = np.arange(0, int(np.ceil(limit / interval)) * interval + interval, interval)
    crossed, guarded = np.zeros(len(result), bool), np.zeros(len(result), bool)
    for boundary in boundaries:
        crossed |= (start <= boundary) & (end >= boundary)
        guarded |= (start <= boundary + config.restart_guard_cycles) & (end >= boundary)
    result["crosses_stop_boundary"] = crossed.astype(int)
    result["is_restart_guard"] = guarded.astype(int)
    result["nearest_stop_boundary"] = np.round(center / interval) * interval
    return result


def baseline_mask(frame: pd.DataFrame, config: ContinuousStateV42Config) -> np.ndarray:
    assert_label_free(frame)
    mask = ((frame.end_cycle <= config.baseline_cycles) & frame.is_restart_guard.eq(0)).to_numpy(bool)
    if not mask.any():
        raise ValueError("No non-guard target window is available within the initial frozen baseline period")
    return mask


def robust_location_scale(values: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    iqr = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
    return median, np.maximum.reduce([1.4826 * mad, iqr / 1.349, np.full(values.shape[1], eps)])
