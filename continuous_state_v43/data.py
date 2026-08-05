from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV43Config, FEATURES
from .time_mapping import STOP_CYCLES_ACTUAL


FORBIDDEN_COLUMNS = frozenset({
    "stage", "stage_label", "Stage1to5", "Sa", "Sq", "Sz", "Sku",
    "morphology", "morphology_anchor", "physical_validation_result",
})
WINDOW_COLUMNS = ("dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle", "baseline_window")


def assert_label_free(frame: pd.DataFrame) -> None:
    leaked = sorted(FORBIDDEN_COLUMNS.intersection(frame.columns))
    if leaked:
        raise AssertionError(f"Forbidden stage column(s) reached a v4.2 model boundary: {leaked}")


def load_window_table(config: ContinuousStateV43Config) -> pd.DataFrame:
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


def add_restart_guard(frame: pd.DataFrame, config: ContinuousStateV43Config) -> pd.DataFrame:
    """Mark actual-cycle stop boundaries and post-stop Guard intersections only."""
    assert_label_free(frame)
    result = frame.copy()
    if not {"start_cycle_actual", "end_cycle_actual", "center_cycle_actual"}.issubset(result.columns):
        raise ValueError("Actual-cycle mapping must be attached before Guard construction")
    start, end, center = (result[name].to_numpy(float) for name in ("start_cycle_actual", "end_cycle_actual", "center_cycle_actual"))
    crossed, guarded = np.zeros(len(result), bool), np.zeros(len(result), bool)
    nearest = np.full(len(result), np.nan); distance = np.full(len(result), np.nan)
    for dataset, positions in result.groupby("dataset", sort=False).groups.items():
        index = np.asarray(list(positions), dtype=int); boundaries = STOP_CYCLES_ACTUAL[str(dataset)]
        nearest_index = np.abs(center[index, None] - boundaries[None, :]).argmin(axis=1)
        nearest[index] = boundaries[nearest_index]
        distance[index] = np.abs(center[index] - nearest[index])
    for dataset, positions in result.groupby("dataset", sort=False).groups.items():
        index = np.asarray(list(positions), dtype=int)
        for boundary in STOP_CYCLES_ACTUAL[str(dataset)]:
            crossed[index] |= (start[index] <= boundary) & (end[index] >= boundary)
            guarded[index] |= (start[index] <= boundary + config.restart_guard_cycles) & (end[index] >= boundary)
    result["crosses_stop_boundary"] = crossed.astype(int)
    result["is_restart_guard"] = guarded.astype(int)
    result["nearest_stop_boundary_actual"] = nearest
    result["nearest_stop_distance_actual"] = distance
    return result


def baseline_mask(frame: pd.DataFrame, config: ContinuousStateV43Config) -> np.ndarray:
    assert_label_free(frame)
    # The frozen calibration remains effective-cycle based by design.
    effective_end = frame.get("end_cycle_effective", frame.end_cycle)
    mask = ((effective_end <= config.baseline_cycles) & frame.is_restart_guard.eq(0)).to_numpy(bool)
    if not mask.any():
        raise ValueError("No non-guard target window is available within the initial frozen baseline period")
    return mask


def robust_location_scale(values: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    iqr = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
    return median, np.maximum.reduce([1.4826 * mad, iqr / 1.349, np.full(values.shape[1], eps)])
