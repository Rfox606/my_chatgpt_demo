from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV44Config
from continuous_state_v43.time_mapping import add_actual_cycle_columns


FORBIDDEN_COLUMNS = frozenset({
    "stage", "stage_label", "Stage1to5", "Sa", "Sq", "Sz", "Sku", "morphology",
    "morphology_anchor", "physical_validation_result", "actual_total_cycles",
})
WINDOW_COLUMNS = ("dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle", "baseline_window")


def assert_label_free(frame: pd.DataFrame) -> None:
    leaked = sorted(FORBIDDEN_COLUMNS.intersection(frame.columns))
    if leaked: raise AssertionError(f"Forbidden stage or morphology columns reached v4.4 state boundary: {leaked}")


def load_window_table(config: ContinuousStateV44Config, features: tuple[str, ...]) -> pd.DataFrame:
    """Read only feature-z inputs; stage and physical metadata are excluded at file read time."""
    long = pd.read_csv(config.z_table_path, usecols=[*WINDOW_COLUMNS, "feature_name", "z_value"])
    wide = (long.pivot_table(index=list(WINDOW_COLUMNS), columns="feature_name", values="z_value", aggfunc="first")
            .reset_index().rename_axis(columns=None))
    missing = sorted(set(features).difference(wide.columns))
    if missing: raise ValueError(f"Input z table lacks required features: {missing}")
    frame = wide.loc[:, [*WINDOW_COLUMNS, *features]].sort_values(["dataset", "center_cycle", "window_index"]).reset_index(drop=True)
    if frame.loc[:, list(features)].isna().any().any(): raise ValueError("Candidate feature input contains missing values")
    mapped, _ = add_actual_cycle_columns(frame, config)
    # v4.4 deliberately has no stop-deconfounding or restart Guard treatment.
    mapped["is_restart_guard"] = 0
    mapped["crosses_stop_boundary"] = 0
    assert_label_free(mapped)
    return mapped


def baseline_mask(frame: pd.DataFrame, config: ContinuousStateV44Config) -> np.ndarray:
    assert_label_free(frame)
    mask = (frame.end_cycle_effective.to_numpy(float) <= float(config.baseline_cycles))
    if not mask.any(): raise ValueError("No effective-cycle rows are available for frozen baseline")
    return mask


def robust_location_scale(values: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0); mad = np.median(np.abs(values - median), axis=0)
    iqr = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
    return median, np.maximum.reduce([1.4826 * mad, iqr / 1.349, np.full(values.shape[1], eps)])
