from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV2Config
from .data import assert_label_free


def add_restart_guard(frame: pd.DataFrame, guard_cycles: int, config: ContinuousStateV2Config) -> pd.DataFrame:
    """Guard any window whose complete interval intersects a stop or post-stop region."""
    assert_label_free(frame)
    result = frame.copy()
    start = result.start_cycle.to_numpy(float)
    end = result.end_cycle.to_numpy(float)
    center = result.center_cycle.to_numpy(float)
    interval = config.known_stop_interval_cycles
    max_cycle = max(float(end.max()), float(center.max()))
    boundaries = np.arange(0, int(np.ceil(max_cycle / interval)) * interval + interval, interval, dtype=float)
    crossed = np.zeros(len(result), dtype=bool)
    intersects = np.zeros(len(result), dtype=bool)
    for boundary in boundaries:
        crossed |= (start <= boundary) & (end >= boundary)
        intersects |= (start <= boundary + guard_cycles) & (end >= boundary)
    nearest = np.round(center / interval) * interval
    result["crosses_stop_boundary"] = crossed.astype(int)
    result["intersects_post_stop_guard"] = intersects.astype(int)
    result["is_restart_guard"] = intersects.astype(int)
    result["nearest_stop_boundary"] = nearest
    result["cycles_since_stop_boundary"] = center - np.floor(center / interval) * interval
    result["restart_guard_cycles"] = int(guard_cycles)
    return result


def guard_sensitivity_frames(frame: pd.DataFrame, config: ContinuousStateV2Config) -> dict[int, pd.DataFrame]:
    return {cycles: add_restart_guard(frame, cycles, config) for cycles in config.restart_guard_cycles_grid}
