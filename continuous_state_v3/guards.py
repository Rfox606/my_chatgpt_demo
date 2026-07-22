from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV3Config
from .data import assert_label_free


def add_restart_guard(frame: pd.DataFrame, config: ContinuousStateV3Config) -> pd.DataFrame:
    """Mark complete window/stop-guard interval intersections, never centre-only checks."""
    assert_label_free(frame)
    result = frame.copy()
    start, end, center = (result[name].to_numpy(float) for name in ("start_cycle", "end_cycle", "center_cycle"))
    limit = float(end.max())
    boundaries = np.arange(0, int(np.ceil(limit / config.known_stop_interval_cycles)) * config.known_stop_interval_cycles + config.known_stop_interval_cycles, config.known_stop_interval_cycles)
    crossed, guarded = np.zeros(len(result), bool), np.zeros(len(result), bool)
    for boundary in boundaries:
        crossed |= (start <= boundary) & (end >= boundary)
        guarded |= (start <= boundary + config.restart_guard_cycles) & (end >= boundary)
    result["crosses_stop_boundary"] = crossed.astype(int)
    result["is_restart_guard"] = guarded.astype(int)
    result["nearest_stop_boundary"] = np.round(center / config.known_stop_interval_cycles) * config.known_stop_interval_cycles
    return result
