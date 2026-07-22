from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV31Config
from .data import assert_label_free


def add_restart_guard(frame: pd.DataFrame, config: ContinuousStateV31Config) -> pd.DataFrame:
    """Mark full restart-guard intersections, rather than a centre-point approximation."""
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
