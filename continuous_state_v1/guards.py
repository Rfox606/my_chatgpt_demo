from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV1Config
from .data import assert_label_free


def add_restart_guard(frame: pd.DataFrame, config: ContinuousStateV1Config) -> pd.DataFrame:
    """Flag windows crossing a known stop boundary or immediately following one."""
    assert_label_free(frame)
    result = frame.copy()
    start = result["start_cycle"].to_numpy(dtype=float)
    end = result["end_cycle"].to_numpy(dtype=float)
    center = result["center_cycle"].to_numpy(dtype=float)
    interval = float(config.known_stop_interval_cycles)
    crossed = np.floor(start / interval) != np.floor(end / interval)
    prior_boundary = np.floor(center / interval) * interval
    since_prior_boundary = center - prior_boundary
    follows_stop = (since_prior_boundary >= 0.0) & (
        since_prior_boundary <= float(config.restart_guard_cycles)
    )
    result["is_restart_guard"] = (crossed | follows_stop).astype(int)
    result["nearest_stop_boundary"] = prior_boundary
    result["cycles_since_stop_boundary"] = since_prior_boundary
    return result
