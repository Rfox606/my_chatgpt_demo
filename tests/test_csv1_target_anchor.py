from __future__ import annotations

import numpy as np
import pandas as pd

from continuous_state_v1.config import ContinuousStateV1Config
from continuous_state_v1.target_anchor import score_awr


def test_baseline_anchor_centres_relative_awr_at_zero() -> None:
    config = ContinuousStateV1Config()
    n = 50
    values = np.arange(n, dtype=float)
    frame = pd.DataFrame(
        {
            "window_id": np.arange(n), "window_index": np.arange(n), "start_cycle": values * 5,
            "end_cycle": values * 5 + 1, "center_cycle": values * 5 + 0.5, "baseline_window": 1,
            "is_restart_guard": 0,
            **{feature: values for feature in config.stable_plus_features},
        }
    )
    scored, anchor = score_awr(frame, np.full(10, 0.1), 1.0, config)
    assert abs(np.median(scored.loc[anchor.baseline_mask, "AWR_rel"])) < 1e-10
