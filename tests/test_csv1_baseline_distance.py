from __future__ import annotations

import numpy as np
import pandas as pd

from continuous_state_v1.baseline_distance import fit_baseline_distance, score_baseline_distance
from continuous_state_v1.config import ContinuousStateV1Config


def test_baseline_distance_increases_for_shifted_features() -> None:
    config = ContinuousStateV1Config()
    rng = np.random.default_rng(1)
    baseline = rng.normal(0, 0.4, size=(50, 10))
    shifted = rng.normal(2.0, 0.4, size=(50, 10))
    values = np.vstack([baseline, shifted])
    frame = pd.DataFrame(
        {
            "window_id": np.arange(100), "window_index": np.arange(100), "start_cycle": np.arange(100) * 10,
            "end_cycle": np.r_[np.full(50, 400.0), np.full(50, 1200.0)], "center_cycle": np.arange(100) * 10,
            "baseline_window": 0, "is_restart_guard": 0,
            **{feature: values[:, index] for index, feature in enumerate(config.stable_plus_features)},
        }
    )
    model = fit_baseline_distance(frame, config)
    scored = score_baseline_distance(frame, model, config)
    assert scored.BD.iloc[50:].median() > scored.BD.iloc[:50].median()
