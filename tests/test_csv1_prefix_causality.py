from __future__ import annotations

import numpy as np
import pandas as pd

from continuous_state_v1.config import ContinuousStateV1Config
from continuous_state_v1.diagnostics import source_support_table
from run_continuous_state_v1 import _score_dataset


def _frame(config: ContinuousStateV1Config, n: int = 150) -> pd.DataFrame:
    centers = np.arange(n, dtype=float) * 20 + 10
    return pd.DataFrame(
        {
            "dataset": "target", "window_id": np.arange(n), "window_index": np.arange(n),
            "start_cycle": centers - 5, "end_cycle": centers + 5, "center_cycle": centers,
            "baseline_window": (centers <= 500).astype(int), "is_restart_guard": 0,
            **{feature: np.sin(centers / 100 + index) + centers / 1000 for index, feature in enumerate(config.stable_plus_features)},
        }
    )


def test_prefix_scores_do_not_depend_on_future_target_windows() -> None:
    config = ContinuousStateV1Config()
    frame = _frame(config)
    weight = np.full(len(config.stable_plus_features), 1 / len(config.stable_plus_features))
    support = source_support_table(frame.assign(dataset="source"), "synthetic", config)
    full, _, _ = _score_dataset(frame, "target", "synthetic", "source", "target", weight, 1.0, config, support)
    prefix = frame.loc[frame.center_cycle <= 1000].copy()
    early, _, _ = _score_dataset(prefix, "target", "synthetic", "source", "target", weight, 1.0, config, support)
    columns = ["AWR_raw", "AWR_rel", "AWR_scaled", "BD", "BD_diag", "oos_fraction"]
    joined = early.merge(full[["window_index", *columns]], on="window_index", suffixes=("_early", "_full"))
    for column in columns:
        assert np.max(np.abs(joined[f"{column}_early"] - joined[f"{column}_full"])) < 1e-10
