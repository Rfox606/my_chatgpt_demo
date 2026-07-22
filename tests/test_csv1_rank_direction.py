from __future__ import annotations

import numpy as np
import pandas as pd

from continuous_state_v1.config import ContinuousStateV1Config
from continuous_state_v1.pair_sampling import split_source_windows
from continuous_state_v1.rank_head import fit_final_rank_model, select_rank_C


def test_rank_head_learns_noisy_long_term_direction_with_local_declines() -> None:
    config = ContinuousStateV1Config(max_pairs_per_gap_bin=400)
    rng = np.random.default_rng(72)
    n = 160
    centers = np.arange(n, dtype=float) * 100 + 10
    trend = centers / centers.max() - 0.35 * np.exp(-((centers - 9000) / 900) ** 2)
    frame = pd.DataFrame(
        {
            "window_id": np.arange(n), "window_index": np.arange(n), "start_cycle": centers - 5,
            "end_cycle": centers + 5, "center_cycle": centers, "baseline_window": 0,
            "is_restart_guard": 0,
            **{
                feature: (trend if index < 4 else 0.25 * trend) + rng.normal(0, 0.05, n)
                for index, feature in enumerate(config.stable_plus_features)
            },
        }
    )
    train, validation, _ = split_source_windows(frame, config)
    C, _, _, validation_pairs = select_rank_C(train, validation, config)
    model, _, metrics = fit_final_rank_model(frame, validation_pairs, C, config)
    assert metrics["source_pair_auc"] > 0.80
    assert model.normalized_weight[0] > 0
