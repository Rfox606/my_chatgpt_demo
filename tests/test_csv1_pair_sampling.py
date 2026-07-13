from __future__ import annotations

import numpy as np
import pandas as pd

from continuous_state_v1.config import ContinuousStateV1Config
from continuous_state_v1.pair_sampling import build_pair_batch, sample_temporal_pairs, split_source_windows


def _frame(config: ContinuousStateV1Config) -> pd.DataFrame:
    n = 150
    centers = np.arange(n, dtype=float) * 100.0 + 10.0
    return pd.DataFrame(
        {
            "dataset": "synthetic", "window_id": np.arange(n), "window_index": np.arange(n),
            "start_cycle": centers - 5.0, "end_cycle": centers + 5.0, "center_cycle": centers,
            "baseline_window": 0, "is_restart_guard": np.where(np.arange(n) % 17 == 0, 1, 0),
            **{feature: np.linspace(0, 1, n) for feature in config.stable_plus_features},
        }
    )


def test_pair_sampling_honours_gaps_guards_and_mirroring() -> None:
    config = ContinuousStateV1Config(max_pairs_per_gap_bin=70)
    frame = _frame(config)
    pairs = sample_temporal_pairs(frame, config)
    assert not pairs.empty
    assert (pairs.cycle_gap >= 500).all()
    assert (pairs.later_center_cycle > pairs.earlier_center_cycle).all()
    batch = build_pair_batch(frame, config)
    assert batch.delta_x.shape[0] == 2 * batch.pair_count
    assert batch.labels.sum() == batch.pair_count
    assert np.allclose(batch.delta_x[: batch.pair_count], -batch.delta_x[batch.pair_count :])


def test_source_split_has_an_embargo_gap() -> None:
    config = ContinuousStateV1Config(source_gap_windows=20)
    train, validation, gap = split_source_windows(_frame(config), config)
    assert len(gap) == 20
    assert train.center_cycle.max() < validation.center_cycle.min()
    assert set(train.window_id).isdisjoint(set(validation.window_id))
