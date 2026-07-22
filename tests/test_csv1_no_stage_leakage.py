from __future__ import annotations

import pandas as pd
import pytest

from continuous_state_v1.candidate_selection import select_physical_validation_candidates
from continuous_state_v1.config import ContinuousStateV1Config
from continuous_state_v1.data import assert_label_free
from continuous_state_v1.pair_sampling import split_source_windows
from continuous_state_v1 import rank_head
from continuous_state_v1.target_anchor import score_awr


def test_label_columns_are_rejected_at_model_and_scoring_boundaries() -> None:
    config = ContinuousStateV1Config()
    frame = pd.DataFrame(
        {
            "window_id": [1], "window_index": [1], "start_cycle": [1.0], "end_cycle": [20.0],
            "center_cycle": [10.0], "baseline_window": [1], "is_restart_guard": [0], "stage": [1],
            **{feature: [0.0] for feature in config.stable_plus_features},
        }
    )
    with pytest.raises(AssertionError):
        assert_label_free(frame)
    with pytest.raises(AssertionError):
        score_awr(frame, [0.1] * len(config.stable_plus_features), 1.0, config)
    with pytest.raises(AssertionError):
        select_physical_validation_candidates(frame, config)


def test_rank_fit_wrapper_receives_only_pair_differences(monkeypatch: pytest.MonkeyPatch) -> None:
    """The train and C-selection entry point cannot receive historical label fields."""
    config = ContinuousStateV1Config(max_pairs_per_gap_bin=50)
    n = 120
    values = pd.Series(range(n), dtype=float)
    frame = pd.DataFrame(
        {
            "window_id": range(n), "window_index": range(n), "start_cycle": values * 100,
            "end_cycle": values * 100 + 10, "center_cycle": values * 100 + 5,
            "baseline_window": 0, "is_restart_guard": 0,
            **{feature: values / (position + 1) for position, feature in enumerate(config.stable_plus_features)},
        }
    )
    train, validation, _ = split_source_windows(frame, config)
    original_fit = rank_head._fit
    observed: list[tuple[str, ...]] = []

    def wrapped_fit(batch, C, received_config):  # type: ignore[no-untyped-def]
        observed.append(batch.feature_names)
        assert "stage" not in batch.feature_names
        assert "stage_label" not in batch.feature_names
        assert "Stage1to5" not in batch.feature_names
        return original_fit(batch, C, received_config)

    monkeypatch.setattr(rank_head, "_fit", wrapped_fit)
    rank_head.select_rank_C(train, validation, config)
    assert observed
