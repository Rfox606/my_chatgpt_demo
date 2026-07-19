from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from continuous_state_v4.config import FEATURES, ContinuousStateV4Config
from continuous_state_v4.data import add_restart_guard, assert_label_free
from continuous_state_v4.evaluation import guard_pause_check, prefix_causality
from continuous_state_v4.forecast import safe_gate_select
from continuous_state_v4.state_engine import EVIDENCE_NAMES, _run_evidence_tracks, run_target_state


def _config(**changes: object) -> ContinuousStateV4Config:
    base = ContinuousStateV4Config(
        baseline_cycles=100,
        known_stop_interval_cycles=100,
        restart_guard_cycles=15,
        evidence_confirm_cycles=10,
        evidence_reset_cycles=5,
        forecast_horizons_cycles=(20,),
    )
    return replace(base, **changes)


def _stream(end_cycle: int = 11000) -> pd.DataFrame:
    center = np.arange(10.5, end_cycle, 5.0)
    frame: dict[str, object] = {
        "dataset": "Synthetic", "window_id": np.arange(len(center)), "window_index": np.arange(len(center)),
        "start_cycle": center - 9.5, "end_cycle": center + 9.5, "center_cycle": center,
        "baseline_window": (center <= 100).astype(int),
    }
    for number, feature in enumerate(FEATURES):
        frame[feature] = np.sin(center / (25.0 + number)) + .002 * number * center
    return pd.DataFrame(frame)


def _scored(end_cycle: int = 11000):
    config = _config()
    frame = add_restart_guard(_stream(end_cycle), config)
    return frame, run_target_state(frame, "SYNTHETIC", FEATURES, config)[0], config


def test_stage_columns_are_rejected_at_v4_boundaries() -> None:
    with pytest.raises(AssertionError):
        assert_label_free(pd.DataFrame({"stage": [1]}))
    _, states, _ = _scored(1000)
    assert not {"stage", "stage_label", "Stage1to5"}.intersection(states.columns)


def test_future_target_values_do_not_change_the_past() -> None:
    frame, full, config = _scored(1200)
    changed = frame.copy()
    changed.loc[changed.center_cycle > 600, list(FEATURES)] += 1000.0
    replay, _, _ = run_target_state(changed, "SYNTHETIC", FEATURES, config)
    columns = ("D_state", "V20_norm", "V50_norm", "V100_norm", "A_state", "abrupt_cusum", *EVIDENCE_NAMES)
    before = full.loc[full.center_cycle <= 600, columns].to_numpy(float)
    replay_before = replay.loc[replay.center_cycle <= 600, columns].to_numpy(float)
    assert np.allclose(before, replay_before, equal_nan=True)


def test_restart_guard_pauses_every_evidence_track() -> None:
    _, states, _ = _scored(1200)
    result = guard_pause_check(states)
    assert result["status"] == "PASS"


def test_prefix_replay_matches_the_full_causal_run() -> None:
    frame, states, config = _scored()
    result = prefix_causality(frame, states, "SYNTHETIC", FEATURES, config)
    assert result["status"] == "PASS"


def test_four_evidence_tracks_can_activate_without_a_fixed_order() -> None:
    n = 40
    frame = pd.DataFrame({"dataset": "Synthetic", "center_cycle": np.arange(n) * 5.0 + 5.0, "is_restart_guard": np.zeros(n, int)})
    conditions = {name: np.zeros(n, dtype=bool) for name in EVIDENCE_NAMES}
    # Deliberately use the reverse of the old ordered-state intuition.
    conditions["acceleration_evidence"][0:4] = True
    conditions["directed_change_evidence"][5:9] = True
    conditions["low_activity_evidence"][10:14] = True
    abrupt = np.arange(n, dtype=float)
    values, events, _ = _run_evidence_tracks(
        frame, conditions, abrupt,
        {"abrupt_baseline_center": 0.0, "abrupt_baseline_scale": 1.0, "abrupt_score_min": 1.0},
        _config(evidence_confirm_cycles=10), "SYNTHETIC",
    )
    onsets = events.loc[events.event.eq("algorithm_evidence_onset"), "evidence_type"].tolist()
    assert "acceleration_evidence" in onsets
    assert "directed_change_evidence" in onsets
    assert "low_activity_evidence" in onsets
    assert values["low_activity_evidence"].max() == 1


def test_safe_gate_compares_online_to_current_best_static_baseline() -> None:
    static = {"Zero_Delta": (2.0, 30), "Local_Linear": (.5, 30), "Kalman": (.8, 30), "Frozen_Ridge": (1.0, 30)}
    selected, reference, online_used = safe_gate_select(static, (.4, 30), 20)
    assert selected == "Online_RLS"
    assert reference == "Local_Linear"
    assert online_used
    selected, reference, online_used = safe_gate_select(static, (.6, 30), 20)
    assert selected == "Local_Linear"
    assert reference == "Local_Linear"
    assert not online_used
