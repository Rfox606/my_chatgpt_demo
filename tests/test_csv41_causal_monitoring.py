from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from continuous_state_v41.config import FEATURES, ContinuousStateV41Config
from continuous_state_v41.data import add_restart_guard, assert_label_free
from continuous_state_v41.evaluation import guard_pause_check, prefix_causality
from continuous_state_v41.forecast import safe_gate_select
from continuous_state_v41.state_engine import EVIDENCE_NAMES, _run_evidence_tracks, fit_source_support, run_target_state


def _config(**changes: object) -> ContinuousStateV41Config:
    return replace(ContinuousStateV41Config(
        baseline_cycles=100, known_stop_interval_cycles=100, restart_guard_cycles=15,
        evidence_confirm_cycles=10, evidence_reset_cycles=5, low_activity_confirm_cycles=20,
        low_activity_release_cycles=10, forecast_horizons_cycles=(20,),
    ), **changes)


def _stream(end_cycle: int = 11000) -> pd.DataFrame:
    center = np.arange(10.5, end_cycle, 5.0)
    frame: dict[str, object] = {
        "dataset": "Synthetic", "window_id": np.arange(len(center)), "window_index": np.arange(len(center)),
        "start_cycle": center - 9.5, "end_cycle": center + 9.5, "center_cycle": center, "baseline_window": (center <= 100).astype(int),
    }
    for number, feature in enumerate(FEATURES): frame[feature] = np.sin(center / (25.0 + number)) + .002 * number * center
    return pd.DataFrame(frame)


def _scored(end_cycle: int = 11000):
    config = _config(); frame = add_restart_guard(_stream(end_cycle), config); support = fit_source_support(frame, FEATURES)
    states, events, reference = run_target_state(frame, "SYNTHETIC", FEATURES, config, support)
    return frame, states, events, reference, support, config


def test_calibration_period_has_no_formal_output() -> None:
    _, states, events, _, _, config = _scored(1200)
    assert (states.start_cycle > config.baseline_cycles).all()
    assert events.empty or (events.cycle > config.baseline_cycles).all()


def test_late_calibration_change_updates_frozen_parameters_without_earlier_output() -> None:
    frame, states, _, reference, support, config = _scored(1200)
    changed = frame.copy()
    changed.loc[(changed.center_cycle >= 55) & (changed.center_cycle < 100), list(FEATURES)] += 5.0
    replay, _, changed_reference = run_target_state(changed, "SYNTHETIC", FEATURES, config, support)
    assert not np.allclose(reference.groups["rs"].location, changed_reference.groups["rs"].location)
    assert (states.start_cycle > config.baseline_cycles).all() and (replay.start_cycle > config.baseline_cycles).all()


def test_no_labels_or_future_target_values_reach_online_output() -> None:
    with pytest.raises(AssertionError): assert_label_free(pd.DataFrame({"stage": [1]}))
    frame, full, _, _, support, config = _scored(1200)
    changed = frame.copy(); changed.loc[changed.center_cycle > 600, list(FEATURES)] += 1000.0
    replay, _, _ = run_target_state(changed, "SYNTHETIC", FEATURES, config, support)
    columns = ("D_state", "V100_norm", "V500_norm", "V1000_norm", "A_state", "abrupt_cusum", *EVIDENCE_NAMES)
    original = full.loc[full.end_cycle <= 600, columns].to_numpy(float)
    changed_prefix = replay.loc[replay.end_cycle <= 600, columns].to_numpy(float)
    assert np.allclose(original, changed_prefix, equal_nan=True)
    assert not {"stage", "stage_label", "Stage1to5"}.intersection(full.columns)


def test_guard_pauses_and_prefix_replay_is_identical() -> None:
    frame, states, _, _, support, config = _scored()
    assert guard_pause_check(states)["status"] == "PASS"
    assert prefix_causality(frame, states, "SYNTHETIC", FEATURES, config, support)["status"] == "PASS"


def test_evidence_tracks_do_not_require_an_order() -> None:
    n = 40; frame = pd.DataFrame({"dataset": "Synthetic", "center_cycle": np.arange(n) * 5.0 + 105.0, "is_restart_guard": np.zeros(n, int)})
    conditions = {name: np.zeros(n, dtype=bool) for name in EVIDENCE_NAMES}
    conditions["acceleration_evidence"][0:4] = True; conditions["directed_change_evidence"][5:9] = True; conditions["low_activity_evidence"][10:15] = True
    scores = {name: np.ones(n) for name in EVIDENCE_NAMES}
    _, events, _ = _run_evidence_tracks(frame, conditions, np.arange(n, dtype=float), {"residual_center": 0.0, "residual_scale": 1.0, "residual_score_min": 1.0}, scores, _config(evidence_confirm_cycles=10), "SYNTHETIC")
    onsets = events.loc[events.event.eq("algorithm_evidence_onset"), "evidence_type"].tolist()
    assert {"acceleration_evidence", "directed_change_evidence", "low_activity_evidence"}.issubset(onsets)


def test_safe_gate_uses_current_best_static_comparator() -> None:
    static = {"Zero_Delta": (2.0, 30), "Local_Linear": (.5, 30), "Kalman": (.8, 30), "Frozen_Ridge": (1.0, 30)}
    selected, reference, online = safe_gate_select(static, (.4, 30), 20)
    assert (selected, reference, online) == ("Online_RLS", "Local_Linear", True)
    selected, reference, online = safe_gate_select(static, (.6, 30), 20)
    assert (selected, reference, online) == ("Local_Linear", "Local_Linear", False)
