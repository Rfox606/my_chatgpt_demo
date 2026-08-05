from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from continuous_state_v43.config import FEATURES, ContinuousStateV43Config
from continuous_state_v43.consensus import ConfigurationRecord, consensus_trajectories, detect_change_episodes
from continuous_state_v43.data import add_restart_guard, assert_label_free
from continuous_state_v43.deconfounding import stop_deconfounding
from continuous_state_v43.evaluation import guard_pause_check, prefix_causality
from continuous_state_v43.forecast import safe_gate_select
from continuous_state_v43.state_engine import fit_source_support, run_target_state
from continuous_state_v43.time_mapping import map_effective_to_actual


def _config(**changes: object) -> ContinuousStateV43Config:
    return replace(ContinuousStateV43Config(
        baseline_cycles=100, restart_guard_cycles=15, evidence_confirm_cycles=10, evidence_reset_cycles=5,
        low_activity_confirm_cycles=20, low_activity_release_cycles=10, forecast_horizons_cycles=(20,),
        consensus_emit_start_cycles=100, episode_min_cycles=5, episode_split_min_actual_cycles=100,
        episode_split_persistence_cycles=20,
    ), **changes)


def _stream(end_cycle: int = 1200) -> pd.DataFrame:
    center = np.arange(10.5, end_cycle, 5.0)
    frame: dict[str, object] = {
        "dataset": "Synthetic", "window_id": np.arange(len(center)), "window_index": np.arange(len(center)),
        "start_cycle": center - 9.5, "end_cycle": center + 9.5, "center_cycle": center, "baseline_window": (center <= 100).astype(int),
        "start_cycle_effective": center - 9.5, "end_cycle_effective": center + 9.5, "center_cycle_effective": center,
        "start_cycle_actual": center - 9.5, "end_cycle_actual": center + 9.5, "center_cycle_actual": center,
        "cycle_effective": center, "cycle_actual": center, "is_restart_guard": np.zeros(len(center), dtype=int), "crosses_stop_boundary": np.zeros(len(center), dtype=int),
    }
    for number, feature in enumerate(FEATURES): frame[feature] = np.sin(center / (25.0 + number)) + .002 * number * center
    return pd.DataFrame(frame)


def _scored(end_cycle: int = 1200):
    config = _config(); frame = _stream(end_cycle); support = fit_source_support(frame, FEATURES)
    states, _, _ = run_target_state(frame, "SYNTHETIC", FEATURES, config, support)
    return frame, states, support, config


def test_mapping_is_monotone_and_configured_endpoints_are_exact() -> None:
    config = _config()
    assert np.allclose(map_effective_to_actual("Exp1", np.asarray([1., 7575., 45590.]), config), [1., 8000., 53000.])
    assert np.allclose(map_effective_to_actual("Exp2", np.asarray([1., 11705., 14100.]), config), [501., 20000., 24000.])
    values = map_effective_to_actual("Exp2", np.arange(1., 14101., 10.), config)
    assert np.all(np.diff(values) >= 0)


def test_actual_cycle_guard_uses_actual_stops_not_effective_interval() -> None:
    frame = pd.DataFrame({
        "dataset": ["Exp1", "Exp1", "Exp1"], "start_cycle": [1000., 7565., 7600.], "end_cycle": [1019., 7585., 7620.], "center_cycle": [1009.5, 7575., 7610.],
        "start_cycle_effective": [1000., 7565., 7600.], "end_cycle_effective": [1019., 7585., 7620.], "center_cycle_effective": [1009.5, 7575., 7610.],
        "start_cycle_actual": [12000., 7988., 8010.], "end_cycle_actual": [12020., 8012., 8030.], "center_cycle_actual": [12010., 8000., 8020.],
    })
    guarded = add_restart_guard(frame, _config())
    assert guarded.is_restart_guard.tolist() == [0, 1, 1]
    assert guarded.crosses_stop_boundary.tolist() == [0, 1, 0]


def test_no_stage_or_morphology_leakage_and_traceable_time_fields() -> None:
    _, states, _, config = _scored()
    assert (states.start_cycle_effective > config.baseline_cycles).all()
    assert {"cycle_effective", "cycle_actual", "start_cycle_actual", "end_cycle_actual", "center_cycle_actual"}.issubset(states.columns)
    with pytest.raises(AssertionError): assert_label_free(pd.DataFrame({"stage": [1]}))
    with pytest.raises(AssertionError): assert_label_free(pd.DataFrame({"Sa": [5.32]}))


def test_future_change_and_prefix_replay_are_causal_and_guard_pauses() -> None:
    frame, full, support, config = _scored()
    changed = frame.copy(); changed.loc[changed.center_cycle_effective > 600, list(FEATURES)] += 1000.0
    replay, _, _ = run_target_state(changed, "SYNTHETIC", FEATURES, config, support)
    columns = ("D_state", "V100_norm", "V500_norm", "V1000_norm", "A_state", "abrupt_cusum")
    assert np.allclose(full.loc[full.end_cycle_effective <= 600, columns].to_numpy(float), replay.loc[replay.end_cycle_effective <= 600, columns].to_numpy(float), equal_nan=True)
    assert prefix_causality(frame, full, "SYNTHETIC", FEATURES, config, support)["status"] == "PASS"
    guarded = full.copy(); guarded.loc[1:3, "is_restart_guard"] = 1
    # The diagnostic verifies tracks are unchanged across any actual Guard run in a scored trajectory.
    guarded.loc[1:3, "evidence_increment_cycles"] = 0.0
    for column in [name for name in guarded if name.endswith("_run_cycles") or name.endswith("_false_cycles") or name.endswith("_evidence")]: guarded.loc[1:3, column] = guarded.loc[0, column]
    assert guard_pause_check(guarded)["status"] == "PASS"


def _episode_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = np.arange(7600., 8401., 20.); effective = actual - 400.
    frame = pd.DataFrame({
        "protocol_id": "B_Exp2_to_Exp1", "dataset": "Exp1", "window_id": np.arange(len(actual)), "window_index": np.arange(len(actual)),
        "start_cycle_effective": effective - 10, "end_cycle_effective": effective + 10, "center_cycle_effective": effective,
        "start_cycle_actual": actual - 10, "end_cycle_actual": actual + 10, "center_cycle_actual": actual, "cycle_effective": effective, "cycle_actual": actual,
        "change_trigger": 1, "change_configuration_support": 1.0, "combined_change_score_q50": 2.0,
        "directed_configuration_support": .4, "rate_divergence_configuration_support": .3, "abrupt_configuration_support": .3,
        "guard_configuration_fraction": 0.0, "stop_boundary_configuration_fraction": 0.0,
    })
    frame.loc[frame.index[len(frame)//2], ["combined_change_score_q50", "change_configuration_support"]] = [.1, .1]
    long = frame.copy(); long["configuration_id"] = "cfg"; long["configuration_combined_change_score"] = long.combined_change_score_q50
    return frame, long


def test_fixed_rule_episode_split_and_actual_stop_exclusion() -> None:
    consensus, long = _episode_inputs(); config = _config()
    episodes = detect_change_episodes(consensus, long, config)
    assert len(episodes) >= 2
    table, variants = stop_deconfounding(consensus, long, episodes, config)
    assert {100, 200}.issubset(variants)
    assert {"interval_iou_actual", "peak_shift_actual", "original_peak_nearest_stop_distance_actual"}.issubset(table.columns)


def test_safe_gate_compares_all_static_baselines() -> None:
    static = {"Zero_Delta": (2.0, 30), "Local_Linear": (.5, 30), "Kalman": (.8, 30), "Frozen_Ridge": (1.0, 30)}
    assert safe_gate_select(static, (.4, 30), 20) == ("Online_RLS", "Local_Linear", True)
    assert safe_gate_select(static, (.6, 30), 20) == ("Local_Linear", "Local_Linear", False)
