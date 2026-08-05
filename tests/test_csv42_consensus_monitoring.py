from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from continuous_state_v42.config import FEATURES, ContinuousStateV42Config
from continuous_state_v42.consensus import ConfigurationRecord, consensus_trajectories, detect_change_episodes
from continuous_state_v42.data import add_restart_guard, assert_label_free
from continuous_state_v42.evaluation import guard_pause_check, prefix_causality
from continuous_state_v42.forecast import safe_gate_select
from continuous_state_v42.state_engine import fit_source_support, run_target_state


def _config(**changes: object) -> ContinuousStateV42Config:
    return replace(ContinuousStateV42Config(
        baseline_cycles=100, known_stop_interval_cycles=100, restart_guard_cycles=15,
        evidence_confirm_cycles=10, evidence_reset_cycles=5, low_activity_confirm_cycles=20, low_activity_release_cycles=10,
        forecast_horizons_cycles=(20,), consensus_emit_start_cycles=100,
    ), **changes)


def _stream(end_cycle: int = 11000) -> pd.DataFrame:
    center = np.arange(10.5, end_cycle, 5.0)
    frame: dict[str, object] = {"dataset": "Synthetic", "window_id": np.arange(len(center)), "window_index": np.arange(len(center)), "start_cycle": center - 9.5, "end_cycle": center + 9.5, "center_cycle": center, "baseline_window": (center <= 100).astype(int)}
    for number, feature in enumerate(FEATURES): frame[feature] = np.sin(center / (25.0 + number)) + .002 * number * center
    return pd.DataFrame(frame)


def _scored(end_cycle: int = 11000):
    config = _config(); frame = add_restart_guard(_stream(end_cycle), config); support = fit_source_support(frame, FEATURES)
    states, _, _ = run_target_state(frame, "SYNTHETIC", FEATURES, config, support)
    return frame, states, support, config


def test_calibration_period_has_no_formal_output_and_no_labels() -> None:
    _, states, _, config = _scored(1200)
    assert (states.start_cycle > config.baseline_cycles).all()
    with pytest.raises(AssertionError): assert_label_free(pd.DataFrame({"stage": [1]}))


def test_future_change_cannot_modify_emitted_prefix_and_guard_pauses() -> None:
    frame, full, support, config = _scored(1200)
    changed = frame.copy(); changed.loc[changed.center_cycle > 600, list(FEATURES)] += 1000.0
    replay, _, _ = run_target_state(changed, "SYNTHETIC", FEATURES, config, support)
    columns = ("D_state", "V100_norm", "V500_norm", "V1000_norm", "A_state", "abrupt_cusum")
    assert np.allclose(full.loc[full.end_cycle <= 600, columns].to_numpy(float), replay.loc[replay.end_cycle <= 600, columns].to_numpy(float), equal_nan=True)
    assert guard_pause_check(full)["status"] == "PASS"


def test_prefix_replay_is_identical() -> None:
    frame, states, support, config = _scored()
    assert prefix_causality(frame, states, "SYNTHETIC", FEATURES, config, support)["status"] == "PASS"


def test_consensus_reports_quantiles_mad_and_effective_configuration_count() -> None:
    _, states, _, config = _scored(1200)
    states = states.copy(); states["directed_change_evidence_condition"] = 0
    other = states.copy(); other["D_state"] += 2.0; other["directed_change_score"] = 2.0; other["directed_change_evidence_condition"] = 1
    records = [ConfigurationRecord("a", 100, "mahalanobis", "none", states), ConfigurationRecord("b", 100, "diagonal", "none", other)]
    consensus, support, _ = consensus_trajectories(records, config)
    assert (consensus.effective_configuration_count == 2).all()
    assert {"D_state_q25", "D_state_q50", "D_state_q75", "D_state_mad", "multi_scale_rate_divergence_q50"}.issubset(consensus.columns)
    assert (support.directed_configuration_support == .5).all()


def test_nearby_change_types_merge_into_one_episode() -> None:
    frame, states, _, config = _scored(1200)
    records = [ConfigurationRecord("a", 100, "mahalanobis", "none", states)]
    consensus, _, long = consensus_trajectories(records, config)
    chosen = consensus.index[10:14]
    consensus.loc[chosen, "change_trigger"] = 1; consensus.loc[chosen, "change_configuration_support"] = 1.0; consensus.loc[chosen, "combined_change_score_q50"] = 2.0
    consensus.loc[chosen[:2], "directed_configuration_support"] = 1.0; consensus.loc[chosen[2:], "abrupt_configuration_support"] = 1.0
    episodes = detect_change_episodes(consensus, long, replace(config, episode_min_cycles=5))
    assert len(episodes) >= 1
    assert "directed=" in episodes.evidence_composition.iloc[0] and "abrupt=" in episodes.evidence_composition.iloc[0]


def test_safe_gate_compares_all_static_baselines() -> None:
    static = {"Zero_Delta": (2.0, 30), "Local_Linear": (.5, 30), "Kalman": (.8, 30), "Frozen_Ridge": (1.0, 30)}
    assert safe_gate_select(static, (.4, 30), 20) == ("Online_RLS", "Local_Linear", True)
    assert safe_gate_select(static, (.6, 30), 20) == ("Local_Linear", "Local_Linear", False)
