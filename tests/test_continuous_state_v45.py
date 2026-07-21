from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from continuous_state_v45.config import ContinuousStateV45Config
from continuous_state_v45.raw_features import SensitiveCycleWaves, add_baseline_corrdist, direct_window_features
from continuous_state_v45.state_engine import assert_label_free, run_state


def _raw_waves() -> SensitiveCycleWaves:
    cycles = np.arange(1, 41)
    phase = np.linspace(0.2, 1.0, 6)
    waves = {
        "rx": np.vstack([phase + cycle * .01 + (cycle > 12) * np.array((0.0, .1, .3, .1, 0.0, -.1)) for cycle in cycles]),
        "ry": np.vstack([phase[::-1] + (cycle > 12) * np.array((.2, 0.0, -.2, .3, .1, -.1)) for cycle in cycles]),
        "rs": np.vstack([np.sqrt((phase + cycle * .01 + (cycle > 12) * np.array((0.0, .1, .3, .1, 0.0, -.1))) ** 2 + (phase[::-1] + (cycle > 12) * np.array((.2, 0.0, -.2, .3, .1, -.1))) ** 2) for cycle in cycles]),
    }
    return SensitiveCycleWaves("Exp1", cycles, waves, 20, (9, 12), 4, "synthetic.csv")


def _state_frame(rows: int = 70) -> pd.DataFrame:
    cycle = np.arange(rows, dtype=float) * 100.0 + 50.0
    frame = pd.DataFrame({"dataset": "Exp1", "window_id": np.arange(rows), "window_index": np.arange(rows),
                          "start_cycle_effective": cycle - 49, "end_cycle_effective": cycle + 49, "center_cycle_effective": cycle,
                          "start_cycle_actual": cycle - 49, "end_cycle_actual": cycle + 49, "center_cycle_actual": cycle, "cycle_effective": cycle, "cycle_actual": cycle})
    for offset, name in enumerate(("rx_mean", "rx_corrdist_base", "ry_p2p", "ry_corrdist_base", "rs_mean", "rs_corrdist_base")):
        frame[name] = np.sin(cycle / (200 + offset * 25)) + offset * .1
    return frame


def test_raw_windows_are_direct_and_label_free() -> None:
    config = ContinuousStateV45Config(window_cycles=4, window_stride_cycles=2)
    table = direct_window_features(_raw_waves(), config)
    assert "stage" not in table.columns and "stage_label" not in table.columns
    assert not any("z_" in column or column == "z_value" for column in table.columns)
    assert "rx_mean" in table.columns and "rs_rms" in table.columns


def test_corrdist_recomputes_for_each_baseline() -> None:
    config = ContinuousStateV45Config(window_cycles=4, window_stride_cycles=2)
    raw = _raw_waves(); direct = direct_window_features(raw, config)
    early = add_baseline_corrdist(raw, direct, 8, config)
    late = add_baseline_corrdist(raw, direct, 20, config)
    assert not np.allclose(early.rx_corrdist_base, late.rx_corrdist_base)
    assert not np.allclose(early.ry_corrdist_base, late.ry_corrdist_base)


def test_state_has_one_internal_normalisation_and_no_labels() -> None:
    config = ContinuousStateV45Config(baseline_cycles=1000)
    features = ("rx_mean", "rx_corrdist_base", "ry_p2p", "ry_corrdist_base", "rs_mean", "rs_corrdist_base")
    states, reference = run_state(_state_frame(), "synthetic", features, config)
    assert reference.baseline_count == 10
    assert (states.start_cycle_effective > 1000).all()
    with pytest.raises(AssertionError):
        run_state(_state_frame().assign(stage=1), "leak", features, config)


def test_suffix_change_is_prefix_causal() -> None:
    config = ContinuousStateV45Config(baseline_cycles=1000); features = ("rx_mean", "rx_corrdist_base", "ry_p2p", "ry_corrdist_base", "rs_mean", "rs_corrdist_base")
    frame = _state_frame(); full, _ = run_state(frame, "full", features, config)
    altered = frame.copy(); altered.loc[altered.center_cycle_effective > 3500, list(features)] += 99.0
    replay, _ = run_state(altered, "replay", features, config)
    metrics = ("D_state", "V1000_norm", "A_state", "state_volatility")
    merged = full.loc[full.center_cycle_effective <= 3500, ["window_index", *metrics]].merge(replay.loc[replay.center_cycle_effective <= 3500, ["window_index", *metrics]], on="window_index", suffixes=("_left", "_right"))
    for metric in metrics:
        assert np.allclose(merged[f"{metric}_left"], merged[f"{metric}_right"], atol=1e-12, rtol=0)


def test_label_boundary_rejects_morphology() -> None:
    with pytest.raises(AssertionError):
        assert_label_free(pd.DataFrame({"Sa": [5.2], "stage_label": ["Stage 1"]}))
