from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from continuous_state_v44.analysis import input_provenance
from continuous_state_v44.config import BASE_FEATURES, EXTENDED_RY_FEATURES, ContinuousStateV44Config
from continuous_state_v44.data import assert_label_free
from continuous_state_v44.state_engine import feature_subset, run_target_state, state_columns


def _frame(rows: int = 70) -> pd.DataFrame:
    centre = np.arange(rows, dtype=float) * 100.0 + 50.0
    data: dict[str, object] = {
        "dataset": ["Exp1"] * rows, "window_id": np.arange(rows), "window_index": np.arange(rows),
        "start_cycle_effective": centre - 49.0, "end_cycle_effective": centre + 49.0, "center_cycle_effective": centre,
        "start_cycle_actual": centre - 49.0, "end_cycle_actual": centre + 49.0, "center_cycle_actual": centre,
        "cycle_effective": centre, "cycle_actual": centre,
    }
    for offset, feature in enumerate(EXTENDED_RY_FEATURES):
        data[feature] = np.sin(centre / (211.0 + offset * 17.0)) + offset * .02
    return pd.DataFrame(data)


def test_baseline_rows_have_no_formal_state_output_and_metadata_is_not_required() -> None:
    config = ContinuousStateV44Config(baseline_cycles=1000, metadata_path="definitely_not_opened_for_state.json")
    states, reference = run_target_state(_frame(), "synthetic", BASE_FEATURES, config)
    assert reference.baseline_count == 10
    assert (states.start_cycle_effective > 1000).all()
    assert not states.empty


def test_suffix_mutation_does_not_change_prefix_states() -> None:
    frame = _frame(); config = ContinuousStateV44Config(baseline_cycles=1000)
    full, _ = run_target_state(frame, "full", BASE_FEATURES, config)
    changed = frame.copy(); changed.loc[changed.center_cycle_effective > 3500, list(BASE_FEATURES)] += 999.0
    replay, _ = run_target_state(changed, "changed", BASE_FEATURES, config)
    merged = full.loc[full.center_cycle_effective <= 3500, ["window_index", *state_columns()]].merge(
        replay.loc[replay.center_cycle_effective <= 3500, ["window_index", *state_columns()]], on="window_index", suffixes=("_a", "_b"))
    assert len(merged) > 0
    for column in state_columns():
        assert np.allclose(merged[f"{column}_a"], merged[f"{column}_b"], atol=1e-12, rtol=0)


def test_stage_and_morphology_are_rejected_at_state_boundary() -> None:
    with pytest.raises(AssertionError):
        assert_label_free(pd.DataFrame({"stage": [1], "Sa": [5.0]}))
    with pytest.raises(AssertionError):
        run_target_state(_frame().assign(Sku=3.2), "leak", BASE_FEATURES, ContinuousStateV44Config())


def test_feature_group_removal_is_prefix_based_and_ry_extension_is_predeclared() -> None:
    no_ry = feature_subset(EXTENDED_RY_FEATURES, "ry")
    assert all(not name.startswith("ry_") for name in no_ry)
    assert {"ry_mean", "ry_rms", "ry_std", "ry_q05", "ry_q95", "ry_corrdist_base"}.issubset(EXTENDED_RY_FEATURES)


def test_existing_input_source_is_traceable_without_stage_read() -> None:
    result = input_provenance(ContinuousStateV44Config())
    assert result["status"] == "PASS"
    assert result["normalized_sensitive_phase"] == [0.45, 0.63]
