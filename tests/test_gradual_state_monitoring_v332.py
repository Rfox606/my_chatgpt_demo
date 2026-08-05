from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gradual_state_monitoring_v33.data import MAIN_FEATURES
from gradual_state_monitoring_v332.config import GradualStateMonitoringV332Config
from gradual_state_monitoring_v332.labels import (
    attach_posthoc_stage, boundary_detection, effective_to_actual, load_existing_stage_mapping,
    ordered_state_mapping, stable_region_mask,
)
from gradual_state_monitoring_v332.online import validate_label_free_online_frame
from gradual_state_monitoring_v332.state import replay_model_prediction_surprise


def _frame(rows: int = 180) -> pd.DataFrame:
    values = np.tile(np.array([.23, .13, -.036, -.08, .091, .242]), (rows, 1))
    values += np.arange(rows)[:, None] * 1e-5
    output = pd.DataFrame(values, columns=MAIN_FEATURES)
    output.insert(0, "dataset", "Exp1")
    output.insert(1, "window_index", np.arange(rows))
    output.insert(2, "center_cycle", np.arange(rows, dtype=float) * 5.0 + 1.0)
    return output


def test_online_boundary_rejects_stage_and_outcomes() -> None:
    config = GradualStateMonitoringV332Config()
    frame = _frame().assign(Stage1to5=1)
    with pytest.raises(AssertionError):
        validate_label_free_online_frame(frame, config)


def test_existing_mapping_is_used_verbatim_and_maps_endpoints() -> None:
    config = GradualStateMonitoringV332Config()
    segments, provenance = load_existing_stage_mapping(config)
    assert provenance["status"] == "EXISTING_REPOSITORY_LABEL_DEFINITION"
    exp1 = segments[(segments.dataset == "Exp1") & (segments.stage == 2)].iloc[0]
    mapped = effective_to_actual([exp1.effective_start, exp1.effective_end], "Exp1", segments)
    assert np.allclose(mapped, [exp1.actual_start, exp1.actual_end])
    online = pd.DataFrame({"dataset": ["Exp1"], "cycle": [float(exp1.effective_start)], "online_state": ["STABLE"], "local_state_id": [1]})
    posthoc = attach_posthoc_stage(online, segments)
    assert int(posthoc.stage.iloc[0]) == 1  # endpoint belongs to the existing lower Stage segment.


def test_buffer_excludes_only_boundary_neighbourhood() -> None:
    values = np.array([700.0, 750.0, 751.0, 1000.0, 1249.0, 1250.0, 1300.0])
    mask = stable_region_mask(values, np.array([1000.0]), 250.0)
    assert mask.tolist() == [True, False, False, False, False, False, True]


def test_boundary_matching_marks_later_alarm_as_duplicate() -> None:
    posthoc = pd.DataFrame({
        "dataset": ["Exp1"] * 5, "input_index": range(5), "cycle": [900., 1000., 1010., 1020., 1030.],
        "actual_cycle": [900., 1000., 1010., 1020., 1030.],
        "online_state": ["STABLE", "TRANSITION", "STABLE", "TRANSITION", "TRANSITION"],
        "local_state_id": [1, 1, 1, 1, 1],
    })
    boundaries = pd.DataFrame({"dataset": ["Exp1"], "from_stage": [1], "to_stage": [2],
                               "boundary_effective_cycle": [1000.], "boundary_actual_cycle": [1000.]})
    metric, log = boundary_detection(posthoc, boundaries, 50.0, model="V331", setting="test")
    assert metric["matched_boundary_count"] == 1
    assert metric["duplicate_alarm_count"] == 1
    assert set(log.match_status) == {"MATCHED", "DUPLICATE"}


def test_ordered_mapping_is_monotone_not_hungarian() -> None:
    posthoc = pd.DataFrame({
        "dataset": ["Exp1"] * 8, "input_index": range(8), "stage": [1, 1, 2, 2, 3, 3, 2, 2],
        "local_state_id": [7, 7, 3, 3, 9, 9, 9, 9],
    })
    mapping, _, _ = ordered_state_mapping(posthoc, model="V331", setting="test")
    mapped = mapping.sort_values("first_appearance_order").mapped_stage.to_numpy(int)
    assert np.all(mapped[1:] >= mapped[:-1])


def test_prediction_error_state_replay_is_prefix_causal() -> None:
    config = GradualStateMonitoringV332Config()
    frame = _frame(180)
    errors = np.linspace(.001, .01, len(frame))
    state, evidence, _ = replay_model_prediction_surprise(frame, errors, config)
    future = _frame(24).copy()
    future.loc[:, "window_index"] += len(frame)
    future.loc[:, "center_cycle"] += float(frame.center_cycle.iloc[-1])
    combined = pd.concat([frame, future], ignore_index=True)
    replay_errors = np.r_[errors, np.linspace(.2, .4, len(future))]
    replay_state, replay_evidence, _ = replay_model_prediction_surprise(combined, replay_errors, config)
    assert state.online_state.tolist() == replay_state.online_state.iloc[:len(state)].tolist()
    assert np.allclose(evidence.change_evidence, replay_evidence.change_evidence.iloc[:len(evidence)], equal_nan=True)
