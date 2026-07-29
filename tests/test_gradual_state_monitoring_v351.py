from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gradual_state_monitoring_v33.data import MAIN_FEATURES, make_synthetic_sequence
from gradual_state_monitoring_v34.model import GradualStateTCN, set_deterministic_seed
from gradual_state_monitoring_v35.online import run_target_online as run_v35_reference
from gradual_state_monitoring_v351.config import GradualStateMonitoringV351Config
from gradual_state_monitoring_v351.online import _event_evidence, run_target_online, run_v35_hard_and, PendingEvent
from gradual_state_monitoring_v351.pipeline import run_pipeline


def _config() -> GradualStateMonitoringV351Config:
    return GradualStateMonitoringV351Config(random_seeds=(3401,), source_epochs=1, upper_bound_epochs=1)


def _frame(rows: int = 300, dataset: str = "Exp1") -> pd.DataFrame:
    data = make_synthetic_sequence("stable", length=rows, seed=351)
    return data.loc[:, ["dataset", "window_index", "center_cycle", *MAIN_FEATURES]].assign(dataset=dataset)


def test_target_stage_stopping_and_total_length_are_rejected_from_online_path() -> None:
    config = _config(); model = GradualStateTCN(config); frame = _frame(); frame["stage"] = 1; frame["total_length"] = len(frame)
    with pytest.raises(AssertionError): run_target_online(model, frame, config, setting="V351_EventGate_CurrentAdapt")


def test_two_phase_calibration_and_online_start_at_256() -> None:
    config = _config(); run = run_target_online(GradualStateTCN(config), _frame(), config, setting="V351_EventGate_CurrentAdapt")
    assert (run.scores.iloc[:128].phase == "NORMALIZER_CALIBRATION").all()
    assert (run.scores.iloc[128:256].phase == "SCORE_CALIBRATION").all()
    assert (run.scores.iloc[256:].phase == "ONLINE").all()
    assert not run.scores.iloc[:256].adapter_updated.any()
    assert not run.scores.iloc[:256].slow_high.any()


def test_short_target_errors_before_scoring() -> None:
    with pytest.raises(ValueError, match="256"):
        run_target_online(GradualStateTCN(_config()), _frame(255), _config(), setting="V351_EventGate_CurrentAdapt")


def test_complete_lags_are_current_and_causal() -> None:
    config = _config(); run = run_target_online(GradualStateTCN(config), _frame(), config, setting="V351_EventGate_CurrentAdapt")
    audit = run.adaptation[run.adaptation.phase == "ONLINE"]
    assert audit.complete_lags.eq(1).all()
    assert (audit.maximum_online_input_index <= audit.window_index).all()
    assert all(min(map(int, str(value).split("|"))) >= 0 for value in audit.reference_indices)


def test_prefix_causality() -> None:
    config = _config(); set_deterministic_seed(3401); model = GradualStateTCN(config)
    first = run_target_online(model, _frame(280), config, setting="V351_EventGate_WarningFreeze")
    second = run_target_online(model, _frame(300), config, setting="V351_EventGate_WarningFreeze")
    pd.testing.assert_frame_equal(first.scores.reset_index(drop=True), second.scores.iloc[:280].reset_index(drop=True))


def test_score_and_freeze_are_committed_before_update() -> None:
    config = _config(); run = run_target_online(GradualStateTCN(config), _frame(), config, setting="V351_EventGate_CurrentAdapt")
    assert run.adaptation.score_before_update.eq(1).all()
    assert set(run.scores.adapter_updated.unique()).issubset({0, 1})


def test_peak_evidence_can_only_select_slow_high_window() -> None:
    config = _config(); rows = {}
    for index, slow, high in ((260, 2.0, 1), (261, 0.2, 0), (262, 1.5, 1), (263, .1, 0), (264, .2, 0), (265, .2, 0), (266, .2, 0), (267, .2, 0)):
        rows[index] = {"dataset": "Exp1", "setting": "x", "center_cycle": float(index), "slow_final_score": slow, "slow_alarm_threshold": 1.0, "residual_alarm_threshold": 1.0, "residual_persistent": .5, "slow_high": high, "adapter_frozen": 1}
    evidence = _event_evidence(PendingEvent(260, 262, [260, 262]), rows, 267, config)
    assert evidence["peak_cycle"] == 260.0 and evidence["peak_slow_high"] == 1


def test_event_emit_never_precedes_end_plus_five() -> None:
    config = _config(); run = run_target_online(GradualStateTCN(config), _frame(360), config, setting="V351_EventGate_CurrentAdapt")
    if len(run.event_audit): assert (run.event_audit.event_emit_window >= run.event_audit.event_end_window + 5).all()


def test_calibration_does_not_update_adapter_or_normalizer() -> None:
    config = _config(); run = run_target_online(GradualStateTCN(config), _frame(), config, setting="V351_EventGate_WarningFreeze")
    calibration = run.adaptation[run.adaptation.phase != "ONLINE"]
    assert not calibration.adapter_updated.any() and calibration.normalizer_updates.eq(0).all()


def test_all_five_settings_start_from_identical_source_weights_and_a_matches_v35() -> None:
    config = _config(); model = GradualStateTCN(config); frame = _frame(); before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    control = run_v35_hard_and(model, frame, config)
    reference = run_v35_reference(model, frame, config, setting="V35_SlowResidual_OnlineAdaptation")
    for setting in config.v351_settings[1:]: run_target_online(model, frame, config, setting=setting)
    assert all(np.array_equal(before[name].numpy(), value.detach().numpy()) for name, value in model.state_dict().items())
    assert control.scores.setting.eq("V35_HardAND_Current").all()
    np.testing.assert_allclose(control.scores.final_transition_score.to_numpy(float), reference.scores.final_transition_score.to_numpy(float), equal_nan=True)
    assert control.scores.candidate_active.to_list() == reference.scores.candidate_active.to_list()


def test_pipeline_freezes_before_target_stage_evaluation(tmp_path) -> None:
    exp1, exp2 = _frame(280, "Exp1"), _frame(280, "Exp2"); input_path = tmp_path / "input.csv"; pd.concat([exp1, exp2]).to_csv(input_path, index=False)
    segments = [{"dataset": dataset, "stage": stage, "effective_start": start, "effective_end": end, "actual_start": start, "actual_end": end, "note": "test"} for dataset in ("Exp1", "Exp2") for stage, start, end in ((1, 0., 120.), (2, 120., 300.))]
    mapping = tmp_path / "mapping.json"; mapping.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    config = replace(_config(), input_path=str(input_path), cycle_mapping_path=str(mapping), output_dir=str(tmp_path / "out"), source_train_fraction=.95, source_max_training_examples=64)
    run_pipeline(config)
    audit = json.loads((tmp_path / "out" / "v351_online_causality_audit.json").read_text(encoding="utf-8"))
    assert audit["target_labels_read_before_online_freeze"] is False
    assert set(pd.read_csv(tmp_path / "out" / "v351_online_scores.csv").setting.unique()) == set(config.v351_settings)
