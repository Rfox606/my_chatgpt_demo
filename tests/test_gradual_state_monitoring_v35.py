from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gradual_state_monitoring_v33.data import MAIN_FEATURES, make_synthetic_sequence
from gradual_state_monitoring_v34.model import GradualStateTCN, set_deterministic_seed
from gradual_state_monitoring_v34.training import robust_location_scale
from gradual_state_monitoring_v341.online import run_target_online as run_v341_reference
from gradual_state_monitoring_v35.config import GradualStateMonitoringV35Config
from gradual_state_monitoring_v35.online import (extract_candidate_events, run_target_online,
                                                  run_v341_baseline)
from gradual_state_monitoring_v35.pipeline import run_pipeline


def _config() -> GradualStateMonitoringV35Config:
    return GradualStateMonitoringV35Config(random_seeds=(3401,), source_epochs=1, upper_bound_epochs=1, adaptation_update_interval=1)


def _frame(rows: int = 170, dataset: str = "Exp1") -> pd.DataFrame:
    value = make_synthetic_sequence("stable", length=rows, seed=350)
    return value.loc[:, ["dataset", "window_index", "center_cycle", *MAIN_FEATURES]].assign(dataset=dataset)


def _normalizer(frame: pd.DataFrame, config: GradualStateMonitoringV35Config):
    return robust_location_scale(frame.loc[:, MAIN_FEATURES].to_numpy(float), 128, config.eps)


def test_target_stage_labels_do_not_enter_online_path() -> None:
    config = _config(); model = GradualStateTCN(config); frame = _frame(); frame["stage"] = 1
    with pytest.raises(AssertionError, match="rejects label"):
        run_target_online(model, frame, config, setting="V35_SlowResidual_OnlineAdaptation")


def test_v35_online_adaptation_is_prefix_causal() -> None:
    config = _config(); set_deterministic_seed(3401); model = GradualStateTCN(config)
    prefix, full = _frame(150), _frame(170)
    first = run_target_online(model, prefix, config, setting="V35_SlowResidual_OnlineAdaptation")
    second = run_target_online(model, full, config, setting="V35_SlowResidual_OnlineAdaptation")
    pd.testing.assert_frame_equal(first.scores.reset_index(drop=True), second.scores.iloc[:len(first.scores)].reset_index(drop=True))


def test_all_references_are_reencoded_with_current_adapter_and_normalizer(monkeypatch) -> None:
    import gradual_state_monitoring_v35.online as online
    config = _config(); model = GradualStateTCN(config); frame = _frame(); calls = []; original = online._encode_many
    def capture(*args, **kwargs):
        calls.append(args[2]); return original(*args, **kwargs)
    monkeypatch.setattr(online, "_encode_many", capture)
    run_target_online(model, frame, config, setting="V35_SlowOnly_OnlineAdaptation")
    assert [128, 112, 108, 64, 0] in calls


def test_current_score_is_committed_before_any_online_update(monkeypatch) -> None:
    import gradual_state_monitoring_v35.online as online
    config = _config(); model = GradualStateTCN(config); frame = _frame(); order = []
    original_row, original_update = online._row, online._adapter_update
    def row_capture(*args, **kwargs):
        if args[3] == "ONLINE": order.append(("score", args[1]))
        return original_row(*args, **kwargs)
    def update_capture(*args, **kwargs):
        order.append(("update", args[3])); return original_update(*args, **kwargs)
    monkeypatch.setattr(online, "_row", row_capture); monkeypatch.setattr(online, "_adapter_update", update_capture)
    run = run_target_online(model, frame, config, setting="V35_SlowOnly_OnlineAdaptation")
    updated = run.scores[run.scores.adapter_updated == 1].window_index.to_list()
    assert updated
    for index in updated:
        assert order.index(("score", index)) < order.index(("update", index))


def test_prediction_residual_is_causal() -> None:
    config = _config(); run = run_target_online(GradualStateTCN(config), _frame(), config, setting="V35_SlowResidual_OnlineAdaptation")
    audit = run.adaptation
    assert (audit.residual_origin_index <= audit.window_index).all()
    assert (audit.maximum_online_input_index <= audit.window_index).all()


def test_calibration_only_never_updates_adapter_or_normalizer() -> None:
    config = _config(); run = run_target_online(GradualStateTCN(config), _frame(), config, setting="V35_SlowResidual_CalibrationOnly")
    assert not run.scores.adapter_updated.any()
    assert not run.adaptation.adapter_updated.any()
    assert (run.adaptation.normalizer_updates == 0).all()


def test_all_six_settings_start_from_identical_source_weights() -> None:
    config = _config(); model = GradualStateTCN(config); frame = _frame(); normalizer = _normalizer(frame, config)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for setting in config.v35_settings:
        if setting.startswith("V341_"):
            run_v341_baseline(model, frame, config, setting=setting, source_normalizer=normalizer)
        else:
            run_target_online(model, frame, config, setting=setting)
    assert all(np.array_equal(before[name].numpy(), value.detach().numpy()) for name, value in model.state_dict().items())


def test_v341_baseline_wrapper_remains_consistent_with_original_v341() -> None:
    config = _config(); model = GradualStateTCN(config); frame = _frame(); normalizer = _normalizer(frame, config)
    original = run_v341_reference(model, frame, config, setting="OnlineAdaptation", source_normalizer=normalizer)
    wrapped = run_v341_baseline(model, frame, config, setting="V341_OnlineAdaptation", source_normalizer=normalizer)
    np.testing.assert_allclose(original.scores.final_transition_score.to_numpy(float), wrapped.scores.final_transition_score.to_numpy(float), equal_nan=True)
    assert original.scores.candidate_active.to_list() == wrapped.scores.candidate_active.to_list()


def test_pipeline_freezes_online_output_before_stage_evaluation(tmp_path) -> None:
    exp1, exp2 = _frame(210, "Exp1"), _frame(210, "Exp2")
    input_path = tmp_path / "input.csv"; pd.concat([exp1, exp2]).to_csv(input_path, index=False)
    segments = [{"dataset": dataset, "stage": stage, "effective_start": start, "effective_end": end, "actual_start": start, "actual_end": end, "note": "test"}
                for dataset in ("Exp1", "Exp2") for stage, start, end in ((1, 0., 100.), (2, 100., 220.))]
    mapping = tmp_path / "mapping.json"; mapping.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    config = replace(_config(), input_path=str(input_path), cycle_mapping_path=str(mapping), output_dir=str(tmp_path / "out"), source_train_fraction=.95, source_max_training_examples=64)
    run_pipeline(config)
    audit = json.loads((tmp_path / "out" / "v35_online_causality_audit.json").read_text(encoding="utf-8"))
    assert audit["target_labels_read_before_online_freeze"] is False
    assert (tmp_path / "out" / "v35_online_scores.csv").exists()
    assert set(pd.read_csv(tmp_path / "out" / "v35_online_scores.csv").setting.unique()) == set(config.v35_settings)
