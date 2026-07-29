from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from gradual_state_monitoring_v33.data import MAIN_FEATURES, make_synthetic_sequence
from gradual_state_monitoring_v34.model import GradualStateTCN, set_deterministic_seed
from gradual_state_monitoring_v34.training import robust_location_scale
from gradual_state_monitoring_v341.config import GradualStateMonitoringV341Config
from gradual_state_monitoring_v341.online import (SourceScoreBaseline, build_source_score_baseline,
                                                   extract_candidate_events, run_target_online)
from gradual_state_monitoring_v341.pipeline import run_pipeline


def _config() -> GradualStateMonitoringV341Config:
    return GradualStateMonitoringV341Config(random_seeds=(3401,), source_epochs=1, upper_bound_epochs=1, adaptation_update_interval=100)


def _frame(rows: int = 165, dataset: str = "Exp1") -> pd.DataFrame:
    value = make_synthetic_sequence("stable", length=rows, seed=341)
    return value.loc[:, ["dataset", "window_index", "center_cycle", *MAIN_FEATURES]].assign(dataset=dataset)


def _baseline(model: GradualStateTCN, frame: pd.DataFrame, config: GradualStateMonitoringV341Config):
    normalizer = robust_location_scale(frame.loc[:, MAIN_FEATURES].to_numpy(float), 128, config.eps)
    return normalizer, build_source_score_baseline(model, frame, normalizer, config)


def test_adaptive_distances_reencode_references_with_current_model(monkeypatch) -> None:
    import gradual_state_monitoring_v341.online as online
    config = _config(); model = GradualStateTCN(config); frame = _frame()
    normalizer, baseline = _baseline(model, frame, config); calls = []; original = online._encode_many
    def capture(*args, **kwargs):
        calls.append(args[2]); return original(*args, **kwargs)
    monkeypatch.setattr(online, "_encode_many", capture)
    run_target_online(model, frame, config, setting="OnlineAdaptation", source_normalizer=normalizer, source_baseline=baseline)
    assert [128, 112, 64, 0] in calls


def test_online_adaptation_is_prefix_causal() -> None:
    config = _config(); set_deterministic_seed(3401); model = GradualStateTCN(config); prefix = _frame(150); full = _frame(165)
    normalizer, baseline = _baseline(model, full, config)
    first = run_target_online(model, prefix, config, setting="OnlineAdaptation", source_normalizer=normalizer, source_baseline=baseline)
    second = run_target_online(model, full, config, setting="OnlineAdaptation", source_normalizer=normalizer, source_baseline=baseline)
    pd.testing.assert_frame_equal(first.scores.reset_index(drop=True), second.scores.iloc[:len(first.scores)].reset_index(drop=True))


def test_direct_transfer_uses_source_score_baseline() -> None:
    config = _config(); model = GradualStateTCN(config); frame = _frame(); normalizer, baseline = _baseline(model, frame, config)
    run = run_target_online(model, frame, config, setting="DirectTransfer", source_normalizer=normalizer, source_baseline=baseline)
    assert (run.scores.candidate_threshold == baseline.candidate_threshold).all()


def test_candidate_requires_three_consecutive_high_windows() -> None:
    config = _config(); scores = pd.DataFrame({"dataset": ["Exp1"] * 4, "setting": ["x"] * 4, "phase": ["ONLINE"] * 4, "window_index": range(4), "center_cycle": range(4), "final_transition_score": [2., 2., 0., 2.], "candidate_threshold": [1.] * 4, "adapter_frozen": [0] * 4})
    assert extract_candidate_events(scores, config).empty


def test_event_merge_occurs_after_persistence_filter() -> None:
    config = replace(_config(), candidate_merge_gap_windows=20); high = [2, 2, 0, 0, 2, 2, 2]
    scores = pd.DataFrame({"dataset": ["Exp1"] * 7, "setting": ["x"] * 7, "phase": ["ONLINE"] * 7, "window_index": range(7), "center_cycle": range(7), "final_transition_score": high, "candidate_threshold": [1.] * 7, "adapter_frozen": [0] * 7})
    events = extract_candidate_events(scores, config)
    assert len(events) == 1 and events.start_cycle.iloc[0] == 4


def test_dual_branch_cannot_score_below_frozen_branch() -> None:
    config = _config(); model = GradualStateTCN(config); frame = _frame(); normalizer, baseline = _baseline(model, frame, config)
    run = run_target_online(model, frame, config, setting="DualBranchAdaptation", source_normalizer=normalizer, source_baseline=baseline)
    value = run.scores.dropna(subset=["score_frozen", "final_transition_score"])
    assert np.all(value.final_transition_score >= value.score_frozen)


def test_target_labels_loaded_only_after_online_outputs_freeze(tmp_path) -> None:
    exp1, exp2 = _frame(210, "Exp1"), _frame(210, "Exp2"); input_path = tmp_path / "input.csv"; pd.concat([exp1, exp2]).to_csv(input_path, index=False)
    segments = [{"dataset": dataset, "stage": stage, "effective_start": start, "effective_end": end, "actual_start": start, "actual_end": end, "note": "test"} for dataset in ("Exp1", "Exp2") for stage, start, end in ((1, 0., 100.), (2, 100., 220.))]
    mapping = tmp_path / "mapping.json"; mapping.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    config = replace(_config(), input_path=str(input_path), cycle_mapping_path=str(mapping), output_dir=str(tmp_path / "out"), source_train_fraction=.95, source_max_training_examples=64)
    run_pipeline(config); audit = json.loads((tmp_path / "out" / "v341_online_causality_audit.json").read_text(encoding="utf-8"))
    assert audit["target_labels_read_before_online_freeze"] is False
    assert (tmp_path / "out" / "v341_online_transition_scores.csv").exists()


def test_all_settings_start_from_identical_source_model() -> None:
    config = _config(); model = GradualStateTCN(config); frame = _frame(); normalizer, baseline = _baseline(model, frame, config)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for setting in ("DirectTransfer", "CalibrationOnly", "OnlineAdaptation", "DualBranchAdaptation"):
        run_target_online(model, frame, config, setting=setting, source_normalizer=normalizer, source_baseline=baseline)
    assert all(np.array_equal(before[name].numpy(), value.detach().numpy()) for name, value in model.state_dict().items())
