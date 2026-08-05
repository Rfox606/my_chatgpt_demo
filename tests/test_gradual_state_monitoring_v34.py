from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gradual_state_monitoring_v33.data import MAIN_FEATURES, make_synthetic_sequence
from gradual_state_monitoring_v3321.forecasting import legal_origins as v3321_legal_origins
from gradual_state_monitoring_v34.config import GradualStateMonitoringV34Config
from gradual_state_monitoring_v34.data import validate_online_frame
from gradual_state_monitoring_v34.evaluation import match_candidate_events
from gradual_state_monitoring_v34.model import GradualStateTCN, set_deterministic_seed
from gradual_state_monitoring_v34.online import extract_candidate_events, run_target_online
from gradual_state_monitoring_v34.pipeline import run_pipeline
from gradual_state_monitoring_v34.training import legal_training_origins


def _config() -> GradualStateMonitoringV34Config:
    return GradualStateMonitoringV34Config(random_seeds=(3401,), source_epochs=1, upper_bound_epochs=1)


def _frame(rows: int = 160) -> pd.DataFrame:
    frame = make_synthetic_sequence("stable", length=rows, seed=34)
    return frame.loc[:, ["dataset", "window_index", "center_cycle", *MAIN_FEATURES]].assign(dataset="Exp1")


def test_target_online_path_rejects_stage_and_unapproved_columns() -> None:
    config = _config()
    with pytest.raises(AssertionError):
        validate_online_frame(_frame().assign(Stage1to5=1), config)
    with pytest.raises(AssertionError):
        validate_online_frame(_frame().assign(morphology=0.1), config)


def test_tcn_receptive_field_covers_required_128_windows() -> None:
    config = _config()
    assert config.receptive_field >= 128
    assert tuple(config.tcn_channels) == (32, 32, 32, 16, 16, 16, 16)


def test_training_origins_obey_strict_target_interval() -> None:
    config = _config()
    origins = legal_training_origins(640, config)
    assert len(origins)
    assert np.all(origins + config.source_prediction_horizon < 640)
    corrected = v3321_legal_origins(800, 128, 300, 640)
    assert np.all(corrected + 300 < 640)


def test_calibration_only_is_prefix_causal_when_future_is_appended() -> None:
    config = _config()
    set_deterministic_seed(3401)
    model = GradualStateTCN(config)
    prefix = _frame(150)
    full = pd.concat([prefix, _frame(170).iloc[150:]], ignore_index=True)
    first = run_target_online(model, prefix, config, setting="CalibrationOnly")
    second = run_target_online(model, full, config, setting="CalibrationOnly")
    left = first.scores.loc[:, ["phase", "transition_score", "final_transition_score", "candidate_active"]].reset_index(drop=True)
    right = second.scores.iloc[:len(first.scores)].loc[:, left.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_adapter_audit_never_references_a_future_target_window() -> None:
    config = _config()
    set_deterministic_seed(3402)
    run = run_target_online(GradualStateTCN(config), _frame(155), config, setting="OnlineAdaptation")
    assert (run.adaptation.update_max_input_index <= run.adaptation.current_index).all()
    assert set(run.scores.phase.unique()) == {"CALIBRATION", "ONLINE"}


def test_high_change_period_freezes_online_adapter() -> None:
    config = _config()
    frame = _frame(160)
    frame.loc[130:, list(MAIN_FEATURES)] += 10.0
    set_deterministic_seed(3403)
    run = run_target_online(GradualStateTCN(config), frame, config, setting="OnlineAdaptation")
    assert run.adaptation.adapter_frozen.sum() > 0


def test_candidate_extraction_requires_persistence_and_keeps_one_peak() -> None:
    config = replace(_config(), candidate_min_windows=3, candidate_merge_gap_windows=10, candidate_cooldown_windows=40)
    scores = pd.DataFrame({
        "dataset": ["Exp1"] * 7, "setting": ["OnlineAdaptation"] * 7,
        "phase": ["ONLINE"] * 7, "window_index": list(range(7)), "center_cycle": np.arange(7, dtype=float),
        "final_transition_score": [.1, .9, 1.3, 1.0, .1, .2, .1],
        "candidate_threshold": [.8] * 7, "short_score": np.ones(7), "medium_score": np.ones(7),
        "long_score": np.ones(7), "source_transition_probability": np.ones(7) * .2,
        "adapter_frozen": np.ones(7),
    })
    events = extract_candidate_events(scores, config)
    assert len(events) == 1
    assert float(events.peak_cycle.iloc[0]) == 2.0


def test_repeated_alerts_are_counted_in_event_precision_denominator() -> None:
    events = pd.DataFrame({
        "dataset": ["Exp1", "Exp1"], "setting": ["OnlineAdaptation", "OnlineAdaptation"],
        "event_id": [1, 2], "peak_cycle": [100., 110.], "peak_actual_cycle": [100., 110.], "peak_score": [1., .9],
    })
    boundaries = pd.DataFrame({"dataset": ["Exp1"], "from_stage": [1], "to_stage": [2],
                               "boundary_cycle": [100.], "boundary_actual_cycle": [100.]})
    _, metric = match_candidate_events(events, boundaries, 20., direction="test", setting="OnlineAdaptation", seed=1)
    assert metric["event_level_precision"] == 0.5
    assert metric["unmatched_candidate_count"] == 1


def test_small_end_to_end_run_writes_the_declared_v34_artifacts(tmp_path) -> None:
    exp1 = _frame(210)
    exp2 = _frame(210).assign(dataset="Exp2")
    exp2.loc[:, list(MAIN_FEATURES)] += 0.02
    input_path = tmp_path / "features.csv"
    pd.concat([exp1, exp2], ignore_index=True).to_csv(input_path, index=False)
    segments = []
    for dataset in ("Exp1", "Exp2"):
        segments.extend([
            {"dataset": dataset, "stage": 1, "effective_start": 0., "effective_end": 100., "actual_start": 0., "actual_end": 100., "note": "fixture"},
            {"dataset": dataset, "stage": 2, "effective_start": 100., "effective_end": 220., "actual_start": 100., "actual_end": 220., "note": "fixture"},
        ])
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    config = replace(_config(), input_path=str(input_path), cycle_mapping_path=str(mapping_path), output_dir=str(tmp_path / "out"),
                     source_train_fraction=.95, source_max_training_examples=64, adaptation_update_interval=100)
    result = run_pipeline(config)
    root = tmp_path / "out"
    assert result["scores"] > 0
    for name in (
        "v34_config.json", "v34_label_provenance.json", "v34_training_protocol.json", "v34_online_causality_audit.json",
        "v34_transition_scores.csv", "v34_candidate_events.csv", "v34_boundary_event_matches.csv", "v34_boundary_metrics.csv",
        "v34_boundary_ranking_metrics.csv", "v34_adaptation_metrics.csv", "v34_method_comparison.csv", "v34_seed_summary.csv",
        "gradual_state_monitoring_v34_report.md", "fig_v34_exp1_to_exp2_transition_scores.png",
        "fig_v34_exp2_to_exp1_transition_scores.png", "fig_v34_boundary_rank_comparison.png",
        "fig_v34_adaptation_ablation.png", "fig_v34_topk_recall.png",
    ):
        assert (root / name).exists(), name
