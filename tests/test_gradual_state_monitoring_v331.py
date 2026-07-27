from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from gradual_state_monitoring_v33.data import MAIN_FEATURES, make_synthetic_sequence
from gradual_state_monitoring_v331.config import GradualStateMonitoringV331Config
from gradual_state_monitoring_v331.engine import FrozenDescriptorScaler, LocalStateLibrary, run_monitor
from gradual_state_monitoring_v331.pipeline import gate_decision, prefix_causality, state_library_validation, synthetic_acceptance


def _fast_config() -> GradualStateMonitoringV331Config:
    return GradualStateMonitoringV331Config(
        descriptor_scaler_calibration_windows=48, history_windows=48,
        evidence_min_history=20, evidence_baseline_windows=96,
    )


def test_frozen_descriptor_scaler_never_changes_after_calibration() -> None:
    scaler = FrozenDescriptorScaler(2, 3, 1e-9)
    for value in (np.array((1.0, 2.0)), np.array((3.0, 4.0)), np.array((5.0, 6.0))):
        scaler.update(value)
    before_mean = scaler.mean.copy(); before_variance = scaler.variance.copy()
    scaler.update(np.array((100.0, -100.0)))
    assert scaler.frozen and scaler.count == 3
    assert np.array_equal(before_mean, scaler.mean)
    assert np.array_equal(before_variance, scaler.variance)


def test_future_append_does_not_change_frozen_monitor_history() -> None:
    config = _fast_config()
    frame = make_synthetic_sequence("step", length=360)
    full = run_monitor(frame, config)
    audit = prefix_causality(frame, full, config)
    assert audit["status"] == "PASS", audit


def test_delayed_entry_monitor_retains_slow_fast_models() -> None:
    config = _fast_config()
    result = run_monitor(make_synthetic_sequence("ramp", length=420).iloc[64:].reset_index(drop=True), config, include_dual=True)
    dual = result.dual_predictions
    assert (dual.descriptor_scaler_frozen == 1).any()
    assert dual.slow_prediction_error.notna().any()
    assert dual.fast_prediction_error.notna().any()


def test_state_library_reuses_and_creates_ids_at_robust_distance() -> None:
    config = GradualStateMonitoringV331Config(state_reuse_distance_threshold=2.0)
    library = LocalStateLibrary(config)
    library.initialise(np.array((0.0, 0.0)), np.ones(2), 0.0)
    second, action, _, _ = library.confirm(np.array((5.0, 5.0)), np.ones(2), 1.0)
    third, action3, _, _ = library.confirm(np.array((12.0, 12.0)), np.ones(2), 2.0)
    reused, action4, _, _ = library.confirm(np.array((5.1, 5.0)), np.ones(2), 3.0)
    assert (second, action, third, action3, reused, action4) == (2, "created", 3, "created", 2, "reused")
    assert state_library_validation(config)["status"] == "PASS"


def test_evidence_removal_replays_a_complete_new_state_path() -> None:
    config = _fast_config()
    frame = make_synthetic_sequence("ramp", length=420)
    full = run_monitor(frame, config)
    removed = run_monitor(frame, replace(config, evidence_weights={"prediction_surprise": 0.0, "distribution_shift": .3, "drift_velocity": .3}))
    assert full.state_path is not removed.state_path
    assert len(full.state_path) == len(removed.state_path) == len(frame)
    assert not np.array_equal(full.evidence.change_evidence.to_numpy(), removed.evidence.change_evidence.to_numpy())


def test_all_v331_synthetic_acceptance_cases_pass() -> None:
    results = synthetic_acceptance(GradualStateMonitoringV331Config())
    assert (results.status == "PASS").all(), results.to_dict("records")


def test_gate_b2_rejects_excessive_transition_fraction_even_without_switch_rule() -> None:
    config = GradualStateMonitoringV331Config()
    forecast = pd.DataFrame([{"dataset": "Exp1", "model": "Persistence", "horizon": 100, "mae": 1.0, "stable_mae": 1.0},
                             {"dataset": "Exp1", "model": "Ridge", "horizon": 100, "mae": .9, "stable_mae": 1.0}])
    synthetic = pd.DataFrame([{"synthetic_case": name, "status": "PASS"} for name in ("stable", "step", "ramp", "spike")])
    ablation = pd.DataFrame([{"dataset": "Exp1", "variant": name, "transition_fraction": .41, "state_agreement_with_full": 1.0}
                             for name in ("full", "without_prediction_surprise", "without_distribution_shift", "without_drift_velocity")])
    overlap = pd.DataFrame([{"dataset": "Exp1", "window_overlap_ratio": .5, "state_agreement": .9}, {"dataset": "Exp1", "window_overlap_ratio": 0.0, "state_agreement": .9}])
    delayed = pd.DataFrame([{"dataset": "Exp1", "state_agreement": .9, "transition_interval_iou": .9, "local_state_count_difference": 0, "dual_model_enabled": 1}])
    result = gate_decision(forecast, synthetic, ablation, overlap, delayed, {"status": "PASS"}, {"Exp1": {"status": "PASS"}}, config)
    assert result["gate_B2_real_data_stability"]["status"] == "FAIL"
