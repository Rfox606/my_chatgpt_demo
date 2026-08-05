from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gradual_state_monitoring_v33.config import GradualStateMonitoringV33Config
from gradual_state_monitoring_v33.data import MAIN_FEATURES, make_synthetic_sequence
from gradual_state_monitoring_v33.evaluation import prefix_causality_evaluation, synthetic_acceptance
from gradual_state_monitoring_v33.features import build_causal_descriptors
from gradual_state_monitoring_v33.forecasting import run_causal_benchmark_forecasts
from gradual_state_monitoring_v33.pipeline import run_monitor


def test_multiscale_descriptor_is_causal_when_future_is_appended() -> None:
    frame = make_synthetic_sequence("ramp", length=180)
    original = frame.loc[:, MAIN_FEATURES].to_numpy(float)
    extended = np.vstack((original, original[-30:] + 10.0))
    left = build_causal_descriptors(original)
    right = build_causal_descriptors(extended)[:len(original)]
    assert left.shape[1] == len(MAIN_FEATURES) * 9
    assert np.array_equal(left, right)


def test_all_online_monitor_outputs_are_prefix_causal_after_arbitrary_append() -> None:
    config = GradualStateMonitoringV33Config(history_windows=48, evidence_min_history=20, evidence_baseline_windows=96)
    frame = make_synthetic_sequence("step", length=420)
    audit = prefix_causality_evaluation(frame, config)
    assert audit["status"] == "PASS", audit


def test_synthetic_acceptance_covers_stable_step_ramp_and_spike() -> None:
    results = synthetic_acceptance(GradualStateMonitoringV33Config())
    failed = results.loc[results.status != "PASS", ["synthetic_case", "status"]]
    assert failed.empty, failed.to_dict("records")


def test_benchmark_forecasts_do_not_change_for_existing_origins_when_future_is_appended() -> None:
    config = GradualStateMonitoringV33Config(
        horizons=(5, 12, 24), history_windows=32, benchmark_min_train=20,
        benchmark_training_window=64, benchmark_refit_interval=8,
    )
    frame = make_synthetic_sequence("ramp", length=140)
    values = frame.loc[:, MAIN_FEATURES].to_numpy(float)
    full = run_causal_benchmark_forecasts(build_causal_descriptors(values), values, config, MAIN_FEATURES)
    extended_values = np.vstack((values, values[-30:] + .3))
    extended = run_causal_benchmark_forecasts(build_causal_descriptors(extended_values), extended_values, config, MAIN_FEATURES)
    original = full[full.target_index < len(values)].sort_values(["input_index", "horizon", "model"]).reset_index(drop=True)
    replay = extended[extended.target_index < len(values)].sort_values(["input_index", "horizon", "model"]).reset_index(drop=True)
    columns = ["mae", "prediction_available", *[f"prediction_{name}" for name in MAIN_FEATURES]]
    pd.testing.assert_frame_equal(original.loc[:, columns], replay.loc[:, columns], check_exact=True)


def test_temporary_spike_keeps_local_state_and_does_not_create_new_stable() -> None:
    result = run_monitor(make_synthetic_sequence("spike"), GradualStateMonitoringV33Config())
    states = result.state_path
    assert (states.online_state == "TEMPORARY_DISTURBANCE").any()
    assert not (states.online_state == "NEW_STABLE").any()
    assert states.local_state_id.max() == 1
