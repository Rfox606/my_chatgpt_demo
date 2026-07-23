from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from partial_shared_primitives_progression_v32.bocpd import bocpd_confirmed_segments, segment_descriptors
from partial_shared_primitives_progression_v32.config import FEATURES, V32Config
from partial_shared_primitives_progression_v32.continuous import continuous_process, delayed_entry_convergence
from partial_shared_primitives_progression_v32.forecasting import MODEL_NAMES, fit_source_frozen, prefix_freeze_metrics, run_target_transfer
from partial_shared_primitives_progression_v32.primitives import online_target_states
from partial_shared_primitives_progression_v32.report import write_report


def _frame(rows: int = 240, dataset: str = "Exp1", shift: float = 0.0) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    base = 0.02 * index + np.sin(index / 9.0) + shift
    payload: dict[str, object] = {
        "dataset": [dataset] * rows,
        "window_id": index.astype(int),
        "window_index": index.astype(int),
        "start_cycle": index * 10.0,
        "end_cycle": index * 10.0 + 8.0,
        "center_cycle": index * 10.0 + 4.0,
    }
    for number, name in enumerate(FEATURES):
        payload[name] = base * (1.0 + number / 10.0) + number * 0.1
    return pd.DataFrame(payload)


def _config() -> V32Config:
    return replace(
        V32Config(),
        history_windows=16,
        source_train_windows=120,
        adapter_warmup_windows=24,
        negative_transfer_confirmations=2,
        minimum_segment_windows=8,
        bocpd_max_run_length=64,
        delayed_common_arrived_windows=5,
    )


def _descriptors(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": np.arange(len(values)),
            "start_index": np.arange(len(values)) * 10,
            "end_index": np.arange(len(values)) * 10 + 9,
            "start_cycle": np.arange(len(values)) * 100.0,
            "end_cycle": np.arange(len(values)) * 100.0 + 90.0,
            "descriptor_duration": np.full(len(values), 90.0),
            "descriptor_start_mean": values,
            "descriptor_end_mean": np.asarray(values) + 0.2,
            "descriptor_net_change": np.full(len(values), 0.2),
            "descriptor_overall_slope": np.full(len(values), 0.02),
            "descriptor_first_half_slope": np.full(len(values), 0.02),
            "descriptor_second_half_slope": np.full(len(values), 0.02),
            "descriptor_volatility_start": np.full(len(values), 0.1),
            "descriptor_volatility_end": np.full(len(values), 0.1),
            "descriptor_volatility_change": np.zeros(len(values)),
            "descriptor_prediction_error_mean": np.full(len(values), 0.1),
            "descriptor_prediction_error_trend": np.zeros(len(values)),
            "descriptor_entry_jump": np.zeros(len(values)),
            "descriptor_exit_jump": np.zeros(len(values)),
        }
    )


def test_bocpd_uses_distinct_prior_and_growth_likelihoods_and_detects_step() -> None:
    config = _config()
    stable = np.zeros(80)
    stepped = np.r_[np.zeros(40), np.full(6, 8.0), np.zeros(34)]
    stable_audit, _ = bocpd_confirmed_segments(stable, np.arange(len(stable)), config)
    step_audit, _ = bocpd_confirmed_segments(stepped, np.arange(len(stepped)), config)
    assert stable_audit.bocpd_change_posterior.mean() < 0.02
    assert step_audit.bocpd_change_posterior.iloc[40:46].max() > 0.65
    assert not np.allclose(step_audit.bocpd_change_posterior, config.bocpd_hazard)
    assert (step_audit.bocpd_prior_predictive != step_audit.bocpd_growth_predictive_mean).any()


def test_bocpd_append_only_future_cannot_change_history() -> None:
    config = _config()
    prefix = np.r_[np.zeros(40), np.full(4, 7.0)]
    left, _ = bocpd_confirmed_segments(prefix, np.arange(len(prefix)), config)
    right, _ = bocpd_confirmed_segments(np.r_[prefix, np.arange(30.0)], np.arange(len(prefix) + 30), config)
    pd.testing.assert_frame_equal(left, right.iloc[: len(prefix)].reset_index(drop=True))


def test_fixed_main_features_and_horizons_exclude_deprecated_inputs() -> None:
    config = V32Config()
    assert config.features == ("rx_mean", "rx_q05", "ry_mean", "ry_q05", "ry_p2p", "rs_rms")
    assert config.horizons == (1, 5, 20)
    assert {"rx_absmean", "ry_absmean", "rs_mean"}.issubset(config.jsonable()["forbidden_model_inputs"])


def test_source_frozen_is_immutable_predict_then_update_and_adapter_is_independent() -> None:
    config = _config()
    source, target = _frame(dataset="Exp1"), _frame(dataset="Exp2", shift=3.0)
    frozen = fit_source_frozen(source, config)
    before = frozen.parameter_sha256
    records, weights = run_target_transfer(frozen, target, config)
    assert frozen.parameter_sha256 == before
    assert set(records.model) == set(MODEL_NAMES)
    assert set(records.horizon) == {1, 5, 20}
    assert records.predict_then_update.all()
    assert (records.observed_index - records.origin_index == records.horizon).all()
    assert weights.source_frozen_parameters_updated.eq(False).all()
    gated = records.loc[records.adapter_gate_active_at_issue]
    if not gated.empty:
        joined = gated.pivot_table(index=["origin_index", "horizon"], columns="model", values="absolute_error")
        assert not np.allclose(joined["Source_Adapter"], joined["Target_From_Scratch"])


def test_append_only_target_length_causality_and_prefix_freeze_metrics() -> None:
    config = _config()
    source, target = _frame(dataset="Exp1"), _frame(dataset="Exp2", shift=0.5)
    frozen = fit_source_frozen(source, config)
    base_records, _ = run_target_transfer(frozen, target.iloc[:180].copy(), config)
    appended_tail = target.iloc[180:220].copy()
    appended_tail.loc[:, list(FEATURES)] += 100.0
    appended = pd.concat([target.iloc[:180], appended_tail], ignore_index=True)
    replay_records, _ = run_target_transfer(frozen, appended, config)
    pd.testing.assert_frame_equal(base_records, replay_records.loc[replay_records.observed_index.lt(180)].reset_index(drop=True))
    metrics = prefix_freeze_metrics(frozen, target, config)
    assert set(metrics.model) == set(MODEL_NAMES)
    assert {1, 5, 20, "weighted"}.issubset(set(metrics.horizon))
    assert not metrics.model_updated_during_evaluation.any()


def test_segment_descriptor_rows_equal_confirmed_segments_not_windows() -> None:
    frame = _frame(64)
    segments = [(0, 15), (16, 31), (32, 63)]
    descriptors = segment_descriptors(frame, np.arange(len(frame), dtype=float), np.ones(len(frame)), segments, FEATURES)
    assert len(descriptors) == len(segments) and len(descriptors) != len(frame)
    required = {
        "descriptor_duration", "descriptor_start_mean", "descriptor_end_mean", "descriptor_net_change",
        "descriptor_overall_slope", "descriptor_first_half_slope", "descriptor_second_half_slope",
        "descriptor_volatility_start", "descriptor_volatility_end", "descriptor_volatility_change",
        "descriptor_prediction_error_mean", "descriptor_prediction_error_trend", "descriptor_entry_jump", "descriptor_exit_jump",
    }
    assert required.issubset(descriptors.columns)


def test_private_target_states_can_be_created_online_without_source_k() -> None:
    path, log, decision = online_target_states(_descriptors([0.0, 10.0, 20.0, 20.1]), None, _config())
    assert decision["private_state_can_grow_online"] and not decision["source_k_imposed_on_target"]
    assert set(path.current_state_type) == {"TARGET_PRIVATE"}
    assert (log.event == "created").sum() >= 3
    assert path.target_state_count_so_far.iloc[-1] >= 3


def test_progression_does_not_read_state_id_and_platform_is_lower_than_persistent_change() -> None:
    config = _config()
    target = _frame(80, "Exp2")
    records = pd.DataFrame(
        [
            {"observed_index": i, "horizon": h, "model": "Negative_transfer_Gated_Mixture", "absolute_error": 0.01 if i < 40 else 2.0}
            for i in range(20, 80) for h in (1, 5, 20)
        ]
    )
    bocpd = pd.DataFrame({"window_index": np.arange(80), "bocpd_change_posterior": 0.0, "bocpd_run_length_entropy": 0.1, "boundary_confirmed": False})
    process = continuous_process(
        target, records, bocpd,
        {"initial_progression_prior_mean": 0.2, "initial_progression_prior_std": 0.2, "initial_match_quality": 0.8},
        pd.DataFrame(), config,
    )
    assert process.state_id_input_count.eq(0).all() and not process.cycle_used_as_model_feature.any()
    assert process.progression_increment.iloc[:35].mean() < process.progression_increment.iloc[45:].mean()
    assert process.relative_progression_score.iloc[0] > 0.0


def test_short_spike_does_not_confirm_permanent_bocpd_segment() -> None:
    audit, _ = bocpd_confirmed_segments(np.r_[np.zeros(40), [9.0], np.zeros(40)], np.arange(81), _config())
    assert not audit.boundary_confirmed.any()


def test_delayed_entry_output_has_required_common_window_and_report_is_utf8(tmp_path: Path) -> None:
    config = _config()
    def process(offset: float) -> pd.DataFrame:
        cycles = np.arange(10, dtype=float) * 10.0 + 100.0
        return pd.DataFrame({"center_cycle": cycles, "progression_increment": np.arange(10) + offset, "relative_progression_score": np.arange(10) / 10, "progression_uncertainty": 1 / (1 + np.arange(10))})
    early = process(0.0)
    early.loc[:1, "progression_increment"] += 5.0
    early.loc[:1, "progression_uncertainty"] += 2.0
    delayed = delayed_entry_convergence({0.0: (early, pd.DataFrame()), 20.0: (process(0.0), pd.DataFrame())}, config)
    assert (delayed.common_arrived_windows == config.delayed_common_arrived_windows).all()
    older = delayed.loc[delayed.entry_cycle.eq(0.0)]
    assert older.progression_increment_abs_difference.tail(2).mean() < older.progression_increment_abs_difference.head(2).mean()
    assert older.uncertainty.tail(2).mean() < older.uncertainty.head(2).mean()
    decision = {"status": "FAIL", "directions": {}}
    write_report(tmp_path, decision, {})
    assert "部分共享" in (tmp_path / "partial_shared_primitives_progression_v32_report.md").read_text(encoding="utf-8")


def test_historical_v31_sources_are_not_modified() -> None:
    # Frozen baseline SHA sentinels cover both source and retained v3.1 output.
    root = Path(__file__).resolve().parents[1]
    expected = {
        "partial_shared_primitives_progression_v31/bocpd.py": "4af24555523bb9a838b9c3da45ec8f813b207ba279612c8877a452fd711c9e4b",
        "partial_shared_primitives_progression_v31/forecasting.py": "018680cc3bf81dc45a5e066c1e46fcdf95357db1590b772f615c4169bc1646aa",
        "partial_shared_primitives_progression_v31/evaluation.py": "03570cbdf76c7a0b3c4c376a05c37bcced0f5bbac1d40c52b802bbfc26a2b6f7",
        "outputs_partial_shared_primitives_progression_v31/results/gate_a_multihorizon_forecasts_v31.csv": "d2a15a88207d2ef1dcaa3bdd94e71a643d92ea1a6e4b2c0c07cbe85765dbbe4b",
        "outputs_partial_shared_primitives_progression_v31/diagnostics/gate_decision_v31.json": "d6aa5c824a6a5d776a066d0184f4b989dd7dbabc90159d9f08eab2b3ae13de76",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
