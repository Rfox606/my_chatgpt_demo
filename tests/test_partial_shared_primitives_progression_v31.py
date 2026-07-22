from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from partial_shared_primitives_progression_v31.bocpd import bocpd_confirmed_segments, causal_activity_energy, segment_descriptors
from partial_shared_primitives_progression_v31.config import FEATURES, V31Config
from partial_shared_primitives_progression_v31.continuous import continuous_process, delayed_entry_convergence, synthetic_ood_uncertainty
from partial_shared_primitives_progression_v31.data import assert_formal_input
from partial_shared_primitives_progression_v31.forecasting import fit_source_frozen, run_target_transfer
from partial_shared_primitives_progression_v31.primitives import fit_segment_clusters, primitive_table, private_state_path


def _frame(rows: int = 180, dataset: str = "Exp1", phase: float = 0.0) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    cycle = index * 10.0 + 5.0
    base = .03 * index + np.sin(index / (7.0 + phase)) + .15 * np.cos(index / (2.0 + phase))
    data: dict[str, object] = {"dataset": [dataset] * rows, "window_id": index.astype(int), "window_index": index.astype(int), "start_cycle": cycle - 4.5, "end_cycle": cycle + 4.5, "center_cycle": cycle}
    for number, feature in enumerate(FEATURES):
        data[feature] = base * (1.0 + number / 20.0) + .01 * number
    return pd.DataFrame(data)


def _config() -> V31Config:
    return replace(V31Config(), history_windows=16, horizons=(1, 2, 4), source_train_windows=80, adapter_warmup_windows=24, negative_transfer_confirmations=2, bocpd_max_run_length=64, minimum_segment_windows=8, private_state_calibration_confirmed_segments=6, primitive_k_candidates=(2, 3), private_state_k_candidates=(2, 3), prefix_cutoff_cycles=(300.0, 700.0))


def _descriptors(rows: int = 8) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    return pd.DataFrame({"segment_id": index.astype(int), "start_cycle": index * 100.0, "end_cycle": index * 100.0 + 90.0, "window_count": np.full(rows, 10), "mean_innovation_energy": index / 10.0, "mean_activity_energy": index / 20.0, "mean_rx_mean": np.r_[np.zeros(rows // 2), np.ones(rows - rows // 2)] + index / 100.0, "slope_rx_mean": np.r_[np.full(rows // 2, -.1), np.full(rows - rows // 2, .2)]})


def test_formal_input_rejects_all_forbidden_outcomes() -> None:
    for forbidden in ("Stage", "morphology", "wear_debris", "absolute_wear", "mass_loss"):
        with pytest.raises(AssertionError):
            assert_formal_input(pd.DataFrame({forbidden: [1.0]}))
    assert_formal_input(_frame(2))


def test_gate_a_is_asymmetric_multihorizon_and_frozen_source_never_updates() -> None:
    config = _config(); source = _frame(dataset="Exp1"); target = _frame(dataset="Exp2", phase=.7)
    frozen = fit_source_frozen(source, config); before = {h: weights.copy() for h, weights in frozen.weights.items()}
    records, gates = run_target_transfer(frozen, target, config)
    assert set(records.model) == {"Source_Frozen", "Target_From_Scratch", "Source_Plus_Adapter_Gated"}
    assert set(records.horizon) == set(config.horizons)
    assert records.prediction_available.all() and not gates.empty
    for horizon, weights in before.items():
        assert np.array_equal(weights, frozen.weights[horizon])


def test_bocpd_is_causal_and_dynamic_primitives_receive_segments_not_windows() -> None:
    config = _config(); frame = _frame(96); energy = causal_activity_energy(frame, config.features)
    audit, segments = bocpd_confirmed_segments(energy, frame.center_cycle.to_numpy(float), config)
    changed = energy.copy(); changed[65:] += 100.0
    replay, _ = bocpd_confirmed_segments(changed, frame.center_cycle.to_numpy(float), config)
    assert np.array_equal(audit.iloc[:65].to_numpy(), replay.iloc[:65].to_numpy())
    descriptors = segment_descriptors(frame, energy, energy, segments, config.features)
    assert len(descriptors) == len(segments) and len(descriptors) < len(frame)
    model = fit_segment_clusters(_descriptors(), (2, 3), config, "source_confirmed_segments_only")
    table = primitive_table(model, _descriptors(), "Exp1")
    assert model is not None and (table.descriptor_provenance == "BOCPD_confirmed_segments").all()
    assert table.window_rows_not_used_as_primitives.all()


def test_target_private_states_are_target_only_and_k_is_selected_not_fixed() -> None:
    path, model, decision = private_state_path(_descriptors(), _config())
    assert decision["status"] == "PASS" and model is not None
    assert model.provenance == "target_confirmed_segments_only"
    assert not path.source_state_centre_used.any() and not path.cross_experiment_state_alignment.any()
    assert path.private_state_name.iloc[-1].startswith("TARGET_PRIVATE_")


def test_continuous_process_has_no_state_input_and_ood_uncertainty_increases() -> None:
    config = _config(); target = _frame(dataset="Exp2", phase=.4)
    records = pd.DataFrame([{ "dataset": "Exp2", "entry_cycle": 0.0, "observed_index": index, "origin_index": index - 1, "center_cycle": float(target.center_cycle.iloc[index]), "horizon": horizon, "model": "Source_Plus_Adapter_Gated", "squared_error": float(.01 * (horizon + index % 3)), "prediction_available": True, "adapter_gate_active": False} for index in range(16, len(target)) for horizon in config.horizons])
    bocpd = pd.DataFrame({"window_index": np.arange(len(target)), "bocpd_run_length_entropy": np.linspace(0., 1., len(target))})
    process = continuous_process(target, records, bocpd, prior=.1, config=config)
    assert np.isfinite(process.loc[:, ["cumulative_progression", "activity", "initial_prior", "uncertainty"]].to_numpy(float)).all()
    assert process.state_id_input_count.eq(0).all() and not process.rolling_z_used.any()
    assert {"multi_horizon_residual_dispersion", "adapter_support_deficit", "bocpd_run_length_entropy"}.issubset(process.columns)
    ood = synthetic_ood_uncertainty(config)
    assert ood["status"] == "PASS" and not ood["state_id_used"]


def test_delayed_entry_uses_fixed_common_arrival_window() -> None:
    config = replace(_config(), delayed_common_arrived_windows=3)
    def path(entry: float, offset: float) -> pd.DataFrame:
        cycles = np.arange(10, dtype=float) * 10.0 + entry
        return pd.DataFrame({"center_cycle": cycles, "progression_increment": np.arange(10, dtype=float) + offset, "cumulative_progression": np.arange(10, dtype=float), "activity": np.ones(10), "initial_prior": np.ones(10), "uncertainty": np.ones(10)})
    result = delayed_entry_convergence({0.0: path(0.0, 0.0), 20.0: path(20.0, 0.0)}, config)
    assert set(result.common_arrived_windows) == {3} and result.finite.all()


def test_protocol_has_fixed_cycle_audits_and_no_relative_completion_parameter() -> None:
    config = V31Config()
    assert config.prefix_cutoff_cycles == (3000.0, 9000.0)
    serialized = config.jsonable()
    assert "relative_complete_progress" in serialized["forbidden_model_inputs"]
    assert "target_final_length" in serialized["forbidden_model_inputs"]
