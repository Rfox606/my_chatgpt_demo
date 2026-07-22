from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from partial_shared_primitives_progression_v3.config import FEATURES, PartialSharedPrimitivesConfig
from partial_shared_primitives_progression_v3.data import assert_label_free
from partial_shared_primitives_progression_v3.evaluation import prefix_causality_audit, run_v3_pipeline
from partial_shared_primitives_progression_v3.progression import score_conditioned_continuous_progression
from partial_shared_primitives_progression_v3.shared import run_shared_causal_representation
from partial_shared_primitives_progression_v3.states import synthetic_state_revisit


def _frame(rows: int = 720, dataset: str = "Exp1", phase: float = 0.) -> pd.DataFrame:
    cycle = np.arange(rows, dtype=float) * 100. + 50.
    data: dict[str, object] = {"dataset": [dataset] * rows, "window_id": np.arange(rows), "window_index": np.arange(rows), "start_cycle": cycle - 49., "end_cycle": cycle + 49., "center_cycle": cycle}
    base = np.sin(cycle / (400. + 40. * phase)) + .35 * np.sin(cycle / (67. + phase * 5.))
    for number, feature in enumerate(FEATURES): data[feature] = base * (1. + number / 10.) + .01 * number + .1 * np.cos(cycle / (113. + number * 9.))
    return pd.DataFrame(data)


def _config() -> PartialSharedPrimitivesConfig:
    return replace(PartialSharedPrimitivesConfig(), primitive_calibration_windows=128, state_calibration_windows=256, primitive_bootstrap_replicates=3, causal_context_windows=16, continuous_minimum_history=16, continuous_calibration_window=64)


def test_forbidden_labels_morphology_debris_and_absolute_wear_rejected() -> None:
    for column in ("Stage", "morphology", "wear_debris", "absolute_wear"):
        with pytest.raises(AssertionError): assert_label_free(pd.DataFrame({column: [1.]}))
    assert_label_free(_frame(2))


def test_shared_representation_is_causal_under_future_change() -> None:
    config = _config(); source = pd.concat([_frame(dataset="Exp1"), _frame(dataset="Exp2", phase=1.)], ignore_index=True)
    original = run_shared_causal_representation(source, config).frame
    changed = source.copy(); changed.loc[(changed.dataset == "Exp2") & (changed.window_index >= 500), list(FEATURES)] += 999.
    replay = run_shared_causal_representation(changed, config).frame
    a = original.loc[original.dataset.eq("Exp2")].iloc[:500]; b = replay.loc[replay.dataset.eq("Exp2")].iloc[:500]
    assert np.array_equal(a.loc[:, ["forecast_mae", "forecast_activity", "shared_z0"]].to_numpy(), b.loc[:, ["forecast_mae", "forecast_activity", "shared_z0"]].to_numpy())


def test_independent_local_state_models_have_no_alignment_or_source_centres() -> None:
    config = _config(); frame = pd.concat([_frame(dataset="Exp1"), _frame(dataset="Exp2", phase=1.)], ignore_index=True); pipeline = run_v3_pipeline(frame, config)
    table = pipeline["state_models_table"]
    assert len(table) == 2 and (table.state_centre_provenance == "local_experiment_only").all()
    assert not table.state_id_alignment_performed.any() and not table.source_state_centre_used.any()
    names = pipeline["state_paths"].local_state_name.astype(str)
    assert names.str.startswith(("Exp1_", "Exp2_")).all()


def test_continuous_score_is_state_independent_and_nonconstant() -> None:
    config = _config(); representation = run_shared_causal_representation(pd.concat([_frame(dataset="Exp1"), _frame(dataset="Exp2", phase=1.)], ignore_index=True), config).frame
    score = score_conditioned_continuous_progression(representation, config)
    assert not score.state_model_input.any() and score.continuous_progression_score.std() > 1e-6
    assert np.isfinite(score.continuous_progression_score).all()


def test_prefix_audit_and_synthetic_state_revisit_without_monotonicity() -> None:
    config = _config(); frame = pd.concat([_frame(dataset="Exp1"), _frame(dataset="Exp2", phase=1.)], ignore_index=True)
    assert prefix_causality_audit(frame, config)["status"] == "PASS"
    synthetic = synthetic_state_revisit(config)
    assert synthetic["returns_are_allowed"] and synthetic["no_monotonic_state_constraint"]


def test_v45_and_prior_v2_history_are_unchanged() -> None:
    def digest(path: str) -> str: return hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert digest("continuous_state_v45/state_engine.py") == "80beaca1f7a58c1aed0f5135cd5493b3d00c75110169a366aabb5103df45da22"
    assert digest("outputs_continuous_state_v45/results/window_feature_raw_v45.csv") == "5a6a20752a2132da62993ffd94575e2c2088c6021fad44adbcc7383eb8eae1ab"
    assert digest("cross_experiment_adaptive_state_v1/online.py") == "b0d4373039f3b3b1de8994c28012d2910b051dd7e3fbb40f6f7aa0ef13e5d78d"

