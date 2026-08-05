from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cross_experiment_adaptive_state_v1.analysis import delayed_entry_summary
from cross_experiment_adaptive_state_v1.config import FEATURE_CONFIGS, CrossExperimentAdaptiveConfig
from cross_experiment_adaptive_state_v1.data import assert_formal_frame
from cross_experiment_adaptive_state_v1.model import train_source_models
from cross_experiment_adaptive_state_v1.online import run_target_online


def _config(**changes: object) -> CrossExperimentAdaptiveConfig:
    defaults: dict[str, object] = {
        "source_max_pairs_per_gap_bin": 80, "target_update_pair_limit": 60,
        "target_initialization_cycles": 1000.0, "target_update_interval_cycles": 500.0,
        "lambda_ramp_cycles": 2000.0, "random_seed": 17,
    }
    defaults.update(changes)
    return CrossExperimentAdaptiveConfig(**defaults)


def _frame(dataset: str, n: int = 160, *, scale: float = 1.0, constant: bool = False) -> pd.DataFrame:
    cycles = np.arange(n, dtype=float) * 100.0 + 50.0
    trend = np.zeros(n) if constant else np.linspace(0.0, scale, n)
    rows: dict[str, object] = {
        "dataset": dataset, "window_id": np.arange(n), "window_index": np.arange(n),
        "start_cycle": cycles - 45.0, "end_cycle": cycles + 45.0, "center_cycle": cycles,
    }
    for position, feature in enumerate(FEATURE_CONFIGS["F_core_v45"]):
        rows[feature] = trend * (1.0 + position * .08) + np.sin(cycles / (400.0 + position * 30.0)) * .02
    return pd.DataFrame(rows)


@pytest.fixture()
def source_models() -> tuple[dict[str, object], CrossExperimentAdaptiveConfig]:
    config = _config()
    return train_source_models(_frame("Exp1"), config), config


def test_forbidden_stage_morphology_and_debris_are_rejected() -> None:
    for column in ("Stage1to5", "Sa", "Sq", "Sz", "Sku", "wear_debris_count"):
        with pytest.raises(AssertionError):
            assert_formal_frame(_frame("Exp2", 20).assign(**{column: 1}))


def test_future_target_mutation_does_not_change_emitted_prefix(source_models: tuple[dict[str, object], CrossExperimentAdaptiveConfig]) -> None:
    models, config = source_models; target = _frame("Exp2")
    original = run_target_online(target, models, 0.0, config).scores
    altered = target.copy(); altered.loc[altered.center_cycle > 7000, list(FEATURE_CONFIGS["F_core_v45"])] += 90.0
    replay = run_target_online(altered, models, 0.0, config).scores
    columns = ["progression_prior", "progression_adapted", "activity_score", "state_uncertainty"]
    left = original.loc[original.center_cycle <= 7000, columns].to_numpy(float)
    right = replay.loc[replay.center_cycle <= 7000, columns].to_numpy(float)
    assert np.allclose(left, right, atol=1e-12, rtol=0)


def test_adapter_freezes_during_initialization_and_only_uses_arrived_pairs(source_models: tuple[dict[str, object], CrossExperimentAdaptiveConfig]) -> None:
    models, config = source_models; result = run_target_online(_frame("Exp2"), models, 0.0, config)
    early = result.scores.loc[result.scores.center_cycle <= config.target_initialization_cycles]
    assert not early.adapter_update_applied.any()
    applied = result.scores.loc[result.scores.adapter_update_applied.eq(1)]
    assert (applied.target_pair_latest_cycle < applied.center_cycle).all()


def test_source_model_is_frozen_and_target_entry_is_not_zero(source_models: tuple[dict[str, object], CrossExperimentAdaptiveConfig]) -> None:
    models, config = source_models; before = {name: model.coefficients.copy() for name, model in models.items()}
    result = run_target_online(_frame("Exp2"), models, 0.0, config)
    assert result.source_frozen
    assert abs(float(result.scores.progression_prior.iloc[0])) > 1e-12
    for name, model in models.items():
        assert np.array_equal(before[name], model.coefficients)


def test_constant_signal_does_not_cause_continuing_adapter_drift(source_models: tuple[dict[str, object], CrossExperimentAdaptiveConfig]) -> None:
    models, config = source_models; target = _frame("Exp2", constant=True)
    target.loc[:, list(FEATURE_CONFIGS["F_core_v45"])] = target.loc[0, list(FEATURE_CONFIGS["F_core_v45"])].to_numpy()
    result = run_target_online(target, models, 0.0, config)
    assert result.scores.adapter_parameter_norm.max() <= 1e-12


def test_gradual_drift_increases_progression_and_plateau_lowers_activity(source_models: tuple[dict[str, object], CrossExperimentAdaptiveConfig]) -> None:
    models, config = source_models; target = _frame("Exp2")
    target.loc[target.index >= 105, list(FEATURE_CONFIGS["F_core_v45"])] = target.loc[104, list(FEATURE_CONFIGS["F_core_v45"])] .to_numpy()
    result = run_target_online(target, models, 0.0, config).scores
    assert result.progression_adapted.iloc[95] > result.progression_adapted.iloc[15]
    assert result.activity_score.iloc[-15:].median() < result.activity_score.iloc[80:105].median()


def test_short_jump_is_activity_transient_not_permanent_progression_jump(source_models: tuple[dict[str, object], CrossExperimentAdaptiveConfig]) -> None:
    models, config = source_models; target = _frame("Exp2", constant=True)
    target.loc[70:74, list(FEATURE_CONFIGS["F_core_v45"])] += 8.0
    result = run_target_online(target, models, 0.0, config).scores
    assert result.activity_score.iloc[70:76].max() > result.activity_score.iloc[20:60].median()
    assert abs(result.progression_adapted.iloc[-1] - result.progression_adapted.iloc[60]) < .15


def test_ood_raises_uncertainty_and_suppresses_update(source_models: tuple[dict[str, object], CrossExperimentAdaptiveConfig]) -> None:
    models, config = source_models; target = _frame("Exp2")
    target.loc[target.index >= 90, list(FEATURE_CONFIGS["F_core_v45"])] += 1e5
    result = run_target_online(target, models, 0.0, config).scores
    ood = result.loc[result.ood_ratio > 2.5]
    assert not ood.empty and (ood.adapter_update_reason == "ood_suppressed").any()
    assert ood.state_uncertainty.median() > result.state_uncertainty.iloc[:50].median()


def test_adapter_bounds_and_delayed_entry_common_suffix(source_models: tuple[dict[str, object], CrossExperimentAdaptiveConfig]) -> None:
    models, _ = source_models; config = _config(adapter_max_norm=.03, adapter_max_step_norm=.01)
    target = _frame("Exp2")
    results = []
    for entry in (0.0, 3000.0, 6000.0):
        result = run_target_online(target, models, entry, config).scores.assign(direction="Exp1_to_Exp2")
        assert result.adapter_parameter_norm.max() <= config.adapter_max_norm + 1e-12
        results.append(result)
    summary = delayed_entry_summary(pd.concat(results, ignore_index=True))
    common = summary.loc[summary.row_type.eq("common_suffix_convergence")]
    assert not common.empty and common.common_windows.iloc[0] > 0


def test_objective_json_is_present_and_disables_absolute_wear_comparison() -> None:
    payload = json.loads(Path("metadata/research_objective_cross_experiment_adaptive_state.json").read_text(encoding="utf-8"))
    assert payload["objective_version"] == "ceap_v1"
    assert payload["absolute_wear_comparison_required"] is False


def test_v45_code_and_output_are_unchanged() -> None:
    expected = {
        "continuous_state_v45/state_engine.py": "80beaca1f7a58c1aed0f5135cd5493b3d00c75110169a366aabb5103df45da22",
        "outputs_continuous_state_v45/results/window_feature_raw_v45.csv": "5a6a20752a2132da62993ffd94575e2c2088c6021fad44adbcc7383eb8eae1ab",
    }
    for name, digest in expected.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
