from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from multistage_trajectory_state_v2.adapter_ablation import _adapter_fit
from multistage_trajectory_state_v2.audit import _coefficient_runs, _fit_change_points, _recurrence_ratio, local_monotonicity_audit
from multistage_trajectory_state_v2.config import FEATURE_CONFIGS, MultiStageTrajectoryConfig
from multistage_trajectory_state_v2.data import assert_label_free
from multistage_trajectory_state_v2.evaluation import prefix_causality_check, synthetic_multistage_validation
from multistage_trajectory_state_v2.online_filter import run_online_filter
from multistage_trajectory_state_v2.regime_model import RegimeStructure
from multistage_trajectory_state_v2.segmentation import causal_descriptors


def _raw(rows: int = 160, dataset: str = "Exp1", pattern: np.ndarray | None = None) -> pd.DataFrame:
    cycle = np.arange(rows, dtype=float) * 100 + 50
    pattern = np.sin(cycle / 700) if pattern is None else pattern
    data: dict[str, object] = {"dataset": [dataset] * rows, "window_id": np.arange(rows), "window_index": np.arange(rows), "start_cycle": cycle - 49, "end_cycle": cycle + 49, "center_cycle": cycle}
    for index, feature in enumerate(FEATURE_CONFIGS["F_core_v45"]): data[feature] = pattern + .03 * index + .02 * np.cos(cycle / (211 + index * 11))
    return pd.DataFrame(data)


def _structure(columns: tuple[str, ...]) -> RegimeStructure:
    d = len(columns)
    return RegimeStructure(np.vstack((np.zeros(d), np.ones(d))), np.full((2, d), .1), np.array([[.92, .08], [.08, .92]]), np.array([20., 20.]), 2.0, 2, "unit_frozen", columns)


def test_label_morphology_debris_and_future_boundaries() -> None:
    with pytest.raises(AssertionError): assert_label_free(pd.DataFrame({"Stage": [1]}))
    with pytest.raises(AssertionError): assert_label_free(pd.DataFrame({"Sa": [1.]}))
    with pytest.raises(AssertionError): assert_label_free(pd.DataFrame({"wear_debris_count": [1]}))
    assert_label_free(_raw().iloc[:2])


def test_local_spearman_and_persistent_reversal_are_distinguished_from_noise() -> None:
    config = replace(MultiStageTrajectoryConfig(), audit_window_fractions=(.20,), audit_step_fraction=.10, local_bootstrap_replicates=2)
    cycle = np.arange(240, dtype=float) * 100; pattern = np.r_[np.arange(120), np.arange(120, 0, -1)]
    audit = local_monotonicity_audit(_raw(240, pattern=pattern), config)
    summary = audit.loc[(audit.row_type == "summary") & (audit.feature == "rx_mean")]
    assert (summary.PERSISTENT_DIRECTION_REVERSAL == "PASS").any()
    rng = np.random.default_rng(7); noisy = local_monotonicity_audit(_raw(240, pattern=rng.normal(size=240)), config)
    assert not (noisy.loc[(noisy.row_type == "summary") & (noisy.feature == "rx_mean"), "PERSISTENT_DIRECTION_REVERSAL"] == "PASS").any()


def test_rolling_ranker_reversal_and_far_time_neighbour_exclusion() -> None:
    assert _coefficient_runs([1, 1, -1, -1], 2)
    assert not _coefficient_runs([1, -1, 1, -1], 2)
    ratio, pairs = _recurrence_ratio(np.asarray([[0.0], [.1], [.05], [4.0]]), np.asarray([0., 1., 100., 101.]), 2, 20.)
    assert ratio > 0 and any(abs(left - right) >= 2 for left, right in pairs)


def test_change_point_bootstrap_core_and_adapter_numerical_safety() -> None:
    config = replace(MultiStageTrajectoryConfig(), cp_min_segment_fraction=.05)
    value = np.r_[np.zeros((80, 2)), np.ones((80, 2))]
    bkps, bic, setting = _fit_change_points(value, config, "Binseg_l2")
    assert np.isfinite(bic) and setting in config.cp_binseg_counts and any(abs(point - 80) < 25 for point in bkps[:-1])
    z = np.zeros((80, 3)); cycles = np.arange(80, dtype=float) * 600
    residual, trace, aborted, _ = _adapter_fit(z, cycles, np.ones(3), config, "Unbounded_L2", 1)
    assert not aborted and np.isfinite(residual).all() and all(item.get("numeric_abort", 0) == 0 for item in trace)


def test_constant_signal_freeze_and_prefix_causality() -> None:
    config = replace(MultiStageTrajectoryConfig(), regime_min_dwell_windows=3)
    raw = _raw(80, pattern=np.zeros(80)); descriptors, _, columns = causal_descriptors(raw, config); structure = _structure(columns)
    scores, _, runner = run_online_filter(descriptors, structure, config, adaptive=True, freeze_after_index=20)
    assert np.isfinite(scores.state_uncertainty).all() and runner.support.sum() <= 20
    assert prefix_causality_check(descriptors, structure, config)["status"] == "PASS"


def test_synthetic_multistage_spike_revisit_and_unknown_novel() -> None:
    result = synthetic_multistage_validation(MultiStageTrajectoryConfig())
    assert result["status"] == "PASS"
    assert result["short_spike_no_permanent_transition"] and result["state_revisit_retained"] and result["unknown_novel_triggered"]


def test_causal_descriptor_statistics_are_dataset_independent_and_no_duplicate_features() -> None:
    config = MultiStageTrajectoryConfig(); exp1 = _raw(90, "Exp1"); exp2 = _raw(90, "Exp2"); combined = pd.concat([exp1, exp2], ignore_index=True)
    full, _, columns = causal_descriptors(combined, config); changed = combined.copy(); changed.loc[changed.dataset.eq("Exp2"), list(FEATURE_CONFIGS["F_core_v45"])] += 999
    replay, _, _ = causal_descriptors(changed, config)
    left = full.loc[full.dataset.eq("Exp1"), list(columns)].to_numpy(float); right = replay.loc[replay.dataset.eq("Exp1"), list(columns)].to_numpy(float)
    assert np.allclose(left, right)
    assert "F_xy" not in FEATURE_CONFIGS and len({tuple(values) for values in FEATURE_CONFIGS.values()}) == len(FEATURE_CONFIGS)


def test_v45_and_ceap_v1_history_are_unchanged() -> None:
    def digest(path: str) -> str:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert digest("continuous_state_v45/state_engine.py") == "80beaca1f7a58c1aed0f5135cd5493b3d00c75110169a366aabb5103df45da22"
    assert digest("outputs_continuous_state_v45/results/window_feature_raw_v45.csv") == "5a6a20752a2132da62993ffd94575e2c2088c6021fad44adbcc7383eb8eae1ab"
    assert digest("cross_experiment_adaptive_state_v1/online.py") == "b0d4373039f3b3b1de8994c28012d2910b051dd7e3fbb40f6f7aa0ef13e5d78d"
