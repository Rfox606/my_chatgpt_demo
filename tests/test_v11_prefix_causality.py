import numpy as np
import pandas as pd

from adaptive_awr_v11.causal_metrics import build_metric_references
from adaptive_awr_v11.config import AdaptiveAWRV11Config
from adaptive_awr_v11.risk_head import RobustScaler, SoftRiskHead
from adaptive_awr_v11.target_calibration import fit_target_logit_alignment
from run_adaptive_cross_domain_awr_v11 import run_target_model, target_setup


def _target(config: AdaptiveAWRV11Config, count: int) -> pd.DataFrame:
    values = np.linspace(0.0, 2.0, count)
    frame = pd.DataFrame({feature: values for feature in config.stable_plus_features})
    frame["window_index"] = np.arange(count)
    frame["start_cycle"] = np.arange(count, dtype=float) + 1
    frame["end_cycle"] = np.arange(count, dtype=float) + 1
    frame["center_cycle"] = np.arange(count, dtype=float) + 1
    frame["BDall_xy_v2"] = values * 0.2
    frame["BDshape_v2"] = values * 0.1
    frame["is_restart_guard"] = 0
    frame["nearest_stop_boundary"] = 0.0
    frame["cycles_since_stop_boundary"] = 0.0
    return frame


def test_prefix_outputs_match_full_run() -> None:
    config = AdaptiveAWRV11Config(baseline_cycles=20, known_stop_interval_cycles=10000, safe_run_required=2)
    full_target = _target(config, 80)
    refs = build_metric_references(np.zeros(20), np.zeros(20), np.zeros(20), full_target.loc[:19, list(config.stable_plus_features)], config)
    head = SoftRiskHead(RobustScaler(np.zeros(5), np.ones(5)), np.array([-1.0, 0.3, 0.2, 0.1, 0.1, 0.1]), 1.0, True, "synthetic", 0.0)
    context = {
        "directions": {feature: 1 for feature in config.stable_plus_features}, "head": head,
        "source_awr_high": 1.0, "source_bd_high": 1.0, "source_tes_threshold": 3.0, "source_rs_threshold": 0.005,
        "source_early_logits": np.array([-1.0, 0.0, 1.0]), "source_tes_reference": np.zeros(20), "source_bd_jump_reference": np.zeros(20),
        "thresholds": {"watch_logit_threshold": -0.25, "high_logit_threshold": 0.5},
    }
    setup_full = target_setup(full_target, context, config)
    prefix = full_target.iloc[:50].copy()
    setup_prefix = target_setup(prefix, context, config)
    full_scores, full_rel, full_params, _ = run_target_model(full_target, "R5", context, setup_full, config)
    prefix_scores, prefix_rel, prefix_params, _ = run_target_model(prefix, "R5", context, setup_prefix, config)
    for column in ["AWR_adaptive", "RS50", "TES_clean", "aligned_logit", "final_risk", "residual_online_offset"]:
        assert np.array_equal(prefix_scores[column].to_numpy(), full_scores.loc[:49, column].to_numpy(), equal_nan=True)
    assert prefix_scores["adapter_state"].tolist() == full_scores.loc[:49, "adapter_state"].tolist()
    assert np.array_equal(prefix_rel["reliability_after"].to_numpy(), full_rel.iloc[: len(prefix_rel)]["reliability_after"].to_numpy())
    assert np.array_equal(prefix_params["adapter_state"].to_numpy(), full_params.loc[:49, "adapter_state"].to_numpy())
