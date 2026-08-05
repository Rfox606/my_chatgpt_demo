from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FEATURE_CONFIGS: dict[str, tuple[str, ...]] = {
    "F_core_v45": (
        "rx_mean", "rx_absmean", "rx_q05", "ry_mean", "ry_absmean", "ry_q05", "ry_p2p", "rs_mean", "rs_rms",
    ),
    "F_no_rs": ("rx_mean", "rx_absmean", "rx_q05", "ry_mean", "ry_absmean", "ry_q05", "ry_p2p"),
    "F_reduced_independent": ("rx_mean", "rx_q05", "ry_mean", "ry_q05", "ry_p2p", "rs_rms"),
}


@dataclass(frozen=True)
class MultiStageTrajectoryConfig:
    output_dir: str = "outputs_multistage_trajectory_state_v2"
    input_path: str = "outputs_continuous_state_v45/results/window_feature_raw_v45.csv"
    random_seed: int = 20260722
    feature_configs: tuple[str, ...] = ("F_core_v45", "F_no_rs", "F_reduced_independent")
    primary_feature_config: str = "F_core_v45"
    # Phase A: fixed before running the audit.
    audit_window_fractions: tuple[float, ...] = (0.10, 0.20, 0.30)
    audit_step_fraction: float = 0.025
    offline_smoothing_fraction: float = 0.02
    local_audit_alpha: float = 0.05
    local_bootstrap_replicates: int = 30
    bootstrap_block_fraction: float = 0.05
    rolling_ranker_block_fraction: float = 0.20
    rolling_ranker_step_fraction: float = 0.05
    rolling_ranker_max_pairs: int = 800
    rolling_ranker_c: float = 0.20
    rolling_ranker_persistent_blocks: int = 2
    recurrence_max_samples: int = 1200
    recurrence_neighbours: int = 8
    recurrence_exclusion_fraction: float = 0.05
    recurrence_null_replicates: int = 30
    cp_max_samples: int = 1200
    cp_min_segment_fraction: float = 0.05
    cp_pelt_penalties: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0)
    cp_binseg_counts: tuple[int, ...] = (1, 2, 3, 4)
    cp_bootstrap_replicates: int = 30
    cp_consensus_tolerance_fraction: float = 0.03
    # Phase B: intentionally no post-run tuning.
    adapter_l2: float = 0.15
    adapter_learning_rate: float = 0.08
    adapter_bounded_norm: float = 0.55
    adapter_bounded_step: float = 0.08
    adapter_unbounded_abort_norm: float = 100.0
    adapter_weak_l2_multiplier: float = 0.25
    adapter_prefix_fractions: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80)
    exp1_entry_cycles: tuple[float, ...] = (0.0, 8000.0, 16000.0, 24000.0, 32000.0)
    exp2_entry_cycles: tuple[float, ...] = (0.0, 3000.0, 6000.0, 9000.0)
    # Phase C: source-only topology, soft target adaptation.
    regime_k_candidates: tuple[int, ...] = (2, 3, 4, 5)
    regime_min_duration_fraction: float = 0.03
    regime_min_segment_count: int = 2
    regime_stickiness: float = 0.92
    regime_min_dwell_windows: int = 10
    regime_novelty_quantile: float = 0.995
    regime_adapter_l2: float = 0.20
    regime_adapter_learning_rate: float = 0.10
    regime_update_min_posterior: float = 0.75
    regime_update_max_novelty: float = 0.85
    regime_update_max_volatility: float = 4.0
    regime_bootstrap_replicates: int = 30

    def paths(self) -> dict[str, Path]:
        root = Path(self.output_dir)
        paths = {name: root / name for name in ("configs", "results", "diagnostics", "figures", "reports")}
        paths["root"] = root
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def jsonable(self) -> dict[str, Any]:
        value = asdict(self)
        value["feature_definitions"] = {name: list(FEATURE_CONFIGS[name]) for name in self.feature_configs}
        value["offline_diagnostic_only"] = True
        value["formal_outputs"] = [
            "regime_probability", "most_likely_regime", "regime_duration", "within_regime_progress",
            "activity_score", "trajectory_match_score", "novelty_score", "state_uncertainty",
        ]
        value["forbidden_model_inputs"] = ["Stage", "morphology", "wear_debris", "future_data", "offline_symmetric_smoothing"]
        value["global_monotonic_observation_assumed"] = False
        return value
