from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FEATURES = (
    "rs_corrdist_base", "rs_mean", "rs_q05", "rx_corrdist_base", "rs_rms",
    "ry_p2p", "rx_mean", "rx_absmean", "rx_q05",
)
FEATURE_GROUPS = {
    "rs": tuple(name for name in FEATURES if name.startswith("rs_")),
    "rx": tuple(name for name in FEATURES if name.startswith("rx_")),
    "ry": tuple(name for name in FEATURES if name.startswith("ry_")),
}


@dataclass(frozen=True)
class ContinuousStateV43Config:
    """Pre-registered, label-free parameters for actual-cycle deconfounded v4.3."""

    output_dir: str = "outputs_continuous_state_v43"
    z_table_path: str = "outputs_weighted_awrcore_v1/results/window_feature_z_table.csv"
    baseline_cycles: int = 1000
    restart_guard_cycles: int = 100
    distance_form: str = "mahalanobis"
    velocity_windows_cycles: tuple[int, ...] = (100, 500, 1000)
    volatility_window_cycles: int = 500
    evidence_confirm_cycles: int = 100
    evidence_reset_cycles: int = 50
    low_activity_confirm_cycles: int = 500
    low_activity_release_cycles: int = 250
    residual_ewma_alpha: float = .02
    abrupt_cusum_threshold: float = 5.0
    consensus_emit_start_cycles: int = 1000
    consensus_support_min: float = .50
    consensus_score_min: float = 1.0
    episode_merge_gap_cycles: int = 250
    episode_min_cycles: int = 100
    episode_coordinate: str = "actual"
    episode_split_min_actual_cycles: int = 500
    episode_split_valley_fraction: float = 0.75
    episode_split_support_decline: float = 0.50
    episode_split_persistence_cycles: int = 100
    stop_deconfounding_half_widths_actual: tuple[int, ...] = (100, 200)
    cycle_mapping_config_path: str = "outputs_physical_validation_candidates_v1/configs/cycle_mapping_config.json"
    morphology_anchor_actual_cycles: tuple[int, ...] = (0, 8000, 16000, 24000, 32000, 40000, 48000)
    covariance_ridge: float = 1e-3
    source_train_fraction: float = .70
    source_gap_windows: int = 20
    pair_gap_bins: tuple[tuple[int, int | None], ...] = ((500, 2000), (2000, 5000), (5000, None))
    max_pairs_per_gap_bin: int = 20000
    correlation_prune_threshold: float = .98
    forecast_history_windows: int = 20
    forecast_horizons_cycles: tuple[int, ...] = (100, 500, 1000)
    forecast_issue_stride_cycles: int = 50
    forecast_rolling_window_cycles: int = 5000
    safe_gate_min_observations: int = 20
    rls_forgetting_factor: float = .999
    rls_initial_covariance: float = 100.0
    rls_gain_norm_max: float = .10
    rls_theta_norm_max: float = 20.0
    rls_covariance_max: float = 1e4
    forecast_delta_clip: float = 30.0
    random_seed: int = 20260719
    eps: float = 1e-9

    def paths(self) -> dict[str, Path]:
        root = Path(self.output_dir)
        paths = {name: root / name for name in ("configs", "results", "diagnostics", "figures", "reports")}
        paths["root"] = root
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("pair_gap_bins", "forecast_horizons_cycles", "velocity_windows_cycles"):
            data[key] = [list(value) if isinstance(value, tuple) else value for value in data[key]]
        data["features"] = list(FEATURES)
        data["feature_groups"] = {key: list(value) for key, value in FEATURE_GROUPS.items()}
        data["code_version"] = "csv43"
        return data
