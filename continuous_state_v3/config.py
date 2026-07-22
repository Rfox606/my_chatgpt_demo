from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FEATURES = (
    "rs_corrdist_base", "rs_mean", "rs_absmean", "rs_q05", "rx_corrdist_base",
    "rs_rms", "ry_p2p", "rx_mean", "rx_absmean", "rx_q05",
)


@dataclass(frozen=True)
class ContinuousStateV3Config:
    output_dir: str = "outputs_continuous_state_v3"
    z_table_path: str = "outputs_weighted_awrcore_v1/results/window_feature_z_table.csv"
    baseline_cycles: int = 500
    known_stop_interval_cycles: int = 500
    restart_guard_cycles: int = 100
    source_train_fraction: float = .70
    source_gap_windows: int = 20
    pair_gap_bins: tuple[tuple[int, int | None], ...] = ((500, 2000), (2000, 5000), (5000, None))
    max_pairs_per_gap_bin: int = 20000
    correlation_prune_threshold: float = .98
    random_seed: int = 20260713
    plateau_min_cycle: int = 2000
    plateau_candidate_cycles: int = 300
    plateau_lock_cycles: int = 500
    plateau_reference_cycles: int = 500
    plateau_exit_candidate_cycles: int = 300
    plateau_exit_confirm_cycles: int = 500
    target_clip: float = 8.0
    source_plateau_threshold_quantile: float = .75
    weighted_oos_max: float = .35
    severe_eta: float = .01
    severe_direction_consistency_min: float = .60
    severe_direction_cosine_min: float = .70
    severe_score_threshold: float = 2.0
    forecast_history_windows: int = 20
    forecast_horizons_cycles: tuple[int, ...] = (100, 500, 1000)
    rls_forgetting_factor: float = .999
    rls_initial_covariance: float = 100.0
    rls_gain_norm_max: float = .10
    ensemble_window: int = 200
    ensemble_alpha_step: float = .05
    ensemble_freeze_cycles: int = 500
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
        for key in ("pair_gap_bins", "forecast_horizons_cycles"):
            data[key] = [list(value) if isinstance(value, tuple) else value for value in data[key]]
        data["features"] = list(FEATURES)
        return data
