from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STABLE_PLUS_FEATURES = (
    "rs_corrdist_base", "rs_mean", "rs_absmean", "rs_q05", "rx_corrdist_base",
    "rs_rms", "ry_p2p", "rx_mean", "rx_absmean", "rx_q05",
)


@dataclass(frozen=True)
class ContinuousStateV2Config:
    output_dir: str = "outputs_continuous_state_v2"
    z_table_path: str = "outputs_weighted_awrcore_v1/results/window_feature_z_table.csv"
    baseline_cycles: int = 500
    known_stop_interval_cycles: int = 500
    restart_guard_cycles_grid: tuple[int, ...] = (50, 100, 150)
    primary_restart_guard_cycles: int = 100
    source_train_fraction: float = 0.70
    source_gap_windows: int = 20
    pair_gap_bins: tuple[tuple[int, int | None], ...] = ((500, 2000), (2000, 5000), (5000, None))
    max_pairs_per_gap_bin: int = 20000
    pair_random_seed: int = 20260713
    rank_C_grid: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
    correlation_prune_threshold: float = 0.98
    bootstrap_repeats: int = 200
    bootstrap_block_windows: int = 50
    common_sign_stability_min: float = 0.80
    common_min_median_abs_weight: float = 0.01
    middle_fraction: tuple[float, float] = (0.40, 0.60)
    terminal_fraction: tuple[float, float] = (0.80, 1.00)
    rs_horizons_windows: tuple[int, ...] = (20, 50, 100)
    tes_reference_cycles: int = 500
    weighted_oos_update_max: float = 0.35
    tes_update_max_quantile: float = 0.95
    adapter_learning_rate: float = 0.002
    adapter_learning_rate_min: float = 0.00005
    adapter_clip: float = 3.0
    baseline_replay_size: int = 50
    baseline_replay_p_drift_max: float = 0.10
    baseline_replay_bd_drift_max: float = 0.15
    forecast_history_windows: int = 20
    forecast_horizons_cycles: tuple[int, ...] = (100, 500, 1000)
    rls_forgetting_factor: float = 0.995
    rls_initial_covariance: float = 100.0
    random_seed: int = 20260713
    eps: float = 1e-9

    def paths(self) -> dict[str, Path]:
        root = Path(self.output_dir)
        result = {name: root / name for name in ("configs", "results", "diagnostics", "figures", "reports")}
        result["root"] = root
        for path in result.values():
            path.mkdir(parents=True, exist_ok=True)
        return result

    def jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("restart_guard_cycles_grid", "pair_gap_bins", "rank_C_grid", "rs_horizons_windows", "forecast_horizons_cycles"):
            data[key] = [list(x) if isinstance(x, tuple) else x for x in data[key]]
        data["stable_plus_features"] = list(STABLE_PLUS_FEATURES)
        return data
