from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple


STABLE_PLUS_FEATURES: Tuple[str, ...] = (
    "rs_corrdist_base",
    "rs_mean",
    "rs_absmean",
    "rs_q05",
    "rx_corrdist_base",
    "rs_rms",
    "ry_p2p",
    "rx_mean",
    "rx_absmean",
    "rx_q05",
)

STAGE_SOFT_TARGET: Dict[int, float] = {1: 0.00, 2: 0.15, 3: 0.40, 4: 0.70, 5: 1.00}
STAGE_SAMPLE_WEIGHT: Dict[int, float] = {1: 1.0, 2: 1.0, 3: 1.0, 4: 2.0, 5: 4.0}


@dataclass(frozen=True)
class AdaptiveAWRV11Config:
    output_dir: str = "outputs_adaptive_cross_domain_awr_v11"
    z_table_path: str = "outputs_weighted_awrcore_v1/results/window_feature_z_table.csv"
    state_v2_path: str = "outputs_aux_state_metrics_v2/window_state_scores_v2.csv"
    baseline_cycles: int = 500
    window_k: int = 20
    stride: int = 5
    source_gap_windows: int = 4
    stable_plus_features: Tuple[str, ...] = STABLE_PLUS_FEATURES
    eps: float = 1e-9
    z_clip_abs: float = 11.9
    reliability_window: int = 50
    reliability_min: float = 0.50
    reliability_max: float = 1.00
    reliability_max_down_step: float = 0.02
    reliability_max_up_step: float = 0.01
    occupancy_window: int = 100
    rs_horizons: Tuple[int, int, int] = (20, 50, 100)
    risk_alpha_up: float = 0.15
    risk_alpha_down: float = 0.02
    event_risk_weight: float = 0.75
    l2_grid: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0)
    beta0_bounds: Tuple[float, float] = (-10.0, 10.0)
    beta_bounds: Tuple[float, float] = (0.0, 5.0)
    known_stop_interval_cycles: int = 500
    restart_guard_cycles: int = 100
    safe_run_required: int = 10
    gate_history_windows: int = 20
    event_freeze_windows: int = 100
    risk_freeze_windows: int = 100
    cooldown_windows: int = 200
    checkpoint_interval: int = 200
    rollback_eval_windows: int = 50
    rollback_risk_drop: float = 0.15
    rollback_extension_min_windows: int = 20
    online_offset_eta: float = 0.002
    online_offset_bounds: Tuple[float, float] = (-0.5, 0.5)
    online_update_interval: int = 20
    source_tes_floor: float = 3.0
    source_rs_floor: float = 0.005
    source_high_percentile: float = 95.0
    random_seed: int = 20260712

    def output_paths(self) -> Dict[str, Path]:
        root = Path(self.output_dir)
        paths = {
            "root": root,
            "configs": root / "configs",
            "results": root / "results",
            "diagnostics": root / "diagnostics",
            "reports": root / "reports",
            "figures": root / "figures",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def as_jsonable(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["stable_plus_features"] = list(self.stable_plus_features)
        payload["l2_grid"] = list(self.l2_grid)
        payload["rs_horizons"] = list(self.rs_horizons)
        return payload
