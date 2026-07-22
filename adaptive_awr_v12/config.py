from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

STABLE_PLUS_FEATURES: Tuple[str, ...] = ("rs_corrdist_base", "rs_mean", "rs_absmean", "rs_q05", "rx_corrdist_base", "rs_rms", "ry_p2p", "rx_mean", "rx_absmean", "rx_q05")
SOFT_TARGET: Dict[int, float] = {1: 0.0, 2: 0.15, 3: 0.40, 4: 0.70, 5: 1.0}
SAMPLE_WEIGHT: Dict[int, float] = {1: 1.0, 2: 1.0, 3: 1.0, 4: 2.0, 5: 4.0}

@dataclass(frozen=True)
class AdaptiveAWRV12Config:
    output_dir: str = "outputs_adaptive_cross_domain_awr_v12"
    z_table_path: str = "outputs_weighted_awrcore_v1/results/window_feature_z_table.csv"
    state_v2_path: str = "outputs_aux_state_metrics_v2/window_state_scores_v2.csv"
    baseline_cycles: int = 500
    source_gap_windows: int = 4
    stable_plus_features: Tuple[str, ...] = STABLE_PLUS_FEATURES
    eps: float = 1e-9
    reliability_window: int = 50
    occupancy_window: int = 100
    z_clip_abs: float = 11.9
    known_stop_interval_cycles: int = 500
    restart_guard_cycles: int = 50
    restart_guard_cycles_grid: Tuple[int, ...] = (20, 50, 100)
    l2_grid: Tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 2.0)
    max_standardized_beta: float = 3.0
    max_effective_beta: float = 5.0
    beta0_bounds: Tuple[float, float] = (-10.0, 10.0)
    threshold_quantiles: int = 200
    stable_high_required_windows: int = 10
    random_seed: int = 20260712
    def paths(self) -> Dict[str, Path]:
        root = Path(self.output_dir); data={"root":root,"configs":root/"configs","results":root/"results","diagnostics":root/"diagnostics","reports":root/"reports","figures":root/"figures"}
        for path in data.values(): path.mkdir(parents=True, exist_ok=True)
        return data
    def jsonable(self) -> Dict[str, object]:
        payload=asdict(self); payload["stable_plus_features"]=list(self.stable_plus_features); payload["l2_grid"]=list(self.l2_grid); payload["restart_guard_cycles_grid"]=list(self.restart_guard_cycles_grid); return payload
