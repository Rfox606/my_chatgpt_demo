from __future__ import annotations

from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True)
class AdaptiveAWRConfig:
    """Fixed configuration for the first adaptive, source-only AWR protocol."""

    output_dir: str = "outputs_adaptive_cross_domain_awr_v1"
    z_table_path: str = "outputs_weighted_awrcore_v1/results/window_feature_z_table.csv"
    state_v2_path: str = "outputs_aux_state_metrics_v2/window_state_scores_v2.csv"
    baseline_cycles: int = 500
    window_k: int = 20
    stride: int = 5
    stable_plus_features: Tuple[str, ...] = STABLE_PLUS_FEATURES
    source_gap_windows: int = 4
    eps: float = 1e-9
    reliability_window: int = 50
    reliability_ewma: float = 0.05
    reliability_min: float = 0.25
    reliability_max: float = 1.0
    saturation_abs_z: float = 11.9
    occupancy_window: int = 100
    positive_class_weight: float = 4.0
    risk_head_l2: float = 0.01
    risk_alpha_up: float = 0.15
    risk_alpha_down: float = 0.02
    event_risk_weight: float = 0.75
    online_offset_eta: float = 0.005
    online_offset_min: float = -1.0
    online_offset_max: float = 1.0
    online_update_interval: int = 20
    event_freeze_windows: int = 100
    rollback_freeze_windows: int = 200
    checkpoint_interval: int = 200
    rollback_eval_windows: int = 50
    rollback_risk_drop: float = 0.15
    gate_history_windows: int = 20
    target_safe_risk: float = 0.05
    source_validation_fraction: float = 0.30
    source_tes_floor: float = 3.0
    source_rs_floor: float = 0.005
    source_high_percentile: float = 95.0
    stable_high_windows: int = 3
    random_seed: int = 20260712
    raw_files: Dict[str, str] = field(
        default_factory=lambda: {
            "Exp1": "Exp1_original_Fx_Fy_Fz_labels.csv",
            "Exp2": "Exp2_original_Fx_Fy_Fz_labels.csv",
        }
    )

    def as_jsonable(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["stable_plus_features"] = list(self.stable_plus_features)
        return payload

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
