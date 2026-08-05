from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BASE_FEATURES = (
    "rs_corrdist_base", "rs_mean", "rs_q05", "rx_corrdist_base", "rs_rms",
    "ry_p2p", "rx_mean", "rx_absmean", "rx_q05",
)
RY_EXTENSION_FEATURES = ("ry_mean", "ry_rms", "ry_std", "ry_q05", "ry_q95", "ry_corrdist_base")
EXTENDED_RY_FEATURES = tuple(dict.fromkeys((*BASE_FEATURES, *RY_EXTENSION_FEATURES)))
STATE_METRICS = ("D_state", "V500_norm", "V1000_norm", "A_state", "state_volatility")


@dataclass(frozen=True)
class ContinuousStateV44Config:
    output_dir: str = "outputs_continuous_state_v44"
    z_table_path: str = "outputs_weighted_awrcore_v1/results/window_feature_z_table.csv"
    metadata_path: str = "metadata/knee_wear_experiment_metadata.json"
    cycle_mapping_config_path: str = "outputs_physical_validation_candidates_v1/configs/cycle_mapping_config.json"
    baseline_cycles: int = 1000
    distance_form: str = "mahalanobis"
    velocity_windows_cycles: tuple[int, ...] = (100, 500, 1000)
    volatility_window_cycles: int = 500
    covariance_ridge: float = 1e-3
    consensus_emit_start_cycles: int = 1000
    high_value_quantile: float = .90
    trend_segments: int = 5
    eps: float = 1e-9

    def paths(self) -> dict[str, Path]:
        root = Path(self.output_dir)
        paths = {name: root / name for name in ("configs", "results", "diagnostics", "figures", "reports")}
        paths["root"] = root
        for path in paths.values(): path.mkdir(parents=True, exist_ok=True)
        return paths

    def jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["velocity_windows_cycles"] = list(self.velocity_windows_cycles)
        data["base_features"] = list(BASE_FEATURES)
        data["extended_ry_features"] = list(EXTENDED_RY_FEATURES)
        data["state_metrics"] = list(STATE_METRICS)
        data["code_version"] = "csv44"
        return data
