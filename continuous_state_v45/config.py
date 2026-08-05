from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Fixed before this run.  These are direct sensitive-phase force-ratio summaries,
# not values selected from stage labels or morphology outcomes.
CORE_FEATURES = (
    "rx_mean", "rx_absmean", "rx_q05",
    "ry_p2p",
    "rs_mean", "rs_q05", "rs_rms",
)
CORRDIST_FEATURES = ("rx_corrdist_base", "ry_corrdist_base", "rs_corrdist_base")
# Existing v4.4 ry summaries are audited as a group; no new ry family is invented in v4.5.
RY_EXTENSION_FEATURES = ("ry_mean", "ry_rms", "ry_std", "ry_q05", "ry_q95", "ry_corrdist_base")
STATE_METRICS = ("D_state", "V1000_norm", "A_state", "state_volatility")
METRIC_OUTPUT_NAMES = {"A_state": "multi_scale_rate_divergence"}


@dataclass(frozen=True)
class ContinuousStateV45Config:
    output_dir: str = "outputs_continuous_state_v45"
    raw_files: tuple[tuple[str, str], ...] = (
        ("Exp1", "Exp1_original_Fx_Fy_Fz_labels.csv"),
        ("Exp2", "Exp2_original_Fx_Fy_Fz_labels.csv"),
    )
    sensitive_phase: tuple[float, float] = (.45, .63)
    window_cycles: int = 20
    window_stride_cycles: int = 5
    baseline_cycles: int = 1000
    distance_form: str = "mahalanobis"
    velocity_short_cycles: int = 100
    velocity_long_cycles: int = 1000
    volatility_window_cycles: int = 500
    comparison_start_cycles: int = 2000
    high_value_quantile: float = .90
    trend_segments: int = 5
    cycle_mapping_config_path: str = "outputs_physical_validation_candidates_v1/configs/cycle_mapping_config.json"
    metadata_path: str = "metadata/knee_wear_experiment_metadata.json"
    v44_consensus_path: str = "outputs_continuous_state_v44/results/consensus_state_trajectories_v44.csv"
    eps: float = 1e-9

    def paths(self) -> dict[str, Path]:
        root = Path(self.output_dir)
        result = {name: root / name for name in ("configs", "results", "diagnostics", "figures", "reports")}
        result["root"] = root
        for path in result.values():
            path.mkdir(parents=True, exist_ok=True)
        return result

    def jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_files"] = dict(self.raw_files)
        payload["core_features"] = list(CORE_FEATURES)
        payload["corrdist_features"] = list(CORRDIST_FEATURES)
        payload["ry_extension_features"] = list(RY_EXTENSION_FEATURES)
        payload["state_metrics"] = list(STATE_METRICS)
        payload["code_version"] = "csv45"
        return payload
