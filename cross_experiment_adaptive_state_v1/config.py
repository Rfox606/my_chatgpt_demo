from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# These configurations are fixed before inspecting the target results.  They contain
# only direct force-ratio window summaries, never Stage, morphology, or debris fields.
FEATURE_CONFIGS: dict[str, tuple[str, ...]] = {
    "F_xy": (
        "rx_mean", "rx_absmean", "rx_q05",
        "ry_mean", "ry_absmean", "ry_q05", "ry_p2p",
    ),
    "F_core_v45": (
        "rx_mean", "rx_absmean", "rx_q05",
        "ry_mean", "ry_absmean", "ry_q05", "ry_p2p",
        "rs_mean", "rs_rms",
    ),
    "F_no_rs": (
        "rx_mean", "rx_absmean", "rx_q05",
        "ry_mean", "ry_absmean", "ry_q05", "ry_p2p",
    ),
}
PRIMARY_FEATURE_CONFIG = "F_core_v45"


@dataclass(frozen=True)
class CrossExperimentAdaptiveConfig:
    output_dir: str = "outputs_cross_experiment_adaptive_state_v1"
    input_path: str = "outputs_continuous_state_v45/results/window_feature_raw_v45.csv"
    feature_configs: tuple[str, ...] = ("F_xy", "F_core_v45", "F_no_rs")
    primary_feature_config: str = PRIMARY_FEATURE_CONFIG
    source_gap_bins: tuple[tuple[float, float], ...] = ((500.0, 1000.0), (1000.0, 3000.0), (3000.0, 5000.0))
    source_max_pairs_per_gap_bin: int = 800
    source_embargo_cycles: float = 500.0
    source_train_fraction: float = 0.65
    rank_c_values: tuple[float, ...] = (0.05, 0.2, 1.0)
    target_initialization_cycles: float = 1000.0
    target_update_interval_cycles: float = 500.0
    target_update_pair_limit: int = 500
    target_l2: float = 0.15
    target_learning_rate: float = 0.08
    adapter_max_norm: float = 0.55
    adapter_max_step_norm: float = 0.08
    lambda_max: float = 0.45
    lambda_ramp_cycles: float = 5000.0
    ood_quantile: float = 0.95
    high_volatility_gate: float = 5.0
    random_seed: int = 20260721
    exp1_entry_cycles: tuple[float, ...] = (0.0, 8000.0, 16000.0, 24000.0, 32000.0)
    exp2_entry_cycles: tuple[float, ...] = (0.0, 3000.0, 6000.0, 9000.0)

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
        value["formal_outputs"] = ["progression_score", "activity_score", "state_uncertainty"]
        value["forbidden_online_inputs"] = ["target Stage1to5", "morphology", "wear debris count", "future target data"]
        value["cycle_is_model_input"] = False
        value["code_version"] = "ceap_v1"
        return value
