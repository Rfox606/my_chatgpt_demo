from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class OrderedRegimeConfig:
    output_dir: str = "outputs_ordered_regime_v21"
    z_table_path: str = "outputs_weighted_awrcore_v1/results/window_feature_z_table.csv"
    state_path: str = "outputs_aux_state_metrics_v2/window_state_scores_v2.csv"
    v2_model_dir: str = "outputs_temporal_prototype_v2/source_models"
    v12_scores_path: str = "outputs_adaptive_cross_domain_awr_v12/results/adaptive_window_scores_v12.csv"
    stable_plus_features: tuple[str, ...] = (
        "rs_corrdist_base", "rs_mean", "rs_absmean", "rs_q05", "rx_corrdist_base",
        "rs_rms", "ry_p2p", "rx_mean", "rx_absmean", "rx_q05",
    )
    dynamic_features: tuple[str, ...] = ("BDall_xy_v2", "BDshape_v2", "RS20", "RS50", "RS100", "TES", "BD_jump")
    sequence_length: int = 20
    known_stop_interval_cycles: int = 500
    restart_guard_cycles: int = 50
    post_restart_cooldown_windows: int = 10
    source_train_fraction: float = 0.70
    source_gap_windows: int = 4
    memory_per_state: int = 300
    minimum_state_support: int = 30
    prototype_recompute_every: int = 10
    candidate_gap_tolerance: int = 5
    candidate_purity_min: float = 0.80
    candidate_radius_multiplier: float = 1.25
    candidate_separation_multiplier: float = 1.20
    stable_required_windows: int = 10
    change_threshold_grid: tuple[float, ...] = (2.0, 2.5, 3.0)
    candidate_min_windows_grid: tuple[int, ...] = (15, 25, 40)
    min_dwell_windows_grid: tuple[int, ...] = (50, 100, 200)
    random_seed: int = 20260713

    @property
    def input_features(self) -> tuple[str, ...]:
        return self.stable_plus_features + self.dynamic_features

    def paths(self) -> dict[str, Path]:
        root = Path(self.output_dir)
        paths = {name: root / name for name in ("configs", "source", "results", "diagnostics", "snapshots", "figures", "reports")}
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        paths["root"] = root
        return paths

    def jsonable(self) -> dict:
        payload = asdict(self)
        payload["input_features"] = list(self.input_features)
        return payload
