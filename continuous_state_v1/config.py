from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STABLE_PLUS_FEATURES = (
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
class ContinuousStateV1Config:
    output_dir: str = "outputs_continuous_state_v1"
    z_table_path: str = "outputs_weighted_awrcore_v1/results/window_feature_z_table.csv"
    baseline_cycles: int = 500
    known_stop_interval_cycles: int = 500
    restart_guard_cycles: int = 50
    stable_plus_features: tuple[str, ...] = STABLE_PLUS_FEATURES
    pair_gap_bins: tuple[tuple[int, int | None], ...] = (
        (500, 2000),
        (2000, 5000),
        (5000, None),
    )
    max_pairs_per_gap_bin: int = 20000
    pair_random_seed: int = 20260713
    source_train_fraction: float = 0.70
    source_gap_windows: int = 20
    rank_C_grid: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
    rank_solver: str = "liblinear"
    candidate_min_spacing_cycles: int = 500
    candidate_top_k_per_type: int = 20
    scientific_source_pair_auc_min: float = 0.60
    scientific_target_concordance_min: float = 0.55
    eps: float = 1e-9

    def paths(self) -> dict[str, Path]:
        root = Path(self.output_dir)
        paths = {
            "root": root,
            "configs": root / "configs",
            "results": root / "results",
            "diagnostics": root / "diagnostics",
            "figures": root / "figures",
            "reports": root / "reports",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["stable_plus_features"] = list(self.stable_plus_features)
        data["pair_gap_bins"] = [list(item) for item in self.pair_gap_bins]
        data["rank_C_grid"] = list(self.rank_C_grid)
        return data
