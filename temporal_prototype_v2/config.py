from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TemporalPrototypeConfig:
    output_dir: str = "outputs_temporal_prototype_v2"
    z_table_path: str = "outputs_weighted_awrcore_v1/results/window_feature_z_table.csv"
    state_path: str = "outputs_aux_state_metrics_v2/window_state_scores_v2.csv"
    stable_plus_features: tuple[str, ...] = (
        "rs_corrdist_base", "rs_mean", "rs_absmean", "rs_q05", "rx_corrdist_base",
        "rs_rms", "ry_p2p", "rx_mean", "rx_absmean", "rx_q05",
    )
    # RS50_positive is derived from RS50 for gating, rather than duplicated in the 17-D encoder.
    encoder_dynamic_features: tuple[str, ...] = (
        "BDall_xy_v2", "BDshape_v2", "RS20", "RS50", "RS100", "TES", "BD_jump",
    )
    sequence_length: int = 20
    known_stop_interval_cycles: int = 500
    restart_guard_cycles: int = 50
    source_train_fraction: float = 0.70
    source_gap_windows: int = 4
    clip_abs_z: float = 12.0
    seeds: tuple[int, ...] = (20260712, 20260713, 20260714)
    epochs: int = 100
    patience: int = 15
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    prototype_eta_base: float = 0.01
    prototype_eta_max: float = 0.05
    teacher_decay: float = 0.995
    memory_per_state: int = 200
    min_memory_to_update: int = 20
    update_every_accepted: int = 20
    online_steps: int = 5
    online_learning_rate: float = 1e-4
    freeze_windows: int = 100
    checkpoint_interval: int = 200
    snapshot_fractions: tuple[float, ...] = tuple(x / 10 for x in range(11))
    confidence_threshold: float = 0.80
    js_threshold: float = 0.05
    posterior_margin_threshold: float = 0.20
    entropy_threshold: float = 1.00
    transition_stay: float = 0.970
    transition_forward: float = 0.025
    transition_backward: float = 0.005
    stage_health: tuple[float, ...] = (0.0, 0.2, 0.45, 0.7, 1.0)

    @property
    def input_features(self) -> tuple[str, ...]:
        return self.stable_plus_features + self.encoder_dynamic_features

    def paths(self) -> dict[str, Path]:
        root = Path(self.output_dir)
        paths = {name: root / name for name in (
            "configs", "source_models", "results", "diagnostics", "reports", "figures", "snapshots",
        )}
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        paths["root"] = root
        return paths

    def jsonable(self) -> dict:
        value = asdict(self)
        value["input_features"] = list(self.input_features)
        return value
