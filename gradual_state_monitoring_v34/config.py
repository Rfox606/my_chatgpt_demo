from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GradualStateMonitoringV34Config:
    """Predeclared V3.4 protocol.

    All thresholds and model choices live here so target Stage values cannot select
    an online decision, an adaptation update, or a candidate event.
    """

    input_path: str = "outputs_continuous_state_v45/results/window_feature_raw_v45.csv"
    cycle_mapping_path: str = "outputs_physical_validation_candidates_v1/configs/cycle_mapping_config.json"
    output_dir: str = "outputs_gradual_state_monitoring_v34"
    random_seeds: tuple[int, ...] = (3401, 3402, 3403, 3404, 3405)
    calibration_windows: int = 128
    history_windows: int = 128
    embedding_dim: int = 16
    adapter_hidden_dim: int = 16
    tcn_channels: tuple[int, ...] = (32, 32, 32, 16, 16, 16, 16)
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    kernel_size: int = 2
    dropout: float = 0.10
    source_prediction_horizon: int = 20
    source_train_fraction: float = 0.80
    source_max_training_examples: int = 1024
    source_epochs: int = 2
    upper_bound_epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 1e-3
    adapter_learning_rate: float = 5e-4
    source_boundary_sigma_actual_cycles: float = 250.0
    source_boundary_support_actual_cycles: float = 1000.0
    loss_boundary_weight: float = 1.0
    loss_rank_weight: float = 0.3
    loss_consistency_weight: float = 0.2
    loss_prediction_weight: float = 0.2
    score_lags: tuple[int, int, int] = (16, 64, 128)
    score_weights: tuple[float, float, float] = (0.25, 0.35, 0.40)
    source_probability_weight: float = 0.5
    score_baseline_quantile: float = 0.95
    score_minimum_scale: float = 1e-5
    candidate_min_windows: int = 3
    candidate_merge_gap_windows: int = 10
    candidate_cooldown_windows: int = 40
    candidate_min_score: float = 0.0
    adaptation_update_interval: int = 1
    teacher_ema_decay: float = 0.99
    target_normalizer_max_history: int = 2048
    target_score_max_history: int = 2048
    evaluation_tolerances_actual_cycles: tuple[float, ...] = (100.0, 250.0, 500.0, 1000.0)
    proximity_limit_actual_cycles: float = 1000.0
    device: str = "cpu"
    eps: float = 1e-9
    online_allowed_columns: tuple[str, ...] = (
        "dataset", "window_index", "center_cycle",
        "rx_mean", "rx_q05", "ry_mean", "ry_q05", "ry_p2p", "rs_rms",
    )
    target_forbidden_columns: tuple[str, ...] = field(default_factory=lambda: (
        "stage", "stage_label", "stage1to5", "stage_boundary", "boundary",
        "morphology", "sa", "sq", "sz", "wear", "wear_debris",
        "target_final_length", "cycle_fraction", "total_length", "test_statistics",
    ))

    @property
    def receptive_field(self) -> int:
        # Each residual block has two causal convolutions.  This is deliberately
        # reported and tested rather than inferred from an implementation detail.
        return 1 + 2 * (self.kernel_size - 1) * sum(self.tcn_dilations)

    def paths(self) -> dict[str, Path]:
        root = Path(self.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        return {"root": root}

    def jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "random_seeds", "tcn_channels", "tcn_dilations", "score_lags",
            "score_weights", "evaluation_tolerances_actual_cycles", "online_allowed_columns",
            "target_forbidden_columns",
        ):
            payload[key] = list(payload[key])
        payload.update({
            "code_version": "gradual_state_monitoring_v34",
            "base_commit": "62f75ae7502a65fb1bcb76b973bdc9e4f05d10d9",
            "tcn_effective_receptive_field_windows": self.receptive_field,
            "causality_contract": (
                "Target online execution accepts only the current and preceding force windows. "
                "Target Stage, boundaries, total length, future windows, morphology, wear, and "
                "test-set aggregates are prohibited until frozen outputs enter offline evaluation."
            ),
            "target_label_protocol": "Target Stage is loaded only after every label-free online run has frozen.",
        })
        return payload
