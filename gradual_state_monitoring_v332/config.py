from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GradualStateMonitoringV332Config:
    """Predeclared V3.3.2 protocol.  Stage data is evaluation-only."""

    input_path: str = "outputs_continuous_state_v45/results/window_feature_raw_v45.csv"
    v331_output_dir: str = "outputs_gradual_state_monitoring_v331"
    output_dir: str = "outputs_gradual_state_monitoring_v332"
    cycle_mapping_path: str = "outputs_physical_validation_candidates_v1/configs/cycle_mapping_config.json"
    random_seed: int = 3322
    feature_history: int = 128
    scaler_calibration_windows: int = 128
    horizons: tuple[int, int, int] = (20, 100, 300)
    within_train_fraction: float = 0.40
    min_training_examples: int = 256
    max_training_examples: int = 1024
    ridge_alpha: float = 1.0
    rbf_components: int = 32
    rbf_gamma: float = 0.20
    tcn_channels: tuple[int, int, int, int] = (32, 32, 16, 16)
    tcn_kernel_size: int = 3
    tcn_dilations: tuple[int, int, int, int] = (1, 2, 4, 8)
    tcn_dropout: float = 0.10
    gru_hidden_size: int = 32
    deep_epochs: int = 3
    deep_batch_size: int = 64
    deep_learning_rate: float = 0.001
    huber_delta: float = 1.0
    stage_buffer_actual_cycles: float = 250.0
    boundary_tolerances_actual_cycles: tuple[float, float, float] = (100.0, 250.0, 500.0)
    primary_boundary_tolerance_actual_cycles: float = 250.0
    state_models: tuple[str, str, str, str, str] = (
        "Persistence", "Ridge", "RBF_Ridge", "TCN", "GRU",
    )
    gate_boundary_recall_min: float = 0.75
    gate_boundary_precision_min: float = 0.50
    gate_stage_stable_rate_min: float = 0.70
    gate_false_switches_per_1000_max: float = 5.0
    gate_false_alarm_reduction_min: float = 0.20
    eps: float = 1e-9
    label_prohibited_online_fields: tuple[str, ...] = field(default_factory=lambda: (
        "stage", "stage_label", "stage1to5", "morphology", "sa", "sq", "sz",
        "wear_debris", "target_final_length", "cycle_fraction",
    ))

    def paths(self) -> dict[str, Path]:
        root = Path(self.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        return {"root": root}

    def jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        for field_name in (
            "horizons", "tcn_channels", "tcn_dilations", "boundary_tolerances_actual_cycles",
            "state_models", "label_prohibited_online_fields",
        ):
            payload[field_name] = list(payload[field_name])
        payload.update({
            "code_version": "gradual_state_monitoring_v332",
            "base_commit": "6162244be47f7c095924e34a1794a698cfa75428",
            "online_protocol": (
                "Only the six raw force features and the observed 128-window history reach forecasting, "
                "evidence, or the state machine. Prediction targets are available only when their horizon "
                "has elapsed; Stage is joined only after all online outputs are frozen."
            ),
            "label_protocol": (
                "Stage boundaries are read verbatim from the pre-existing cycle_mapping_config.json for "
                "post-hoc evaluation and plotting only; they never select a model, threshold, or training row."
            ),
        })
        return payload
