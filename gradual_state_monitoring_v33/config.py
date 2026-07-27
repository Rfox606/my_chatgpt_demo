from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GradualStateMonitoringV33Config:
    """All V3.3 thresholds are fixed here before an online run begins."""

    input_path: str = "outputs_continuous_state_v45/results/window_feature_raw_v45.csv"
    output_dir: str = "outputs_gradual_state_monitoring_v33"
    random_seed: int = 3303
    history_windows: int = 64
    horizons: tuple[int, int, int] = (20, 100, 300)
    benchmark_training_window: int = 256
    benchmark_refit_interval: int = 64
    benchmark_min_train: int = 64
    ridge_alpha: float = 1.0
    rbf_components: int = 32
    rbf_gamma: float = 0.20
    source_initialisation_enabled: bool = True
    source_initialisation_training_window: int = 512
    source_initialisation_early_windows: int = 512
    dual_horizon: int = 20
    rls_initial_covariance: float = 20.0
    rls_min_updates: int = 8
    slow_forgetting_factor: float = 0.998
    fast_forgetting_factor: float = 0.960
    evidence_baseline_windows: int = 128
    evidence_min_history: int = 32
    evidence_baseline_refresh: int = 8
    evidence_weights: dict[str, float] = field(default_factory=lambda: {
        "prediction_surprise": 0.40,
        "distribution_shift": 0.30,
        "drift_velocity": 0.30,
    })
    surprise_floor: float = 0.0
    transition_evidence_threshold: float = 5.00
    transition_persistence: int = 5
    recovery_evidence_threshold: float = 0.85
    recovery_persistence: int = 3
    transition_velocity_threshold: float = 4.00
    stable_velocity_threshold: float = 1.80
    new_state_distance_threshold: float = 2.00
    return_distance_threshold: float = 1.15
    new_stable_persistence: int = 10
    new_stable_hold: int = 16
    temporary_hold: int = 6
    temporary_max_duration: int = 48
    stable_reference_learning_rate: float = 0.05
    delayed_entry_offsets: tuple[int, int] = (64, 128)
    delayed_entry_warmup: int = 128
    delayed_convergence_threshold: float = 0.75
    base_window_width: int = 20
    base_window_stride: int = 5
    overlap_stride_multipliers: tuple[int, int, int] = (1, 2, 4)
    stable_false_transition_limit: int = 1
    real_switch_limit: int = 80
    eps: float = 1e-9

    def paths(self) -> dict[str, Path]:
        root = Path(self.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        return {"root": root}

    def jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["horizons"] = list(self.horizons)
        payload["delayed_entry_offsets"] = list(self.delayed_entry_offsets)
        payload["overlap_stride_multipliers"] = list(self.overlap_stride_multipliers)
        payload["code_version"] = "gradual_state_monitoring_v33"
        payload["causality"] = (
            "Descriptors, normalisation, evidence baselines, model updates, and state "
            "decisions use the current row and prior rows only."
        )
        payload["forbidden_online_inputs"] = [
            "Stage", "cycle_fraction_of_total", "target_final_length", "future_window",
            "morphology", "wear_debris", "state_label", "whole_sequence_statistics",
        ]
        return payload
