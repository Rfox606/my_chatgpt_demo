from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GradualStateMonitoringV331Config:
    """Fixed V3.3.1 configuration; all online thresholds live here."""

    input_path: str = "outputs_continuous_state_v45/results/window_feature_raw_v45.csv"
    output_dir: str = "outputs_gradual_state_monitoring_v331"
    metadata_path: str = "metadata/knee_wear_experiment_metadata.json"
    cycle_mapping_path: str = "outputs_physical_validation_candidates_v1/configs/cycle_mapping_config.json"
    random_seed: int = 3311
    history_windows: int = 64
    descriptor_scaler_calibration_windows: int = 128
    descriptor_scaler_modes: tuple[str, str] = ("frozen_descriptor_scaler", "rolling_descriptor_scaler_v33")
    horizons: tuple[int, int, int] = (20, 100, 300)
    benchmark_training_window: int = 256
    benchmark_refit_interval: int = 64
    benchmark_min_train: int = 64
    ridge_alpha: float = 1.0
    rbf_components: int = 32
    rbf_gamma: float = 0.20
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
    model_divergence_weight: float = 0.15
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
    state_reuse_distance_threshold: float = 2.00
    state_library_update_rate: float = 0.20
    delayed_entry_offsets: tuple[int, int] = (64, 128)
    delayed_entry_warmup: int = 128
    delayed_state_agreement_threshold: float = 0.80
    delayed_transition_iou_threshold: float = 0.50
    delayed_local_state_difference_limit: int = 2
    base_window_width: int = 20
    base_window_stride: int = 5
    overlap_stride_multipliers: tuple[int, int, int] = (1, 2, 4)
    b2_transition_fraction_limit: float = 0.40
    b2_overlap50_agreement_min: float = 0.70
    b2_overlap0_agreement_min: float = 0.60
    boundary_guard_cycles: float = 250.0
    eps: float = 1e-9

    def paths(self) -> dict[str, Path]:
        root = Path(self.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        return {"root": root}

    def jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in ("descriptor_scaler_modes", "horizons", "delayed_entry_offsets", "overlap_stride_multipliers"):
            payload[name] = list(payload[name])
        payload["code_version"] = "gradual_state_monitoring_v331"
        payload["base_commit"] = "0900f577e50bd7e00e5c5cc515366c7a2180663f"
        payload["causality"] = (
            "Online descriptors, frozen/rolling scaler calculations, RLS updates, evidence baselines, "
            "state transitions, and state-library decisions use only the current and earlier rows."
        )
        payload["posthoc_only"] = ["cycle mapping segments", "dataset group boundaries", "morphology metadata"]
        return payload
