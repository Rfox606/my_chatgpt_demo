from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# The pre-registered independent main inputs.  Do not substitute absolute
# means or rs_mean: they are intentionally excluded from v3.2.
FEATURES = ("rx_mean", "rx_q05", "ry_mean", "ry_q05", "ry_p2p", "rs_rms")


@dataclass(frozen=True)
class V32Config:
    output_dir: str = "outputs_partial_shared_primitives_progression_v32"
    input_path: str = "outputs_continuous_state_v45/results/window_feature_raw_v45.csv"
    random_seed: int = 20260723
    features: tuple[str, ...] = FEATURES
    history_windows: int = 32
    horizons: tuple[int, ...] = (1, 5, 20)
    source_train_windows: int = 1600
    ridge_alpha: float = 0.25
    adapter_learning_rate: float = 0.06
    adapter_warmup_windows: int = 128
    negative_transfer_excess: float = 0.03
    negative_transfer_confirmations: int = 3

    # These BOCPD parameters are carried over unchanged from the locked v3.1
    # protocol.  v3.2 fixes the likelihood recursion, not the thresholds.
    bocpd_hazard: float = 1 / 160
    bocpd_max_run_length: int = 256
    bocpd_confirmation_posterior: float = 0.65
    bocpd_confirmation_windows: int = 3
    minimum_segment_windows: int = 16

    primitive_k_candidates: tuple[int, ...] = (2, 3, 4, 5, 6)
    minimum_cluster_fraction: float = 0.03
    shared_match_quality_threshold: float = 0.50
    private_descriptor_distance_threshold: float = 2.50
    private_state_min_support_for_novelty: int = 2
    persistence_windows: int = 3
    delayed_entries: tuple[tuple[str, tuple[float, ...]], ...] = (
        ("Exp1", (0.0, 8000.0, 16000.0, 24000.0, 32000.0)),
        ("Exp2", (0.0, 3000.0, 6000.0, 9000.0)),
    )
    delayed_common_arrived_windows: int = 200
    evaluation_prefixes: tuple[float, ...] = (0.10, 0.20, 0.40, 0.60, 0.80)

    def root(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def entries(self, dataset: str) -> tuple[float, ...]:
        return dict(self.delayed_entries)[dataset]

    def jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "version": "v3.2",
                "fixed_main_features": list(FEATURES),
                "forbidden_model_inputs": [
                    "Stage", "morphology", "wear_debris", "absolute_wear",
                    "target_final_length", "relative_complete_progress", "cycle",
                    "state_id", "rx_absmean", "ry_absmean", "rs_mean",
                ],
                "forbidden_operations": [
                    "joint_online_source_target_training", "single_window_primitives",
                    "fixed_target_k", "source_state_centre_transfer",
                    "state_id_alignment", "rolling_z_as_progression",
                ],
                "source_target_training": "strictly_asymmetric_per_direction",
                "source_model_frozen_after_fit": True,
                "predict_then_update": True,
                "continuous_score_independent_of_state_id": True,
            }
        )
        return payload
