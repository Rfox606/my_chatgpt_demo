from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FEATURES = ("rx_mean", "rx_absmean", "rx_q05", "ry_mean", "ry_absmean", "ry_q05", "ry_p2p", "rs_mean", "rs_rms")


@dataclass(frozen=True)
class V31Config:
    output_dir: str = "outputs_partial_shared_primitives_progression_v31"
    input_path: str = "outputs_continuous_state_v45/results/window_feature_raw_v45.csv"
    random_seed: int = 20260722
    features: tuple[str, ...] = FEATURES
    history_windows: int = 32
    horizons: tuple[int, ...] = (1, 4, 16)
    source_train_windows: int = 1600
    ridge_alpha: float = .25
    adapter_learning_rate: float = .06
    adapter_warmup_windows: int = 128
    negative_transfer_excess: float = .03
    negative_transfer_confirmations: int = 3
    bocpd_hazard: float = 1 / 160
    bocpd_max_run_length: int = 256
    bocpd_confirmation_posterior: float = .65
    bocpd_confirmation_windows: int = 3
    minimum_segment_windows: int = 16
    primitive_k_candidates: tuple[int, ...] = (2, 3, 4, 5, 6)
    private_state_k_candidates: tuple[int, ...] = (2, 3, 4, 5, 6)
    minimum_cluster_fraction: float = .03
    private_state_calibration_confirmed_segments: int = 6
    source_prior_predictable_windows: int = 400
    increment_innovation_weight: float = .65
    increment_activity_weight: float = .35
    delayed_entries: tuple[tuple[str, tuple[float, ...]], ...] = (("Exp1", (0., 8000., 16000., 24000.)), ("Exp2", (0., 3000., 6000., 9000.)))
    delayed_common_arrived_windows: int = 200
    # Fixed physical cycle cutoffs are an audit-only intervention.  They do
    # not derive from a target run's final length or relative completion.
    prefix_cutoff_cycles: tuple[float, ...] = (3000., 9000.)

    def paths(self) -> dict[str, Path]:
        root = Path(self.output_dir); paths = {name: root / name for name in ("configs", "results", "diagnostics", "figures", "reports")}; paths["root"] = root
        for path in paths.values(): path.mkdir(parents=True, exist_ok=True)
        return paths

    def entries(self, dataset: str) -> tuple[float, ...]:
        return dict(self.delayed_entries)[dataset]

    def jsonable(self) -> dict[str, Any]:
        result = asdict(self)
        result.update({
            "forbidden_model_inputs": ["Stage", "morphology", "wear_debris", "absolute_wear", "future_target", "target_final_length", "relative_complete_progress"],
            "forbidden_operations": ["joint_online_source_target_training", "online_rotating_svd", "single_window_primitives", "fixed_target_k", "state_id_alignment", "source_state_centre_transfer", "rolling_z_as_progression"],
            "source_target_training": "strictly_asymmetric_per_direction",
            "continuous_score_independent_of_state_id": True,
        })
        return result
