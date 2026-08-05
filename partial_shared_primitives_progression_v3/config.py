from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FEATURES = (
    "rx_mean", "rx_absmean", "rx_q05", "ry_mean", "ry_absmean", "ry_q05", "ry_p2p", "rs_mean", "rs_rms",
)


@dataclass(frozen=True)
class PartialSharedPrimitivesConfig:
    output_dir: str = "outputs_partial_shared_primitives_progression_v3"
    input_path: str = "outputs_continuous_state_v45/results/window_feature_raw_v45.csv"
    random_seed: int = 20260722
    feature_columns: tuple[str, ...] = FEATURES
    # Shared causal predictor: hyperparameters are fixed by the committed protocol.
    causal_context_windows: int = 32
    prediction_horizon_windows: int = 1
    representation_dimension: int = 6
    ridge_alpha: float = 0.20
    huber_delta: float = 1.35
    # The first observations are an explicitly declared calibration period, never selected using labels.
    primitive_calibration_windows: int = 256
    primitive_k_candidates: tuple[int, ...] = (2, 3, 4, 5, 6)
    primitive_min_fraction: float = 0.03
    primitive_bootstrap_replicates: int = 30
    state_calibration_windows: int = 512
    state_k_candidates: tuple[int, ...] = (2, 3, 4, 5, 6)
    state_min_fraction: float = 0.03
    state_stickiness: float = 0.85
    state_min_dwell_windows: int = 8
    continuous_residual_weight: float = 0.65
    continuous_activity_weight: float = 0.35
    continuous_calibration_window: int = 160
    continuous_minimum_history: int = 32
    prefix_cutoffs: tuple[float, ...] = (0.35, 0.60)

    def paths(self) -> dict[str, Path]:
        root = Path(self.output_dir)
        paths = {name: root / name for name in ("configs", "results", "diagnostics", "figures", "reports")}
        paths["root"] = root
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def jsonable(self) -> dict[str, Any]:
        result = asdict(self)
        result.update({
            "shared_only": ["causal_predictor_parameters", "dynamic_primitive_dictionary", "feature_definition"],
            "experiment_specific_only": ["state_k", "state_centres", "state_semantics", "state_path", "continuous_calibration"],
            "forbidden_model_inputs": ["Stage", "morphology", "wear_debris", "absolute_wear", "future_data", "global_time_ranking"],
            "forbidden_operations": ["equal_state_count", "state_id_alignment", "source_state_centres_in_target", "fixed_five_classification", "monotonic_state_constraint"],
            "continuous_score_independent_of_state_model": True,
        })
        return result

