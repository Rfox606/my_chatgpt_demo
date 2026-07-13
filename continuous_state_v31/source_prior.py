from __future__ import annotations

"""Source-only priors for the two prespecified transfer protocols."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ContinuousStateV31Config
from .data import assert_label_free
from .state_engine import PlateauPrior, derive_plateau_prior, run_target_state


@dataclass(frozen=True)
class SourceProtocolModel:
    protocol_id: str
    source_dataset: str
    features: tuple[str, ...]
    feature_strength: dict[str, float]
    plateau_prior: PlateauPrior
    source_states: pd.DataFrame
    severe_direction: np.ndarray | None
    source_exit_cycle: float


def _compact_source_states(states: pd.DataFrame) -> pd.DataFrame:
    """Retain exactly the causal source columns needed downstream.

    The full audit state is written for targets, not retained twice as a source-model
    implementation detail.  This prevents the two-protocol run from holding several
    redundant wide state tables simultaneously.
    """
    required = (
        "dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle", "baseline_window",
        "is_restart_guard", "crosses_stop_boundary", "nearest_stop_boundary", "nominal_stride_cycles",
        "D_state", "V20_norm", "V50_norm", "V100_norm", "direction_consistency", "A_state",
        "state_volatility_20", "state_volatility_50", "weighted_oos", "plateau_locked",
        "instability_score", "severe_direction_available", "S_severe_candidate",
    )
    columns = [column for column in (*required, *states.filter(regex=r"^(rs_|rx_|ry_)").columns) if column in states.columns]
    return states.loc[:, list(dict.fromkeys(columns))].copy()


def build_source_model(
    source: pd.DataFrame,
    features: tuple[str, ...],
    feature_strength: dict[str, float],
    protocol_id: str,
    allow_severe_prior: bool,
    config: ContinuousStateV31Config,
) -> tuple[SourceProtocolModel, pd.DataFrame, pd.DataFrame]:
    """Fit the source prior without target values or stage labels.

    Protocol A intentionally never supplies a severe direction.  Protocol B may do
    so only when the source stream itself causally locked a plateau and subsequently
    confirmed its exit.
    """
    assert_label_free(source)
    prior = derive_plateau_prior(source, features, config)
    source_states, plateau_events, exit_events, _, metadata = run_target_state(
        source,
        source,
        features,
        feature_strength,
        prior,
        None,
        f"{protocol_id}_SOURCE_ONLY",
        config,
    )
    source_exit = float(metadata["exit_cycle"])
    learned = metadata["severe_direction"]
    direction = (
        np.asarray(learned, dtype=float)
        if allow_severe_prior and learned is not None and np.isfinite(source_exit)
        else None
    )
    prior_row = {
        "protocol_id": protocol_id,
        "source_dataset": str(source.dataset.iloc[0]),
        "source_plateau_threshold_quantile": prior.quantile,
        "baseline_D_p95": prior.baseline_d_p95,
        "source_plateau_V50_threshold": prior.v50_threshold,
        "source_plateau_V100_threshold": prior.v100_threshold,
        "source_plateau_volatility_threshold": prior.volatility_threshold,
        "source_plateau_detected": int(not plateau_events.empty),
        "source_exit_detected": int(not exit_events.empty),
        "source_exit_cycle": source_exit,
    }
    if direction is None:
        reason = "EXP1_TO_EXP2_PROTOCOL_FORBIDS_SOURCE_SEVERE_PRIOR" if not allow_severe_prior else "SOURCE_DID_NOT_CAUSALLY_CONFIRM_PLATEAU_EXIT"
        severe_rows = [{
            "protocol_id": protocol_id,
            "source_dataset": str(source.dataset.iloc[0]),
            "source_severe_prior": "NONE",
            "feature_name": "",
            "weight": np.nan,
            "reason": reason,
        }]
    else:
        severe_rows = [{
            "protocol_id": protocol_id,
            "source_dataset": str(source.dataset.iloc[0]),
            "source_severe_prior": "AVAILABLE",
            "feature_name": feature,
            "weight": float(weight),
            "reason": "SOURCE_CAUSAL_PLATEAU_LOCK_THEN_EXIT",
        } for feature, weight in zip(features, direction, strict=True)]
    model = SourceProtocolModel(
        protocol_id=protocol_id,
        source_dataset=str(source.dataset.iloc[0]),
        features=features,
        feature_strength=feature_strength,
        plateau_prior=prior,
        source_states=_compact_source_states(source_states),
        severe_direction=direction,
        source_exit_cycle=source_exit,
    )
    return model, pd.DataFrame([prior_row]), pd.DataFrame(severe_rows)
