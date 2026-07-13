from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ContinuousStateV3Config
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


def build_source_model(source: pd.DataFrame, features: tuple[str, ...], feature_strength: dict[str, float], protocol_id: str, allow_severe_prior: bool, config: ContinuousStateV3Config) -> tuple[SourceProtocolModel, pd.DataFrame, pd.DataFrame]:
    """Build source-only priors. No target data or stage labels enters this function."""
    assert_label_free(source)
    prior, _, _, _ = derive_plateau_prior(source, features, config)
    source_states, plateau_events, exit_events, updates, metadata = run_target_state(source, source, features, feature_strength, prior, None, f"{protocol_id}_SOURCE_ONLY", config)
    direction = np.asarray(metadata["severe_direction"], float) if allow_severe_prior and metadata["severe_direction"] is not None else None
    severe_audit_rows = []
    if direction is None:
        severe_audit_rows.append({"protocol_id": protocol_id, "source_dataset": source.dataset.iloc[0], "source_severe_prior": "NONE", "feature_name": "", "weight": np.nan, "reason": "EXP1_NO_SEVERE_PRIOR" if not allow_severe_prior else "NO_CAUSAL_SOURCE_PLATEAU_EXIT"})
    else:
        for name, weight in zip(features, direction, strict=True):
            severe_audit_rows.append({"protocol_id": protocol_id, "source_dataset": source.dataset.iloc[0], "source_severe_prior": "AVAILABLE", "feature_name": name, "weight": float(weight), "reason": "SOURCE_CAUSAL_PLATEAU_TO_POST_PLATEAU"})
    prior_frame = pd.DataFrame([{"protocol_id": protocol_id, "source_dataset": source.dataset.iloc[0], "baseline_D_p95": prior.baseline_d_p95, "source_plateau_V50_p75": prior.v50_p75, "source_plateau_V100_p75": prior.v100_p75, "source_plateau_volatility_p75": prior.volatility_p75, "source_plateau_detected": int(not plateau_events.empty), "source_exit_detected": int(not exit_events.empty)}])
    model = SourceProtocolModel(protocol_id, str(source.dataset.iloc[0]), features, feature_strength, prior, source_states, direction)
    return model, prior_frame, pd.DataFrame(severe_audit_rows)
