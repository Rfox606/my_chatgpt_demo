from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from gradual_state_monitoring_v33.data import MAIN_FEATURES
from gradual_state_monitoring_v33.evidence import CausalEvidenceEngine
from gradual_state_monitoring_v331.config import GradualStateMonitoringV331Config
from gradual_state_monitoring_v331.engine import LocalStateLibrary, V331StateMachine

from .config import GradualStateMonitoringV332Config
from .online import validate_label_free_online_frame


def fixed_v331_state_config() -> GradualStateMonitoringV331Config:
    """The state/evidence configuration is copied unchanged from the committed V3.3.1 default."""
    return replace(GradualStateMonitoringV331Config())


def replay_model_prediction_surprise(frame: pd.DataFrame, error_at_arrival: np.ndarray,
                                     config: GradualStateMonitoringV332Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Feed only a model's causal h=20 errors into the unchanged V3.3.1 state path.

    Distribution shift, drift velocity, Evidence Engine weights, and the state
    machine thresholds are identical across forecast models.  Both Slow/Fast error
    slots receive the same forecast error so that the comparison changes only the
    prediction-surprise source; the Fast/Slow difference remains zero by design.
    """
    online = validate_label_free_online_frame(frame, config)
    if len(error_at_arrival) != len(online):
        raise ValueError("Prediction-error arrival vector must align exactly with the online frame")
    v331 = fixed_v331_state_config()
    values = online.loc[:, MAIN_FEATURES].to_numpy(float)
    evidence_engine = CausalEvidenceEngine(values.shape[1], v331)
    library = LocalStateLibrary(v331)
    machine = V331StateMachine(values.shape[1], v331, library)
    evidence_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    for index, row in online.iterrows():
        error = float(error_at_arrival[index])
        evidence = evidence_engine.step(values[index], error, error)
        evidence["model_divergence"] = 0.0
        evidence["fast_slow_prediction_difference"] = 0.0
        evidence["model_divergence_in_change_evidence"] = 0
        state = machine.step(
            index, float(row.center_cycle), values[index], evidence, evidence_engine.scale.scale.copy(),
            bool(evidence_engine.scale.frozen and index >= v331.history_windows - 1),
        )
        common = {"dataset": str(row.dataset), "input_index": int(index), "window_index": int(row.window_index), "cycle": float(row.center_cycle)}
        evidence_rows.append({**common, **evidence, "h20_prediction_error_arrival": error})
        state_rows.append({**common, **state})
    transitions = pd.DataFrame(machine.events)
    if len(transitions):
        transitions.insert(0, "dataset", str(online.dataset.iloc[0]))
    return pd.DataFrame(state_rows), pd.DataFrame(evidence_rows), transitions
