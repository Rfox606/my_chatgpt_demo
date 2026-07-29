from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gradual_state_monitoring_v341.config import GradualStateMonitoringV341Config


@dataclass(frozen=True)
class GradualStateMonitoringV35Config(GradualStateMonitoringV341Config):
    """Predeclared V3.5 protocol; thresholds are calibration-only and fixed online."""

    output_dir: str = "outputs_gradual_state_monitoring_v35"
    candidate_merge_gap_windows: int = 20
    fast_lag: int = 16
    prediction_lag: int = 20
    slow_lags: tuple[int, int] = (64, 128)
    slow_weights: tuple[float, float] = (0.4667, 0.5333)
    residual_persistence_windows: int = 5
    adaptation_safe_windows: int = 5
    v35_settings: tuple[str, ...] = (
        "V341_CalibrationOnly",
        "V341_OnlineAdaptation",
        "V35_SlowOnly_CalibrationOnly",
        "V35_SlowOnly_OnlineAdaptation",
        "V35_SlowResidual_CalibrationOnly",
        "V35_SlowResidual_OnlineAdaptation",
    )

    def jsonable(self) -> dict[str, Any]:
        payload = super().jsonable()
        payload.update({
            "code_version": "gradual_state_monitoring_v35",
            "base_commit": "e1b316e",
            "online_input_contract": "Only the present and preceding six-dimensional force windows are read online; stopping position, total duration and target Stage are excluded.",
            "scoring_order": "All scores and the adaptation-gate decision are committed before any Adapter or Normalizer update.",
            "slow_score": "0.4667 * robust_z(||z_t-z_t_64||) + 0.5333 * robust_z(||z_t-z_t_128||)",
            "residual": "||normalizer(x_t)-normalizer(x_t-20)-prediction_head(z_t-20)||, then calibration MAD z-score and causal median over five windows.",
            "candidate_rule": "SlowOnly uses slow_final_score; SlowResidual requires slow_final_score AND residual_persistent; both require three consecutive high windows.",
            "adaptation_gate": "Online adaptation requires five consecutive windows below the fixed fast, slow and residual calibration thresholds.",
        })
        payload["slow_lags"] = list(self.slow_lags)
        payload["slow_weights"] = list(self.slow_weights)
        payload["v35_settings"] = list(self.v35_settings)
        return payload
