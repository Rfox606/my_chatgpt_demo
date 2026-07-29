from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gradual_state_monitoring_v35.config import GradualStateMonitoringV35Config


@dataclass(frozen=True)
class GradualStateMonitoringV351Config(GradualStateMonitoringV35Config):
    output_dir: str = "outputs_gradual_state_monitoring_v351"
    normalizer_calibration_windows: int = 128
    score_calibration_start: int = 128
    score_calibration_end: int = 256
    warning_quantile: float = 0.90
    alarm_quantile: float = 0.95
    warning_safe_windows: int = 10
    event_residual_min_high_count: int = 2
    v351_settings: tuple[str, ...] = (
        "V35_HardAND_Current",
        "V351_EventGate_CurrentAdapt",
        "V351_EventGate_WarningFreeze",
        "V351_EventGate_WarningFreeze_Update5",
        "V351_ResidualRerank_WarningFreeze_Update5",
    )

    def jsonable(self) -> dict[str, Any]:
        payload = super().jsonable()
        payload.update({
            "code_version": "gradual_state_monitoring_v351",
            "base_commit": "6e21cb9",
            "calibration_protocol": "0-127 fit only the target Normalizer; 128-255 score calibration with fixed Adapter/Normalizer and complete lags; online monitoring starts at 256.",
            "event_protocol": "Slow-high regions persist for three windows, merge causally within 20 windows, and use event-level residual evidence from [start-5,end+5] only after it has arrived.",
            "adaptation_protocol": "WarningFreeze settings freeze on any 90th-percentile warning or pending event and require ten consecutive safe windows before updates resume.",
            "slow_score_sign": "V3.5 positive normalized medium/long displacement combination: +0.4667 and +0.5333.",
        })
        payload["v351_settings"] = list(self.v351_settings)
        return payload
