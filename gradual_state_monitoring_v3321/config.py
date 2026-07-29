from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gradual_state_monitoring_v332.config import GradualStateMonitoringV332Config


@dataclass(frozen=True)
class GradualStateMonitoringV3321Config(GradualStateMonitoringV332Config):
    """Do not alter V3.3.2 code or outputs; write the correction separately."""

    output_dir: str = "outputs_gradual_state_monitoring_v3321"

    def jsonable(self) -> dict[str, Any]:
        payload = super().jsonable()
        payload.update({
            "code_version": "gradual_state_monitoring_v3321",
            "correction": "Training origins satisfy origin + max(horizons) < train_end.",
            "legacy_v332_code_modified": False,
        })
        return payload
