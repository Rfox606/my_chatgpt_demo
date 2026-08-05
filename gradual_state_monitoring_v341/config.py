from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gradual_state_monitoring_v34.config import GradualStateMonitoringV34Config


@dataclass(frozen=True)
class GradualStateMonitoringV341Config(GradualStateMonitoringV34Config):
    output_dir: str = "outputs_gradual_state_monitoring_v341"
    candidate_merge_gap_windows: int = 20

    def jsonable(self) -> dict[str, Any]:
        payload = super().jsonable()
        payload.update({
            "code_version": "gradual_state_monitoring_v341",
            "base_commit": "599d1dcbbdec418c411c816b5a2a0f5ded7258aa",
            "distance_protocol": "At t, t-16, t-64 and t-128 are re-encoded with the current Adapter and Normalizer.",
            "direct_transfer_protocol": "Frozen source normalizer, Adapter/TCN, score histories and candidate threshold only.",
            "dual_branch_protocol": "final = frozen + 0.3 * max(0, adaptive - frozen)",
        })
        return payload
