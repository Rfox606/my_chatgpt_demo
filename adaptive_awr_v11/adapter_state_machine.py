from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import AdaptiveAWRV11Config


STATE_CALIBRATION = "CALIBRATION"
STATE_RESTART_GUARD = "RESTART_GUARD"
STATE_ACTIVE_MONITOR = "ACTIVE_MONITOR"
STATE_ACTIVE_UPDATE = "ACTIVE_UPDATE"
STATE_FROZEN_EVENT = "FROZEN_EVENT"
STATE_FROZEN_RISK = "FROZEN_RISK"
STATE_COOLDOWN = "COOLDOWN"


@dataclass
class Checkpoint:
    window_index: int
    reliabilities: dict[str, float]
    residual_offset: float


class AdapterStateMachine:
    """Episode-oriented update/freeze state machine using absolute window limits."""

    def __init__(self, config: AdaptiveAWRV11Config) -> None:
        self.config = config
        self.state = STATE_CALIBRATION
        self.freeze_until_window = -1
        self.freeze_reason: str | None = None
        self.safe_run = 0
        self.events: list[dict[str, object]] = []
        self.checkpoint: Checkpoint | None = None
        self.last_update_window: int | None = None
        self.rollback_evaluated_for_update: int | None = None
        self._last_state = self.state

    def _event(self, window_index: int, event_type: str, reason: str, details: Mapping[str, object] | None = None) -> None:
        self.events.append(
            {
                "window_index": int(window_index),
                "event_type": event_type,
                "reason": reason,
                "state": self.state,
                "freeze_until_window": int(self.freeze_until_window),
                "details": json.dumps(dict(details or {}), ensure_ascii=False, sort_keys=True),
            }
        )

    @staticmethod
    def _priority(reason: str) -> int:
        return {"HIGH_RISK": 4, "TES_EVENT": 3, "HIGH_AWR_HIGH_BD": 2, "SUSTAINED_RS": 1}.get(reason, 0)

    def request_freeze(self, window_index: int, reason: str, details: Mapping[str, object]) -> None:
        candidate_until = int(window_index + self.config.risk_freeze_windows)
        existing_active = window_index <= self.freeze_until_window
        is_higher_priority = self._priority(reason) > self._priority(self.freeze_reason or "")
        extends_enough = candidate_until >= self.freeze_until_window + self.config.rollback_extension_min_windows
        if not existing_active:
            self.freeze_until_window = candidate_until
            self.freeze_reason = reason
            self.state = STATE_FROZEN_RISK if reason == "HIGH_RISK" else STATE_FROZEN_EVENT
            self._event(window_index, "ENTER_STATE", reason, details)
        elif is_higher_priority or extends_enough:
            self.freeze_until_window = max(self.freeze_until_window, candidate_until)
            if is_higher_priority:
                self.freeze_reason = reason
                self.state = STATE_FROZEN_RISK if reason == "HIGH_RISK" else STATE_FROZEN_EVENT
            self._event(window_index, "EXTEND_FREEZE", reason, details)

    def tick(self, window_index: int, calibration: bool, restart_guard: bool, safe_conditions: bool) -> tuple[str, bool]:
        if calibration:
            self.state = STATE_CALIBRATION
            self.safe_run = 0
            return self.state, False
        if restart_guard:
            self.state = STATE_RESTART_GUARD
            self.safe_run = 0
            return self.state, False
        if window_index <= self.freeze_until_window:
            self.safe_run = 0
            return self.state, False
        if self.state in (STATE_FROZEN_EVENT, STATE_FROZEN_RISK, STATE_COOLDOWN):
            self.state = STATE_ACTIVE_MONITOR
            self._event(window_index, "EXIT_FREEZE", self.freeze_reason or "released")
            self.freeze_reason = None
        self.safe_run = self.safe_run + 1 if safe_conditions else 0
        self.state = STATE_ACTIVE_UPDATE if self.safe_run >= self.config.safe_run_required else STATE_ACTIVE_MONITOR
        return self.state, self.state == STATE_ACTIVE_UPDATE

    def mark_update(self, window_index: int, details: Mapping[str, object]) -> None:
        self.last_update_window = int(window_index)
        self._event(window_index, "UPDATE", "safe_run", details)

    def save_checkpoint(self, window_index: int, reliabilities: Mapping[str, float], residual_offset: float) -> None:
        if window_index == 0 or window_index % self.config.checkpoint_interval:
            return
        self.checkpoint = Checkpoint(int(window_index), dict(reliabilities), float(residual_offset))
        self._event(window_index, "CHECKPOINT", "periodic")

    def rollback_if_needed(
        self,
        window_index: int,
        awr_history: list[float],
        bd_history: list[float],
        slow_history: list[float],
        reliabilities: dict[str, float],
        residual_offset: float,
    ) -> tuple[bool, float]:
        if self.checkpoint is None or self.last_update_window is None:
            return False, residual_offset
        if self.last_update_window <= self.checkpoint.window_index or window_index < self.last_update_window + self.config.rollback_eval_windows:
            return False, residual_offset
        if self.rollback_evaluated_for_update == self.last_update_window:
            return False, residual_offset
        self.rollback_evaluated_for_update = self.last_update_window
        window = self.config.rollback_eval_windows
        if len(awr_history) < window:
            return False, residual_offset
        x = np.arange(window, dtype=float)
        awr_slope = float(np.polyfit(x, awr_history[-window:], 1)[0])
        bd_slope = float(np.polyfit(x, bd_history[-window:], 1)[0])
        slow_drop = float(max(slow_history[-window:]) - slow_history[-1])
        if not (awr_slope > 0.0 and bd_slope > 0.0 and slow_drop > self.config.rollback_risk_drop):
            return False, residual_offset
        old = {"reliabilities": dict(reliabilities), "residual_offset": residual_offset}
        reliabilities.update(self.checkpoint.reliabilities)
        restored_offset = self.checkpoint.residual_offset
        self.freeze_until_window = window_index + self.config.cooldown_windows
        self.state = STATE_COOLDOWN
        self._event(
            window_index,
            "ROLLBACK",
            "awr_and_bd_rising_while_slow_risk_suppressed",
            {
                "checkpoint_window": self.checkpoint.window_index,
                "last_update_window": self.last_update_window,
                "evaluation_window": window_index,
                "AWR_slope": awr_slope,
                "BD_slope": bd_slope,
                "slow_risk_drop": slow_drop,
                "old_parameters": old,
                "restored_parameters": {"reliabilities": self.checkpoint.reliabilities, "residual_offset": restored_offset},
            },
        )
        return True, restored_offset
