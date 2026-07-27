from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GradualStateMonitoringV33Config


STABLE = "STABLE"
TRANSITION = "TRANSITION"
NEW_STABLE = "NEW_STABLE"
TEMPORARY_DISTURBANCE = "TEMPORARY_DISTURBANCE"


@dataclass
class _TransitionCandidate:
    start_index: int
    start_cycle: float
    anchor: np.ndarray


class GradualStateMachine:
    """Hysteretic local-state machine; no state transition is caused by one row."""

    def __init__(self, feature_count: int, config: GradualStateMonitoringV33Config) -> None:
        self.config = config
        self.state = STABLE
        self.local_state_id = 1
        self.state_start_cycle = np.nan
        self.state_support = 0
        self.reference = np.zeros(feature_count, dtype=float)
        self.candidate_count = 0
        self.candidate_start: _TransitionCandidate | None = None
        self.transition: _TransitionCandidate | None = None
        self.return_count = 0
        self.settle_count = 0
        self.new_stable_count = 0
        self.temporary_count = 0
        self.events: list[dict[str, object]] = []

    def _change(self, state: str, index: int, cycle: float, reason: str) -> None:
        previous = self.state
        if previous == state:
            return
        self.state = state
        self.state_support = 1
        if state == TRANSITION and self.transition is not None:
            self.state_start_cycle = self.transition.start_cycle
        elif state == NEW_STABLE and self.transition is not None:
            self.local_state_id += 1
            self.state_start_cycle = self.transition.start_cycle
        elif state == TEMPORARY_DISTURBANCE and self.transition is not None:
            self.state_start_cycle = self.transition.start_cycle
        elif state == STABLE and np.isnan(self.state_start_cycle):
            self.state_start_cycle = cycle
        self.events.append({
            "input_index": index, "cycle": cycle, "from_state": previous, "to_state": state,
            "local_state_id": self.local_state_id, "state_start_cycle": self.state_start_cycle,
            "reason": reason,
        })

    def _stable_reference_update(self, value: np.ndarray) -> None:
        rate = self.config.stable_reference_learning_rate
        self.reference = (1.0 - rate) * self.reference + rate * value

    def step(self, index: int, cycle: float, value: np.ndarray, evidence: dict[str, float], *, ready: bool, distance: float) -> dict[str, object]:
        first_row = self.state_support == 0
        if first_row:
            self.reference = value.copy()
            self.state_start_cycle = cycle
            self.state_support = 1
        high_evidence = (
            evidence["change_evidence"] >= self.config.transition_evidence_threshold
            or evidence["prediction_surprise"] >= self.config.transition_evidence_threshold
            or evidence["distribution_shift"] >= self.config.transition_evidence_threshold
            or evidence["drift_velocity"] >= self.config.transition_evidence_threshold
        )
        current_velocity = evidence["raw_drift_velocity"] if np.isfinite(evidence["raw_drift_velocity"]) else 0.0
        distribution_recovered = evidence["distribution_shift"] <= self.config.recovery_evidence_threshold
        fast_recovered = evidence["fast_prediction_surprise"] <= self.config.recovery_evidence_threshold
        velocity_recovered = current_velocity <= self.config.stable_velocity_threshold

        if not ready:
            self._stable_reference_update(value)
        elif self.state == STABLE:
            if high_evidence or current_velocity >= self.config.transition_velocity_threshold:
                if self.candidate_count == 0:
                    self.candidate_start = _TransitionCandidate(index, cycle, self.reference.copy())
                self.candidate_count += 1
                if self.candidate_count >= self.config.transition_persistence and self.candidate_start is not None:
                    self.transition = self.candidate_start
                    self.return_count = 0
                    self.settle_count = 0
                    self._change(TRANSITION, index, cycle, "persistent_change_evidence")
            else:
                self.candidate_count = 0
                self.candidate_start = None
                self._stable_reference_update(value)
        elif self.state == TRANSITION:
            assert self.transition is not None
            duration = index - self.transition.start_index + 1
            anchor_distance = distance
            if duration <= self.config.temporary_max_duration and anchor_distance <= self.config.return_distance_threshold:
                self.return_count += 1
                if self.return_count >= self.config.recovery_persistence:
                    self.temporary_count = 0
                    self._change(TEMPORARY_DISTURBANCE, index, cycle, "short_change_returned_to_anchor")
            else:
                self.return_count = 0
                if (
                    anchor_distance >= self.config.new_state_distance_threshold
                    and velocity_recovered
                    and distribution_recovered
                    and fast_recovered
                ):
                    self.settle_count += 1
                    if self.settle_count >= self.config.new_stable_persistence:
                        self.new_stable_count = 0
                        self.reference = value.copy()
                        self._change(NEW_STABLE, index, cycle, "persistent_new_distribution_with_fast_recovery")
                else:
                    self.settle_count = 0
        elif self.state == NEW_STABLE:
            self.new_stable_count += 1
            self._stable_reference_update(value)
            if self.new_stable_count >= self.config.new_stable_hold:
                self.transition = None
                self.candidate_count = 0
                self._change(STABLE, index, cycle, "new_state_confirmed")
        elif self.state == TEMPORARY_DISTURBANCE:
            self.temporary_count += 1
            if self.temporary_count >= self.config.temporary_hold:
                self.transition = None
                self.candidate_count = 0
                self._change(STABLE, index, cycle, "temporary_disturbance_recovered")

        if not first_row and self.state_support and not (self.events and self.events[-1]["input_index"] == index):
            self.state_support += 1
        return {
            "online_state": self.state,
            "local_state_id": self.local_state_id,
            "state_start_cycle": self.state_start_cycle,
            "state_support": self.state_support,
            "distance_from_local_anchor": distance,
            "transition_candidate_support": self.candidate_count,
        }
