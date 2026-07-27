from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gradual_state_monitoring_v33.data import MAIN_FEATURES
from gradual_state_monitoring_v33.evidence import CausalEvidenceEngine, CausalRobustBaseline
from gradual_state_monitoring_v33.features import CausalDescriptorScaler, build_causal_descriptors
from gradual_state_monitoring_v33.forecasting import RecursiveMultiOutputRidge

from .config import GradualStateMonitoringV331Config


STABLE = "STABLE"
TRANSITION = "TRANSITION"
NEW_STABLE = "NEW_STABLE"
TEMPORARY_DISTURBANCE = "TEMPORARY_DISTURBANCE"


class FrozenDescriptorScaler:
    """Calibrate online for N rows and keep the descriptor coordinate system fixed."""

    def __init__(self, dimensions: int, calibration_windows: int, eps: float) -> None:
        self.dimensions = dimensions
        self.calibration_windows = calibration_windows
        self.eps = eps
        self.count = 0
        self.mean = np.zeros(dimensions, dtype=float)
        self.m2 = np.zeros(dimensions, dtype=float)
        self.variance = np.full(dimensions, np.nan)
        self.frozen = False

    def transform(self, value: np.ndarray) -> np.ndarray | None:
        if not self.frozen:
            return None
        return (np.asarray(value, dtype=float) - self.mean) / np.sqrt(np.maximum(self.variance, self.eps))

    def update(self, value: np.ndarray) -> None:
        if self.frozen:
            return
        self.count += 1
        delta = np.asarray(value, dtype=float) - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (np.asarray(value, dtype=float) - self.mean)
        if self.count >= self.calibration_windows:
            self.variance = self.m2 / max(self.count - 1, 1)
            self.frozen = True


class V331DualTimescaleForecaster:
    """Slow/Fast RLS with either a truly frozen or legacy rolling descriptor scaler."""

    def __init__(self, descriptor_dimensions: int, output_dimensions: int, config: GradualStateMonitoringV331Config, scaler_mode: str) -> None:
        self.config = config
        self.scaler_mode = scaler_mode
        self.frozen_scaler = FrozenDescriptorScaler(descriptor_dimensions, config.descriptor_scaler_calibration_windows, config.eps)
        self.rolling_scaler = CausalDescriptorScaler(descriptor_dimensions, config.eps)
        self.slow = RecursiveMultiOutputRidge(descriptor_dimensions, output_dimensions, config.slow_forgetting_factor, config.rls_initial_covariance, config.rls_min_updates)
        self.fast = RecursiveMultiOutputRidge(descriptor_dimensions, output_dimensions, config.fast_forgetting_factor, config.rls_initial_covariance, config.rls_min_updates)
        self.inputs: list[np.ndarray | None] = []
        self.slow_pending: list[np.ndarray] = []
        self.fast_pending: list[np.ndarray] = []

    @property
    def scaler_frozen(self) -> bool:
        return self.scaler_mode == "rolling_descriptor_scaler_v33" or self.frozen_scaler.frozen

    def _scaled(self, descriptor: np.ndarray) -> np.ndarray | None:
        if self.scaler_mode == "frozen_descriptor_scaler":
            value = self.frozen_scaler.transform(descriptor)
            self.frozen_scaler.update(descriptor)
            return value
        value = self.rolling_scaler.transform(descriptor)
        self.rolling_scaler.update(descriptor)
        return value

    def step(self, descriptor: np.ndarray, target: np.ndarray, index: int) -> dict[str, object]:
        x = self._scaled(descriptor)
        if x is None:
            slow_prediction = np.full(len(target), np.nan)
            fast_prediction = np.full(len(target), np.nan)
        else:
            slow_prediction = self.slow.predict(x)
            fast_prediction = self.fast.predict(x)
        self.inputs.append(None if x is None else x.copy())
        self.slow_pending.append(slow_prediction.copy())
        self.fast_pending.append(fast_prediction.copy())
        due = index - self.config.dual_horizon
        slow_error = np.nan
        fast_error = np.nan
        if due >= 0 and self.inputs[due] is not None:
            due_slow = self.slow_pending[due]
            due_fast = self.fast_pending[due]
            if np.isfinite(due_slow).all():
                slow_error = float(np.mean(np.abs(target - due_slow)))
            if np.isfinite(due_fast).all():
                fast_error = float(np.mean(np.abs(target - due_fast)))
            self.slow.update(self.inputs[due], target)  # type: ignore[arg-type]
            self.fast.update(self.inputs[due], target)  # type: ignore[arg-type]
        difference = float(np.mean(np.abs(slow_prediction - fast_prediction))) if np.isfinite(slow_prediction).all() and np.isfinite(fast_prediction).all() else np.nan
        return {
            "dual_horizon": self.config.dual_horizon,
            "slow_prediction_error": slow_error,
            "fast_prediction_error": fast_error,
            "fast_slow_prediction_difference": difference,
            "descriptor_scaler_mode": self.scaler_mode,
            "descriptor_scaler_frozen": int(self.scaler_frozen),
            "descriptor_scaler_calibration_count": self.frozen_scaler.count if self.scaler_mode == "frozen_descriptor_scaler" else self.rolling_scaler.count,
        }


@dataclass
class LocalStateRecord:
    local_state_id: int
    centre: np.ndarray
    scale: np.ndarray
    support: int
    first_seen_cycle: float
    last_seen_cycle: float


class LocalStateLibrary:
    """Causal experiment-local state memory with robust reuse rather than monotonic IDs."""

    def __init__(self, config: GradualStateMonitoringV331Config) -> None:
        self.config = config
        self.records: dict[int, LocalStateRecord] = {}
        self.reuse_events: list[dict[str, object]] = []
        self.initial_calibrated = False

    @staticmethod
    def _distance(value: np.ndarray, record: LocalStateRecord, current_scale: np.ndarray, eps: float) -> float:
        scale = np.maximum(np.maximum(record.scale, current_scale), eps)
        return float(np.mean(np.abs((value - record.centre) / scale)))

    def initialise(self, value: np.ndarray, scale: np.ndarray, cycle: float) -> int:
        if self.records:
            return min(self.records)
        self.records[1] = LocalStateRecord(1, value.copy(), np.maximum(scale.copy(), self.config.eps), 1, cycle, cycle)
        self.reuse_events.append({"cycle": cycle, "event": "initialise", "assigned_local_state_id": 1, "nearest_local_state_id": np.nan, "distance": np.nan, "action": "created"})
        return 1

    def calibrate_initial(self, state_id: int, value: np.ndarray, scale: np.ndarray, cycle: float) -> None:
        """Replace the pre-calibration placeholder scale with the first causal robust scale."""
        record = self.records.get(state_id)
        if self.initial_calibrated or record is None:
            return
        record.centre = value.copy()
        record.scale = np.maximum(scale.copy(), self.config.eps)
        record.last_seen_cycle = cycle
        self.initial_calibrated = True
        self.reuse_events.append({"cycle": cycle, "event": "calibrate_initial", "assigned_local_state_id": state_id,
                                  "nearest_local_state_id": state_id, "distance": 0.0, "action": "calibrated"})

    def confirm(self, value: np.ndarray, scale: np.ndarray, cycle: float) -> tuple[int, str, float, int | float]:
        if not self.records:
            return self.initialise(value, scale, cycle), "created", np.nan, np.nan
        candidates = [(self._distance(value, record, scale, self.config.eps), state_id) for state_id, record in self.records.items()]
        distance, nearest_id = min(candidates)
        if distance <= self.config.state_reuse_distance_threshold:
            record = self.records[nearest_id]
            rate = self.config.state_library_update_rate
            record.centre = (1.0 - rate) * record.centre + rate * value
            record.scale = (1.0 - rate) * record.scale + rate * np.maximum(scale, self.config.eps)
            record.support += 1
            record.last_seen_cycle = cycle
            action = "reused"
            assigned = nearest_id
        else:
            assigned = max(self.records) + 1
            self.records[assigned] = LocalStateRecord(assigned, value.copy(), np.maximum(scale.copy(), self.config.eps), 1, cycle, cycle)
            action = "created"
        self.reuse_events.append({"cycle": cycle, "event": "confirm_new_stable", "assigned_local_state_id": assigned, "nearest_local_state_id": nearest_id, "distance": distance, "action": action})
        return assigned, action, distance, nearest_id

    def touch(self, state_id: int, cycle: float) -> None:
        if state_id in self.records:
            self.records[state_id].last_seen_cycle = cycle

    def frame(self, dataset: str) -> pd.DataFrame:
        rows = [{
            "dataset": dataset, "local_state_id": record.local_state_id,
            "state_centre": ";".join(f"{value:.12g}" for value in record.centre),
            "state_scale": ";".join(f"{value:.12g}" for value in record.scale),
            "support": record.support, "first_seen_cycle": record.first_seen_cycle,
            "last_seen_cycle": record.last_seen_cycle,
        } for record in self.records.values()]
        return pd.DataFrame(rows)


@dataclass
class TransitionCandidate:
    start_index: int
    start_cycle: float
    anchor: np.ndarray


class V331StateMachine:
    """Persistent/hysteretic four-state controller connected to the local-state library."""

    def __init__(self, dimensions: int, config: GradualStateMonitoringV331Config, library: LocalStateLibrary) -> None:
        self.config = config
        self.library = library
        self.reference = np.zeros(dimensions, dtype=float)
        self.state = STABLE
        self.local_state_id = 1
        self.state_start_cycle = np.nan
        self.state_support = 0
        self.candidate_count = 0
        self.candidate: TransitionCandidate | None = None
        self.transition: TransitionCandidate | None = None
        self.return_count = 0
        self.settle_count = 0
        self.new_hold = 0
        self.temp_hold = 0
        self.events: list[dict[str, object]] = []

    def _emit(self, to_state: str, index: int, cycle: float, reason: str, reuse_action: str = "") -> None:
        previous = self.state
        if previous == to_state:
            return
        self.state = to_state
        self.state_support = 1
        if to_state in (TRANSITION, NEW_STABLE, TEMPORARY_DISTURBANCE) and self.transition is not None:
            self.state_start_cycle = self.transition.start_cycle
        self.events.append({"input_index": index, "cycle": cycle, "from_state": previous, "to_state": to_state,
                            "local_state_id": self.local_state_id, "state_start_cycle": self.state_start_cycle,
                            "reason": reason, "state_reuse_action": reuse_action})

    def _reference_update(self, value: np.ndarray) -> None:
        rate = self.config.stable_reference_learning_rate
        self.reference = (1.0 - rate) * self.reference + rate * value

    def step(self, index: int, cycle: float, value: np.ndarray, evidence: dict[str, float], scale: np.ndarray, ready: bool) -> dict[str, object]:
        first = self.state_support == 0
        if first:
            self.reference = value.copy()
            self.local_state_id = self.library.initialise(value, scale, cycle)
            self.state_start_cycle = cycle
            self.state_support = 1
        distance = float(np.mean(np.abs((value - self.reference) / np.maximum(scale, self.config.eps))))
        current_velocity = evidence["raw_drift_velocity"] if np.isfinite(evidence["raw_drift_velocity"]) else 0.0
        high = any((
            evidence["change_evidence"] >= self.config.transition_evidence_threshold,
            evidence["prediction_surprise"] >= self.config.transition_evidence_threshold,
            evidence["distribution_shift"] >= self.config.transition_evidence_threshold,
            evidence["drift_velocity"] >= self.config.transition_evidence_threshold,
        ))
        distribution_recovered = evidence["distribution_shift"] <= self.config.recovery_evidence_threshold
        fast_recovered = evidence["fast_prediction_surprise"] <= self.config.recovery_evidence_threshold
        velocity_recovered = current_velocity <= self.config.stable_velocity_threshold
        if not ready:
            self._reference_update(value)
        elif self.state == STABLE:
            self.library.calibrate_initial(self.local_state_id, self.reference, scale, cycle)
            if high or current_velocity >= self.config.transition_velocity_threshold:
                if self.candidate_count == 0:
                    self.candidate = TransitionCandidate(index, cycle, self.reference.copy())
                self.candidate_count += 1
                if self.candidate_count >= self.config.transition_persistence and self.candidate is not None:
                    self.transition = self.candidate
                    self.return_count = 0; self.settle_count = 0
                    self._emit(TRANSITION, index, cycle, "persistent_change_evidence")
            else:
                self.candidate_count = 0; self.candidate = None
                self._reference_update(value)
        elif self.state == TRANSITION:
            assert self.transition is not None
            duration = index - self.transition.start_index + 1
            if duration <= self.config.temporary_max_duration and distance <= self.config.return_distance_threshold:
                self.return_count += 1
                if self.return_count >= self.config.recovery_persistence:
                    self.temp_hold = 0
                    self._emit(TEMPORARY_DISTURBANCE, index, cycle, "short_change_returned_to_anchor")
            else:
                self.return_count = 0
                if distance >= self.config.new_state_distance_threshold and velocity_recovered and distribution_recovered and fast_recovered:
                    self.settle_count += 1
                    if self.settle_count >= self.config.new_stable_persistence:
                        self.local_state_id, action, reuse_distance, nearest = self.library.confirm(value, scale, cycle)
                        self.reference = value.copy(); self.new_hold = 0
                        self._emit(NEW_STABLE, index, cycle, f"persistent_new_distribution_{action}", action)
                        self.events[-1].update({"reuse_distance": reuse_distance, "nearest_local_state_id": nearest})
                else:
                    self.settle_count = 0
        elif self.state == NEW_STABLE:
            self.new_hold += 1; self._reference_update(value)
            if self.new_hold >= self.config.new_stable_hold:
                self.transition = None; self.candidate_count = 0
                self._emit(STABLE, index, cycle, "new_state_confirmed")
        elif self.state == TEMPORARY_DISTURBANCE:
            self.temp_hold += 1
            if self.temp_hold >= self.config.temporary_hold:
                self.transition = None; self.candidate_count = 0
                self._emit(STABLE, index, cycle, "temporary_disturbance_recovered")
        self.library.touch(self.local_state_id, cycle)
        if not first and not (self.events and self.events[-1]["input_index"] == index):
            self.state_support += 1
        return {"online_state": self.state, "local_state_id": self.local_state_id, "state_start_cycle": self.state_start_cycle,
                "state_support": self.state_support, "distance_from_local_anchor": distance,
                "transition_candidate_support": self.candidate_count}


@dataclass
class MonitorResult:
    evidence: pd.DataFrame
    state_path: pd.DataFrame
    dual_predictions: pd.DataFrame
    transitions: pd.DataFrame
    local_state_library: pd.DataFrame
    state_reuse_log: pd.DataFrame
    descriptor_scaler_snapshot: dict[str, object]


def run_monitor(frame: pd.DataFrame, config: GradualStateMonitoringV331Config, *, descriptor_scaler_mode: str = "frozen_descriptor_scaler", include_model_divergence: bool = False, include_dual: bool = True) -> MonitorResult:
    """Complete causal V3.3.1 monitor.  Delayed-entry callers must keep include_dual=True."""
    if not include_dual:
        raise ValueError("V3.3.1 does not permit delayed or ablation monitoring without Slow/Fast RLS")
    ordered = frame.sort_values("window_index", kind="stable").reset_index(drop=True)
    values = ordered.loc[:, MAIN_FEATURES].to_numpy(float)
    descriptors = build_causal_descriptors(values)
    dual_model = V331DualTimescaleForecaster(descriptors.shape[1], values.shape[1], config, descriptor_scaler_mode)
    evidence_engine = CausalEvidenceEngine(values.shape[1], config)
    divergence_baseline = CausalRobustBaseline(config.evidence_baseline_windows, config.evidence_min_history, config.eps, config.evidence_baseline_refresh)
    library = LocalStateLibrary(config)
    machine = V331StateMachine(values.shape[1], config, library)
    evidence_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    dual_rows: list[dict[str, object]] = []
    for index, row in ordered.iterrows():
        dual = dual_model.step(descriptors[index], values[index], index)
        evidence = evidence_engine.step(values[index], float(dual["slow_prediction_error"]), float(dual["fast_prediction_error"]))
        model_divergence, _, _ = divergence_baseline.score_then_update(float(dual["fast_slow_prediction_difference"]))
        evidence["model_divergence"] = model_divergence
        evidence["fast_slow_prediction_difference"] = float(dual["fast_slow_prediction_difference"])
        evidence["model_divergence_in_change_evidence"] = int(include_model_divergence)
        if include_model_divergence:
            evidence["change_evidence"] += config.model_divergence_weight * model_divergence
        state = machine.step(index, float(row.center_cycle), values[index], evidence, evidence_engine.scale.scale.copy(), bool(evidence_engine.scale.frozen and index >= config.history_windows - 1))
        common = {"dataset": str(row.dataset), "input_index": index, "window_index": int(row.window_index), "cycle": float(row.center_cycle),
                  "descriptor_scaler_mode": descriptor_scaler_mode}
        evidence_rows.append({**common, **evidence})
        state_rows.append({**common, **state})
        dual_rows.append({**common, **dual})
    transitions = pd.DataFrame(machine.events)
    if len(transitions):
        transitions.insert(0, "dataset", str(ordered.dataset.iloc[0]))
        transitions.insert(1, "descriptor_scaler_mode", descriptor_scaler_mode)
        transitions["model_divergence_in_change_evidence"] = int(include_model_divergence)
    snapshot: dict[str, object]
    if descriptor_scaler_mode == "frozen_descriptor_scaler":
        snapshot = {"mode": descriptor_scaler_mode, "frozen": dual_model.frozen_scaler.frozen, "count": dual_model.frozen_scaler.count,
                    "mean": dual_model.frozen_scaler.mean.copy(), "variance": dual_model.frozen_scaler.variance.copy()}
    else:
        snapshot = {"mode": descriptor_scaler_mode, "frozen": False, "count": dual_model.rolling_scaler.count,
                    "mean": dual_model.rolling_scaler.mean.copy(), "variance": dual_model.rolling_scaler.m2 / max(dual_model.rolling_scaler.count - 1, 1)}
    reuse = pd.DataFrame(library.reuse_events)
    if len(reuse):
        reuse.insert(0, "dataset", str(ordered.dataset.iloc[0]))
        reuse.insert(1, "descriptor_scaler_mode", descriptor_scaler_mode)
    return MonitorResult(pd.DataFrame(evidence_rows), pd.DataFrame(state_rows), pd.DataFrame(dual_rows), transitions,
                         library.frame(str(ordered.dataset.iloc[0])), reuse, snapshot)
