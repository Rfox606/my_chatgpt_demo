from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import ContinuousStateV31Config
from .data import assert_label_free, baseline_mask, robust_location_scale


def _safe_covariance(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        covariance = LedoitWolf().fit(values).covariance_
        precision = np.linalg.pinv(covariance)
        if not np.isfinite(precision).all():
            raise FloatingPointError("non-finite precision")
        return covariance, precision, "ledoit_wolf_mahalanobis"
    except Exception:
        variance = np.maximum(np.var(values, axis=0), 1e-6)
        covariance = np.diag(variance)
        return covariance, np.diag(1. / variance), "diagonal_fallback"


def _distance(values: np.ndarray, centroid: np.ndarray, precision: np.ndarray) -> np.ndarray:
    delta = values - centroid
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, precision, delta), 0.))


def _velocity(values: np.ndarray, position: int, length: int) -> np.ndarray:
    half = length // 2
    if position + 1 < length:
        return np.zeros(values.shape[1], dtype=float)
    recent = np.median(values[position - half + 1:position + 1], axis=0)
    earlier = np.median(values[position - length + 1:position - half + 1], axis=0)
    return recent - earlier


def _volatility(values: np.ndarray, position: int, length: int) -> float:
    part = values[max(0, position - length + 1):position + 1]
    centre = np.median(part, axis=0)
    radius = np.linalg.norm(part - centre, axis=1)
    return float(np.median(np.abs(radius - np.median(radius))))


def _mad_scale(values: np.ndarray, eps: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    middle = float(np.median(values))
    mad = float(np.median(np.abs(values - middle)))
    iqr = float(np.quantile(values, .75) - np.quantile(values, .25))
    return middle, max(1.4826 * mad, iqr / 1.349, eps)


def _normalise(vector: np.ndarray, eps: float) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > eps else None


def _signature(values: np.ndarray | None) -> str:
    if values is None:
        return ""
    return ";".join(f"{value:.12g}" for value in np.asarray(values).ravel())


@dataclass(frozen=True)
class PlateauPrior:
    baseline_d_p95: float
    v50_threshold: float
    v100_threshold: float
    volatility_threshold: float
    quantile: float


@dataclass(frozen=True)
class TargetContext:
    median0: np.ndarray
    scale0: np.ndarray
    centroid: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    method: str
    baseline: np.ndarray
    source_low: np.ndarray
    source_high: np.ndarray
    support_weight: np.ndarray


@dataclass
class EvidenceAccumulator:
    """Evidence measured in valid observed cycles; guard windows are true pauses."""

    candidate_cycles: float
    confirm_cycles: float
    failure_reset_cycles: float
    searching: str
    candidate: str
    confirmed_name: str
    valid_cycles: float = 0.
    failure_cycles: float = 0.
    state: str = ""
    confirmed: bool = False

    def __post_init__(self) -> None:
        if not self.state:
            self.state = self.searching

    def step(self, is_guard: bool, condition: bool, increment: float) -> tuple[bool, str, bool]:
        """Return ``(reset, reason, just_confirmed)`` without changing evidence in guards."""
        if self.confirmed or is_guard:
            return False, "", False
        if condition:
            self.valid_cycles += increment
            self.failure_cycles = 0.
            if self.valid_cycles >= self.confirm_cycles:
                just = not self.confirmed
                self.confirmed = True; self.state = self.confirmed_name
                return False, "", just
            if self.valid_cycles >= self.candidate_cycles:
                self.state = self.candidate
            return False, "", False
        self.failure_cycles += increment
        if self.failure_cycles >= self.failure_reset_cycles:
            self.valid_cycles = 0.; self.failure_cycles = 0.; self.state = self.searching
            return True, "FAILURE_VALID_CYCLES_REACHED", False
        return False, "", False


def infer_nominal_stride(frame: pd.DataFrame, eps: float = 1e-9) -> float:
    centres = frame.sort_values(["center_cycle", "window_index"]).center_cycle.to_numpy(float)
    diff = np.diff(centres)
    diff = diff[diff > eps]
    return float(np.median(diff)) if len(diff) else 1.


def build_target_context(target: pd.DataFrame, source: pd.DataFrame, features: tuple[str, ...], strength: dict[str, float], config: ContinuousStateV31Config) -> tuple[TargetContext, np.ndarray]:
    assert_label_free(target); assert_label_free(source)
    raw = target.loc[:, list(features)].to_numpy(float)
    mask = baseline_mask(target, config)
    median0, scale0 = robust_location_scale(raw[mask], config.eps)
    relative = np.clip((raw - median0) / scale0, -config.target_clip, config.target_clip)
    centroid = np.median(relative[mask], axis=0)
    covariance, precision, method = _safe_covariance(relative[mask])
    source_values = source.loc[:, list(features)].to_numpy(float)
    weight = np.asarray([strength.get(feature, 1.) for feature in features], dtype=float)
    weight = weight / max(float(weight.sum()), config.eps)
    return TargetContext(median0, scale0, centroid, covariance, precision, method, mask,
                         np.quantile(source_values, .01, axis=0), np.quantile(source_values, .99, axis=0), weight), relative


def derive_plateau_prior(source: pd.DataFrame, features: tuple[str, ...], config: ContinuousStateV31Config, quantile: float | None = None) -> PlateauPrior:
    """Fixed source-only thresholds, with no access to stage labels or target values."""
    strength = {name: 1. for name in features}
    context, relative = build_target_context(source, source, features, strength, config)
    d_state = _distance(relative, context.centroid, context.precision)
    v50 = np.asarray([np.linalg.norm(_velocity(relative, position, 50)) for position in range(len(source))])
    v100 = np.asarray([np.linalg.norm(_velocity(relative, position, 100)) for position in range(len(source))])
    volatility = np.asarray([_volatility(relative, position, 50) for position in range(len(source))])
    usable = (source.center_cycle.to_numpy(float) >= config.plateau_min_cycle) & source.is_restart_guard.eq(0).to_numpy(bool)
    if not usable.any():
        usable = source.is_restart_guard.eq(0).to_numpy(bool)
    q = config.source_plateau_threshold_quantile if quantile is None else quantile
    return PlateauPrior(float(np.quantile(d_state[context.baseline], .95)), float(np.quantile(v50[usable], q)),
                        float(np.quantile(v100[usable], q)), float(np.quantile(volatility[usable], q)), q)


def run_target_state(target: pd.DataFrame, source: pd.DataFrame, features: tuple[str, ...], feature_strength: dict[str, float],
                     prior: PlateauPrior, source_severe_direction: np.ndarray | None, protocol_id: str,
                     config: ContinuousStateV31Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run the causal guard-aware state machine on one target stream.

    A guard records a pause, not a failure: neither successful nor failed evidence can
    be changed by it.  The function accepts only label-free frames.
    """
    assert_label_free(target); assert_label_free(source)
    target = target.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    context, relative = build_target_context(target, source, features, feature_strength, config)
    raw = target.loc[:, list(features)].to_numpy(float)
    stride = infer_nominal_stride(target, config.eps)
    d_state = _distance(relative, context.centroid, context.precision)
    d_diagonal = np.sqrt(np.mean((relative - context.centroid) ** 2, axis=1))
    plateau = EvidenceAccumulator(config.plateau_candidate_valid_cycles, config.plateau_lock_valid_cycles,
                                  config.plateau_failure_reset_valid_cycles, "SEARCHING", "CANDIDATE", "LOCKED")
    exit_evidence = EvidenceAccumulator(config.exit_candidate_valid_cycles, config.exit_confirm_valid_cycles,
                                        config.exit_failure_reset_valid_cycles, "PLATEAU", "EXIT_CANDIDATE", "EXIT_CONFIRMED")
    buffer: list[int] = []; a_history: list[float] = []; s_history: list[float] = []
    severe_evidence = EvidenceAccumulator(config.exit_candidate_valid_cycles, config.exit_confirm_valid_cycles,
                                          config.exit_failure_reset_valid_cycles, "SEARCHING", "CANDIDATE", "CONFIRMED")
    locked_cycle = np.nan; ref_start = np.nan; ref_end = np.nan; ref_count = 0; ref_valid = 0.
    plateau_centroid: np.ndarray | None = None; plateau_covariance: np.ndarray | None = None; plateau_precision: np.ndarray | None = None
    ref_distance = ref_v50 = ref_a20 = ref_projection_scale = (0., 1.)
    severe_direction = None if source_severe_direction is None else np.asarray(source_severe_direction, dtype=float).copy()
    exit_cycle = np.nan; severe_first_cycle = np.nan
    plateau_events: list[dict[str, object]] = []; exit_events: list[dict[str, object]] = []; updates: list[dict[str, object]] = []; severe_events: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    v20_items: list[np.ndarray] = []; v50_items: list[np.ndarray] = []; v100_items: list[np.ndarray] = []
    crosses_before_lock = 0

    for position, record in target.iterrows():
        is_guard = bool(record.is_restart_guard)
        if not plateau.confirmed and is_guard:
            crosses_before_lock += int(bool(record.crosses_stop_boundary))
        v20 = _velocity(relative, position, 20); v50 = _velocity(relative, position, 50); v100 = _velocity(relative, position, 100)
        v20_items.append(v20); v50_items.append(v50); v100_items.append(v100)
        v20_norm, v50_norm, v100_norm = (float(np.linalg.norm(vector)) for vector in (v20, v50, v100))
        cosine = float(v20 @ v100 / max(float(np.linalg.norm(v20) * np.linalg.norm(v100)), config.eps))
        acceleration = v20_norm - v100_norm; a_history.append(acceleration)
        a20 = float(np.median(a_history[-20:])); a50 = float(np.median(a_history[-50:]))
        vol20, vol50 = _volatility(relative, position, 20), _volatility(relative, position, 50)
        oos = ((raw[position] < context.source_low) | (raw[position] > context.source_high)).astype(float)
        weighted_oos = float(oos @ context.support_weight)
        d_ok = bool(d_state[position] > prior.baseline_d_p95)
        v50_ok = bool(v50_norm < prior.v50_threshold)
        v100_ok = bool(v100_norm < prior.v100_threshold)
        volatility_ok = bool(vol50 < prior.volatility_threshold)
        plateau_condition = bool(record.center_cycle >= config.plateau_min_cycle and not is_guard and d_ok and v50_ok and v100_ok and volatility_ok)
        plateau_reset, plateau_reason, just_locked = plateau.step(is_guard, plateau_condition, stride)
        if not is_guard and plateau_condition and not plateau.confirmed:
            buffer.append(position)
        if plateau_reset:
            buffer.clear()
        if just_locked:
            # The buffer contains only qualifying, arrived non-guard windows.  Work backwards
            # by valid-cycle equivalent, never by a natural-time slice.
            selected: list[int] = []; cumulative = 0.
            for index in reversed(buffer + [position]):
                if index in selected:
                    continue
                selected.append(index); cumulative += stride
                if cumulative >= config.plateau_reference_valid_cycles:
                    break
            selected = sorted(selected)
            ref_values = relative[selected]
            plateau_centroid = np.median(ref_values, axis=0)
            plateau_covariance, plateau_precision, _ = _safe_covariance(ref_values)
            distances = _distance(ref_values, plateau_centroid, plateau_precision)
            ref_distance = _mad_scale(distances, config.eps)
            ref_v50 = _mad_scale(np.asarray([np.linalg.norm(v50_items[index]) for index in selected]), config.eps)
            ref_a20 = _mad_scale(np.asarray([np.median(a_history[max(0, index - 19):index + 1]) for index in selected]), config.eps)
            # Projection reference is centred at the frozen plateau centroid.  Its scalar scale
            # is fixed now, before any future severe direction is learned.
            ref_projection_scale = (0., max(float(np.median(np.linalg.norm(ref_values - plateau_centroid, axis=1))), config.eps))
            locked_cycle = float(record.center_cycle); ref_start = float(target.center_cycle.iloc[selected[0]])
            ref_end = float(target.center_cycle.iloc[selected[-1]]); ref_count = len(selected); ref_valid = cumulative
            plateau_events.append({"protocol_id": protocol_id, "target_dataset": record.dataset, "event": "PLATEAU_LOCKED",
                                   "cycle": locked_cycle, "plateau_reference_start_cycle": ref_start,
                                   "plateau_reference_end_cycle": ref_end, "plateau_reference_window_count": ref_count,
                                   "plateau_reference_valid_cycles": ref_valid, "guards_crossed_before_lock": crosses_before_lock})

        distance_plateau = instability = np.nan
        exit_condition = False; exit_reset = False; exit_reason = ""; just_exit = False
        if plateau.confirmed and plateau_centroid is not None and plateau_precision is not None:
            distance_plateau = float(_distance(relative[position:position + 1], plateau_centroid, plateau_precision)[0])
            d_med, d_scale = ref_distance; v_med, v_scale = ref_v50; a_med, a_scale = ref_a20
            instability = (.35 * ((distance_plateau - d_med) / d_scale) + .25 * ((v50_norm - v_med) / v_scale)
                           + .20 * ((max(a20, 0.) - a_med) / a_scale) + .20 * max(cosine, 0.))
            exit_condition = bool(not is_guard and instability > config.severe_score_threshold)
            exit_reset, exit_reason, just_exit = exit_evidence.step(is_guard, exit_condition, stride)
            if just_exit:
                exit_cycle = float(record.center_cycle)
                exit_events.append({"protocol_id": protocol_id, "target_dataset": record.dataset, "event": "PLATEAU_EXIT_CONFIRMED",
                                    "cycle": exit_cycle, "instability_score": float(instability), "valid_cycles": exit_evidence.valid_cycles})

        update = 0; update_rejected = 0; reject_reason = ""; previous_cosine = np.nan
        if exit_evidence.confirmed:
            if is_guard:
                reject_reason = "RESTART_GUARD"
            elif cosine < config.severe_direction_consistency_min:
                reject_reason = "DIRECTION_CONSISTENCY_BELOW_MIN"
            elif v50_norm < prior.v50_threshold:
                reject_reason = "V50_NOT_PERSISTENTLY_HIGH"
            elif weighted_oos > config.weighted_oos_max:
                reject_reason = "WEIGHTED_OOS_ABOVE_LIMIT"
            else:
                recent = np.flatnonzero((target.index.to_numpy() <= position) & target.is_restart_guard.eq(0).to_numpy(bool))[-100:]
                proposal = _normalise(np.median(relative[recent], axis=0) - plateau_centroid, config.eps) if len(recent) else None
                if proposal is None:
                    reject_reason = "ZERO_SEVERE_DIRECTION"
                else:
                    previous_cosine = float(proposal @ severe_direction) if severe_direction is not None else 1.
                    if severe_direction is not None and previous_cosine < config.severe_direction_cosine_min:
                        reject_reason = "COSINE_TO_PREVIOUS_BELOW_MIN"
                    else:
                        severe_direction = proposal if severe_direction is None else _normalise((1. - config.severe_eta) * severe_direction + config.severe_eta * proposal, config.eps)
                        update = 1
                        if not np.isfinite(severe_first_cycle):
                            severe_first_cycle = float(record.center_cycle)
                        for feature, weight in zip(features, severe_direction, strict=True):
                            updates.append({"protocol_id": protocol_id, "target_dataset": record.dataset, "cycle": float(record.center_cycle),
                                            "feature_name": feature, "weight": float(weight), "severe_direction_cosine_previous": previous_cosine,
                                            "update_accepted": 1, "used_max_cycle": float(record.center_cycle), "exit_confirmation_cycle": exit_cycle})
            update_rejected = int(update == 0 and bool(reject_reason))

        severe_available = bool(plateau.confirmed and severe_direction is not None)
        severe_score = np.nan
        if severe_available and plateau_centroid is not None and plateau_precision is not None:
            projection = float((relative[position] - plateau_centroid) @ severe_direction)
            p_med, p_scale = ref_projection_scale; d_med, d_scale = ref_distance; v_med, v_scale = ref_v50
            severe_score = (.50 * ((projection - p_med) / p_scale) + .20 * ((distance_plateau - d_med) / d_scale)
                            + .20 * ((v50_norm - v_med) / v_scale) + .10 * max(cosine, 0.))
            s_history.append(float(severe_score))
        s20 = float(np.median(s_history[-20:])) if s_history else np.nan
        s50 = float(np.median(s_history[-50:])) if s_history else np.nan
        severe_condition = bool(severe_available and np.isfinite(s50) and s50 > config.severe_score_threshold)
        _, _, severe_confirmed_now = severe_evidence.step(is_guard, severe_condition, stride)
        if severe_confirmed_now:
            severe_events.append({"protocol_id": protocol_id, "target_dataset": record.dataset, "event": "SEVERE_CANDIDATE_PERSISTENT",
                                  "cycle": float(record.center_cycle), "S_severe_candidate": severe_score, "valid_cycles": severe_evidence.valid_cycles})

        row = {**record.to_dict(), "protocol_id": protocol_id, "nominal_stride_cycles": stride,
               "evidence_increment_cycles": 0. if is_guard else (stride if plateau_condition else 0.),
               "D_state": float(d_state[position]), "D_diagonal": float(d_diagonal[position]), "baseline_distance_method": context.method,
               "baseline_window_count": int(context.baseline.sum()), "baseline_start_cycle": float(target.loc[context.baseline, "center_cycle"].min()),
               "baseline_end_cycle": float(target.loc[context.baseline, "center_cycle"].max()), "baseline_method": "initial_500_non_guard_median_mad_iqr",
               "V20_norm": v20_norm, "V50_norm": v50_norm, "V100_norm": v100_norm, "direction_consistency": cosine,
               "A_state": acceleration, "A_smooth_20": a20, "A_smooth_50": a50, "state_volatility_20": vol20, "state_volatility_50": vol50,
               "D_condition": int(d_ok), "V50_condition": int(v50_ok), "V100_condition": int(v100_ok), "volatility_condition": int(volatility_ok),
               "plateau_condition": int(plateau_condition), "plateau_valid_cycles": plateau.valid_cycles,
               "plateau_failure_valid_cycles": plateau.failure_cycles, "plateau_evidence_state": plateau.state,
               "plateau_reset_event": int(plateau_reset), "plateau_reset_reason": plateau_reason,
               "plateau_candidate": int(plateau.state in ("CANDIDATE", "LOCKED")), "plateau_locked": int(plateau.confirmed),
               "plateau_lock_cycle": locked_cycle, "plateau_reference_start_cycle": ref_start, "plateau_reference_end_cycle": ref_end,
               "plateau_reference_window_count": ref_count, "plateau_reference_valid_cycles": ref_valid,
               "plateau_centroid_signature": _signature(plateau_centroid), "plateau_covariance_signature": _signature(plateau_covariance),
               "plateau_precision_signature": _signature(plateau_precision), "plateau_distance_reference": _signature(np.asarray(ref_distance)),
               "plateau_velocity_reference": _signature(np.asarray(ref_v50)), "plateau_acceleration_reference": _signature(np.asarray(ref_a20)),
               "plateau_projection_reference": _signature(np.asarray(ref_projection_scale)), "distance_from_plateau": distance_plateau,
               "instability_score": instability, "exit_condition": int(exit_condition), "exit_valid_cycles": exit_evidence.valid_cycles,
               "exit_failure_valid_cycles": exit_evidence.failure_cycles, "exit_evidence_state": exit_evidence.state,
               "exit_reset_event": int(exit_reset), "exit_reset_reason": exit_reason,
               "plateau_exit_candidate": int(exit_evidence.state in ("EXIT_CANDIDATE", "EXIT_CONFIRMED")),
               "plateau_exit_confirmed": int(exit_evidence.confirmed), "plateau_exit_cycle": exit_cycle,
               "severe_direction_available": int(severe_available), "severe_direction_update": update,
               "severe_direction_update_rejected": update_rejected, "severe_direction_reject_reason": reject_reason,
               "severe_direction_cosine_previous": previous_cosine, "S_severe_candidate": severe_score,
               "S_smooth_20": s20, "S_smooth_50": s50, "severe_status": "AVAILABLE" if severe_available else "UNAVAILABLE",
               "weighted_oos": weighted_oos, "support_confidence": 1. - weighted_oos}
        for name, velocity20, velocity50, velocity100 in zip(features, v20, v50, v100, strict=True):
            row[f"V20_{name}"] = float(velocity20); row[f"V50_{name}"] = float(velocity50); row[f"V100_{name}"] = float(velocity100)
        rows.append(row)

    metadata = {"protocol_id": protocol_id, "target_dataset": str(target.dataset.iloc[0]), "features": list(features),
                "baseline_median0": context.median0.tolist(), "baseline_scale0": context.scale0.tolist(), "baseline_window_count": int(context.baseline.sum()),
                "nominal_stride_cycles": stride, "plateau_status": plateau.state, "plateau_lock_cycle": locked_cycle,
                "plateau_reference_start_cycle": ref_start, "plateau_reference_end_cycle": ref_end, "plateau_reference_window_count": ref_count,
                "plateau_reference_valid_cycles": ref_valid, "guards_crossed_before_lock": crosses_before_lock,
                "exit_status": exit_evidence.state, "exit_cycle": exit_cycle, "severe_direction": None if severe_direction is None else severe_direction.tolist(),
                "severe_direction_first_cycle": severe_first_cycle, "severe_events": severe_events}
    return (pd.DataFrame(rows), pd.DataFrame(plateau_events), pd.DataFrame(exit_events), pd.DataFrame(updates), metadata)
