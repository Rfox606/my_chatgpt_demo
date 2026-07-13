from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import ContinuousStateV3Config
from .data import assert_label_free, baseline_mask, robust_location_scale


def _mad_scale(values: np.ndarray, eps: float) -> tuple[float, float]:
    values = np.asarray(values, float)
    if not len(values):
        return 0., 1.
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    iqr = float(np.quantile(values, .75) - np.quantile(values, .25))
    return median, max(1.4826 * mad, iqr / 1.349, eps)


def _safe_precision(values: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        covariance = LedoitWolf().fit(values).covariance_
        precision = np.linalg.pinv(covariance)
        if not np.isfinite(precision).all() or float(np.linalg.norm(precision)) < 1e-12:
            raise FloatingPointError
        return precision, "ledoit_wolf_mahalanobis"
    except Exception:
        scale = np.maximum(np.var(values, axis=0), 1e-6)
        return np.diag(1. / scale), "diagonal_fallback"


def _distance(values: np.ndarray, centroid: np.ndarray, precision: np.ndarray) -> np.ndarray:
    delta = values - centroid
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, precision, delta), 0.))


def _velocity(values: np.ndarray, position: int, length: int) -> np.ndarray:
    half = length // 2
    if position + 1 < length:
        return np.zeros(values.shape[1])
    recent = np.median(values[position - half + 1:position + 1], axis=0)
    previous = np.median(values[position - length + 1:position - half + 1], axis=0)
    return recent - previous


def _volatility(values: np.ndarray, position: int, length: int) -> float:
    part = values[max(0, position - length + 1):position + 1]
    centroid = np.median(part, axis=0)
    distance = np.linalg.norm(part - centroid, axis=1)
    return float(np.median(np.abs(distance - np.median(distance))))


def _normalise(vector: np.ndarray, eps: float) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > eps else None


@dataclass(frozen=True)
class TargetContext:
    features: tuple[str, ...]
    median0: np.ndarray
    scale0: np.ndarray
    baseline_centroid: np.ndarray
    baseline_precision: np.ndarray
    baseline_method: str
    baseline_mask: np.ndarray
    source_low: np.ndarray
    source_high: np.ndarray
    support_weight: np.ndarray


@dataclass(frozen=True)
class PlateauPrior:
    baseline_d_p95: float
    v50_p75: float
    v100_p75: float
    volatility_p75: float


def build_target_context(frame: pd.DataFrame, features: tuple[str, ...], source_frame: pd.DataFrame, feature_strength: dict[str, float], config: ContinuousStateV3Config) -> tuple[TargetContext, np.ndarray]:
    """Freeze target baseline location/scale/covariance from initial non-guard windows only."""
    assert_label_free(frame); assert_label_free(source_frame)
    raw = frame.loc[:, list(features)].to_numpy(float)
    mask = baseline_mask(frame, config)
    median0, scale0 = robust_location_scale(raw[mask], config.eps)
    relative = np.clip((raw - median0) / scale0, -config.target_clip, config.target_clip)
    centroid = np.median(relative[mask], axis=0)
    precision, method = _safe_precision(relative[mask])
    source_values = source_frame.loc[:, list(features)].to_numpy(float)
    low, high = np.quantile(source_values, .01, axis=0), np.quantile(source_values, .99, axis=0)
    strength = np.asarray([feature_strength.get(name, 1.) for name in features], float)
    strength = strength / max(float(strength.sum()), config.eps)
    return TargetContext(features, median0, scale0, centroid, precision, method, mask, low, high, strength), relative


def derive_plateau_prior(frame: pd.DataFrame, features: tuple[str, ...], config: ContinuousStateV3Config) -> tuple[PlateauPrior, pd.DataFrame, TargetContext, np.ndarray]:
    """Derive fixed plateau thresholds from source data only, without stage fields."""
    strength = {name: 1. for name in features}
    context, relative = build_target_context(frame, features, frame, strength, config)
    d = _distance(relative, context.baseline_centroid, context.baseline_precision)
    v50 = np.array([np.linalg.norm(_velocity(relative, position, 50)) for position in range(len(frame))])
    v100 = np.array([np.linalg.norm(_velocity(relative, position, 100)) for position in range(len(frame))])
    vol50 = np.array([_volatility(relative, position, 50) for position in range(len(frame))])
    usable = (frame.center_cycle.to_numpy(float) >= config.plateau_min_cycle) & frame.is_restart_guard.eq(0).to_numpy(bool)
    if not usable.any():
        usable = frame.is_restart_guard.eq(0).to_numpy(bool)
    prior = PlateauPrior(
        baseline_d_p95=float(np.quantile(d[context.baseline_mask], .95)),
        v50_p75=float(np.quantile(v50[usable], config.source_plateau_threshold_quantile)),
        v100_p75=float(np.quantile(v100[usable], config.source_plateau_threshold_quantile)),
        volatility_p75=float(np.quantile(vol50[usable], config.source_plateau_threshold_quantile)),
    )
    audit = pd.DataFrame({"D_state": d, "V50_norm": v50, "V100_norm": v100, "state_volatility_50": vol50})
    return prior, audit, context, relative


def run_target_state(
    target: pd.DataFrame, source: pd.DataFrame, features: tuple[str, ...], feature_strength: dict[str, float],
    plateau_prior: PlateauPrior, source_severe_direction: np.ndarray | None, protocol_id: str, config: ContinuousStateV3Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Causal state, frozen plateau reference, exit detection and severe-direction learning."""
    assert_label_free(target); assert_label_free(source)
    target = target.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    context, relative = build_target_context(target, features, source, feature_strength, config)
    raw = target.loc[:, list(features)].to_numpy(float)
    d_state = _distance(relative, context.baseline_centroid, context.baseline_precision)
    d_diagonal = np.sqrt(np.mean((relative - context.baseline_centroid) ** 2, axis=1))
    v20 = np.empty_like(relative); v50 = np.empty_like(relative); v100 = np.empty_like(relative)
    rows: list[dict[str, object]] = []
    a_history: list[float] = []; s_history: list[float] = []
    plateau_start: int | None = None; plateau_locked = False; plateau_lock_cycle = np.nan
    plateau_centroid: np.ndarray | None = None; plateau_precision: np.ndarray | None = None
    plateau_reference: dict[str, tuple[float, float]] | None = None
    exit_start: int | None = None; severe_start: int | None = None
    severe_direction = None if source_severe_direction is None else source_severe_direction.copy()
    update_rows: list[dict[str, object]] = []; plateau_events: list[dict[str, object]] = []; exit_events: list[dict[str, object]] = []; severe_events: list[dict[str, object]] = []
    for position, record in target.iterrows():
        v20[position] = _velocity(relative, position, 20); v50[position] = _velocity(relative, position, 50); v100[position] = _velocity(relative, position, 100)
        v20_norm, v50_norm, v100_norm = (float(np.linalg.norm(item)) for item in (v20[position], v50[position], v100[position]))
        cosine = float(v20[position] @ v100[position] / max(np.linalg.norm(v20[position]) * np.linalg.norm(v100[position]), config.eps))
        persistence = max(cosine, 0.) * v100_norm
        acceleration = v20_norm - v100_norm; a_history.append(acceleration)
        a20, a50 = float(np.median(a_history[-20:])), float(np.median(a_history[-50:]))
        vol20, vol50 = _volatility(relative, position, 20), _volatility(relative, position, 50)
        oos = ((raw[position] < context.source_low) | (raw[position] > context.source_high)).astype(float)
        weighted_oos = float(oos @ context.support_weight)
        support_confidence = 1. - weighted_oos
        candidate_condition = bool(record.center_cycle >= config.plateau_min_cycle and not record.is_restart_guard and d_state[position] > plateau_prior.baseline_d_p95 and v50_norm < plateau_prior.v50_p75 and v100_norm < plateau_prior.v100_p75 and vol50 < plateau_prior.volatility_p75)
        if not plateau_locked:
            plateau_start = position if candidate_condition and plateau_start is None else (plateau_start if candidate_condition else None)
            candidate_duration = 0. if plateau_start is None else float(record.center_cycle - target.center_cycle.iloc[plateau_start])
            plateau_candidate = int(candidate_duration >= config.plateau_candidate_cycles)
            if plateau_start is not None and candidate_duration >= config.plateau_lock_cycles:
                eligible = np.flatnonzero((target.center_cycle.to_numpy(float) <= float(record.center_cycle)) & (target.center_cycle.to_numpy(float) >= float(record.center_cycle) - config.plateau_reference_cycles) & target.is_restart_guard.eq(0).to_numpy(bool))
                if len(eligible) >= 5:
                    plateau_locked = True; plateau_lock_cycle = float(record.center_cycle)
                    plateau_centroid = np.median(relative[eligible], axis=0); plateau_precision, _ = _safe_precision(relative[eligible])
                    ref_distance = _distance(relative[eligible], plateau_centroid, plateau_precision)
                    ref_a = np.asarray([float(np.median(a_history[max(0, int(index) - 19):int(index) + 1])) for index in eligible])
                    plateau_reference = {
                        "distance": _mad_scale(ref_distance, config.eps), "v50": _mad_scale(np.linalg.norm(v50[eligible], axis=1), config.eps),
                        "a20": _mad_scale(np.maximum(ref_a, 0.), config.eps), "projection": (0., 1.),
                    }
                    plateau_events.append({"protocol_id": protocol_id, "target_dataset": record.dataset, "event": "PLATEAU_LOCKED", "cycle": plateau_lock_cycle, "reference_window_count": int(len(eligible)), "plateau_status": "LOCKED"})
            plateau_locked_value = int(plateau_locked)
        else:
            plateau_candidate = 1; plateau_locked_value = 1
        distance_plateau = float("nan"); instability = float("nan"); exit_candidate = 0; exit_confirmed = 0
        if plateau_locked and plateau_centroid is not None and plateau_precision is not None and plateau_reference is not None:
            distance_plateau = float(_distance(relative[position:position + 1], plateau_centroid, plateau_precision)[0])
            dist_m, dist_s = plateau_reference["distance"]; v_m, v_s = plateau_reference["v50"]; a_m, a_s = plateau_reference["a20"]
            instability = .35 * ((distance_plateau - dist_m) / dist_s) + .25 * ((v50_norm - v_m) / v_s) + .20 * ((max(a20, 0.) - a_m) / a_s) + .20 * max(cosine, 0.)
            exit_start = position if (not record.is_restart_guard and instability > config.severe_score_threshold and exit_start is None) else (exit_start if not record.is_restart_guard and instability > config.severe_score_threshold else None)
            exit_duration = 0. if exit_start is None else float(record.center_cycle - target.center_cycle.iloc[exit_start])
            exit_candidate, exit_confirmed = int(exit_duration >= config.plateau_exit_candidate_cycles), int(exit_duration >= config.plateau_exit_confirm_cycles)
            if exit_confirmed and not any(item["event"] == "PLATEAU_EXIT_CONFIRMED" for item in exit_events):
                exit_events.append({"protocol_id": protocol_id, "target_dataset": record.dataset, "event": "PLATEAU_EXIT_CONFIRMED", "cycle": float(record.center_cycle), "instability_score": instability})
            update_ok = bool(exit_confirmed and not record.is_restart_guard and cosine >= config.severe_direction_consistency_min and v50_norm >= plateau_prior.v50_p75 and weighted_oos <= config.weighted_oos_max and vol50 <= plateau_prior.volatility_p75 * 3.)
            severe_update, cosine_previous = 0, np.nan
            if update_ok:
                recent = np.flatnonzero((target.center_cycle.to_numpy(float) <= float(record.center_cycle)) & (target.is_restart_guard.to_numpy(int) == 0))[-100:]
                delta = np.median(relative[recent], axis=0) - plateau_centroid
                proposal = _normalise(delta, config.eps)
                if proposal is not None:
                    cosine_previous = float(proposal @ severe_direction) if severe_direction is not None else 1.
                    if severe_direction is None or cosine_previous >= config.severe_direction_cosine_min:
                        severe_direction = proposal if severe_direction is None else _normalise((1. - config.severe_eta) * severe_direction + config.severe_eta * proposal, config.eps)
                        severe_update = 1
                        for name, weight in zip(features, severe_direction, strict=True):
                            update_rows.append({"protocol_id": protocol_id, "target_dataset": record.dataset, "cycle": float(record.center_cycle), "feature_name": name, "weight": float(weight), "severe_direction_cosine_previous": cosine_previous, "update_accepted": 1})
            else:
                severe_update, cosine_previous = 0, np.nan
        else:
            severe_update, cosine_previous = 0, np.nan
        severe_available = int(plateau_locked and severe_direction is not None)
        severe_score = np.nan
        if severe_available and plateau_centroid is not None and plateau_reference is not None:
            projection = float((relative[position] - plateau_centroid) @ severe_direction)
            if plateau_reference["projection"] == (0., 1.):
                # The fixed plateau projection reference is set once, at the first available severe direction.
                ref_indices = np.flatnonzero((target.center_cycle.to_numpy(float) <= plateau_lock_cycle) & target.is_restart_guard.eq(0).to_numpy(bool))[-max(10, config.plateau_reference_cycles // 5):]
                plateau_reference["projection"] = _mad_scale((relative[ref_indices] - plateau_centroid) @ severe_direction, config.eps)
            p_m, p_s = plateau_reference["projection"]; d_m, d_s = plateau_reference["distance"]; v_m, v_s = plateau_reference["v50"]
            severe_score = .50 * ((projection - p_m) / p_s) + .20 * ((distance_plateau - d_m) / d_s) + .20 * ((v50_norm - v_m) / v_s) + .10 * max(cosine, 0.)
            s_history.append(float(severe_score))
        s20 = float(np.median(s_history[-20:])) if s_history else np.nan; s50 = float(np.median(s_history[-50:])) if s_history else np.nan
        if severe_available and np.isfinite(s50) and s50 > config.severe_score_threshold:
            severe_start = position if severe_start is None else severe_start
            if float(record.center_cycle - target.center_cycle.iloc[severe_start]) >= config.plateau_exit_confirm_cycles and not severe_events:
                severe_events.append({"protocol_id": protocol_id, "target_dataset": record.dataset, "event": "SEVERE_CANDIDATE_PERSISTENT", "cycle": float(record.center_cycle), "S_severe_candidate": severe_score})
        elif not severe_available or not np.isfinite(s50) or s50 <= config.severe_score_threshold:
            severe_start = None
        row: dict[str, object] = {**record.to_dict(), "protocol_id": protocol_id, "D_state": float(d_state[position]), "D_diagonal": float(d_diagonal[position]), "baseline_distance_method": context.baseline_method,
            "V20_norm": v20_norm, "V50_norm": v50_norm, "V100_norm": v100_norm, "direction_consistency": cosine, "persistent_direction_score": persistence,
            "A_state": acceleration, "A_smooth_20": a20, "A_smooth_50": a50, "state_volatility_20": vol20, "state_volatility_50": vol50,
            "plateau_candidate": plateau_candidate, "plateau_locked": plateau_locked_value, "plateau_confidence": float(candidate_condition), "plateau_lock_cycle": plateau_lock_cycle,
            "distance_from_plateau": distance_plateau, "instability_score": instability, "plateau_exit_candidate": exit_candidate, "plateau_exit_confirmed": exit_confirmed,
            "severe_direction_available": severe_available, "severe_direction_update": severe_update, "severe_direction_cosine_previous": cosine_previous,
            "S_severe_candidate": severe_score, "S_smooth_20": s20, "S_smooth_50": s50, "severe_status": "AVAILABLE" if severe_available else "UNAVAILABLE",
            "weighted_oos": weighted_oos, "support_confidence": support_confidence,
            "pre_update_D_state": float(d_state[position]), "pre_update_V50_norm": v50_norm, "pre_update_instability_score": instability, "pre_update_S_severe_candidate": severe_score,
            "online_update": severe_update}
        for name, a, b, c in zip(features, v20[position], v50[position], v100[position], strict=True):
            row[f"V20_{name}"] = float(a); row[f"V50_{name}"] = float(b); row[f"V100_{name}"] = float(c)
        rows.append(row)
    metadata = {"protocol_id": protocol_id, "target_dataset": str(target.dataset.iloc[0]), "baseline_median0": context.median0.tolist(), "baseline_scale0": context.scale0.tolist(), "baseline_window_count": int(context.baseline_mask.sum()), "plateau_status": "LOCKED" if plateau_locked else "NOT_FOUND", "plateau_lock_cycle": plateau_lock_cycle, "source_severe_prior_available": source_severe_direction is not None}
    return pd.DataFrame(rows), pd.DataFrame(plateau_events), pd.DataFrame(exit_events), pd.DataFrame(update_rows), {**metadata, "severe_events": severe_events, "severe_direction": None if severe_direction is None else severe_direction.tolist()}
