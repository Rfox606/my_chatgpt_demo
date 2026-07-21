from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CrossExperimentAdaptiveConfig
from .data import assert_formal_frame, temporal_pairs
from .model import RobustReference, SourceRanker, robust_reference


@dataclass(frozen=True)
class OnlineRun:
    scores: pd.DataFrame
    updates: pd.DataFrame
    source_frozen: bool


def _starts(cycles: np.ndarray, width: float) -> np.ndarray:
    return np.searchsorted(cycles, cycles - width, side="left")


def _speed(values: np.ndarray, cycles: np.ndarray, starts: np.ndarray) -> np.ndarray:
    index = np.arange(len(values)); elapsed = np.maximum(cycles - cycles[starts], 1e-9)
    result = np.linalg.norm(values - values[starts], axis=1) / elapsed * 100.0
    result[starts == index] = 0.0
    return result


def _volatility(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    answer = np.zeros(len(values), dtype=float)
    for index, start in enumerate(starts):
        answer[index] = float(np.sqrt(np.mean(np.var(values[start:index + 1], axis=0))))
    return answer


def _local_dynamics(values: np.ndarray, cycles: np.ndarray, calibration: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustReference]:
    reference = robust_reference(values[calibration])
    z = reference.transform(values)
    distance = np.sqrt(np.mean(z * z, axis=1))
    speed_1000 = _speed(z, cycles, _starts(cycles, 1000.0))
    speed_100 = _speed(z, cycles, _starts(cycles, 100.0))
    divergence = np.abs(speed_100 - speed_1000)
    volatility = _volatility(z, _starts(cycles, 500.0))
    return z, distance, speed_1000, divergence, volatility, reference


def _clip_step(current: np.ndarray, proposal: np.ndarray, config: CrossExperimentAdaptiveConfig) -> tuple[np.ndarray, float, bool]:
    step = proposal - current
    step_norm = float(np.linalg.norm(step))
    clipped = False
    if step_norm > config.adapter_max_step_norm:
        step *= config.adapter_max_step_norm / max(step_norm, 1e-12); clipped = True
    result = current + step
    norm = float(np.linalg.norm(result))
    if norm > config.adapter_max_norm:
        result *= config.adapter_max_norm / max(norm, 1e-12); clipped = True
    return result, float(np.linalg.norm(step)), clipped


def _adapter_gradient(
    source_z: np.ndarray,
    local_z: np.ndarray,
    source_coefficients: np.ndarray,
    residual: np.ndarray,
    pairs_early: np.ndarray,
    pairs_late: np.ndarray,
    lambda_t: float,
    l2: float,
) -> np.ndarray:
    if len(pairs_early) == 0 or lambda_t <= 0:
        return np.zeros_like(residual)
    delta_source = source_z[pairs_late] - source_z[pairs_early]
    delta_local = local_z[pairs_late] - local_z[pairs_early]
    margin = delta_source @ source_coefficients + lambda_t * (delta_local @ residual)
    # Gradient of logistic pair-order likelihood.  Only already arrived target pairs appear here.
    weight = 1.0 / (1.0 + np.exp(np.clip(margin, -40.0, 40.0)))
    return lambda_t * (weight[:, None] * delta_local).mean(axis=0) - l2 * residual


def _calibrated(raw: float, ranker: SourceRanker) -> float:
    return float(np.interp(raw, ranker.rank_knots, ranker.rank_values, left=0.0, right=1.0))


def run_target_online(
    target: pd.DataFrame,
    source_models: dict[str, SourceRanker],
    entry_cycle: float,
    config: CrossExperimentAdaptiveConfig,
) -> OnlineRun:
    """Causal target replay: emit pre-update outputs, then optionally update a bounded residual."""
    assert_formal_frame(target)
    primary = source_models[config.primary_feature_config]
    ordered = target.loc[target.center_cycle >= entry_cycle].sort_values(["center_cycle", "window_index"]).reset_index(drop=True).copy()
    if len(ordered) < 10:
        raise ValueError(f"Entry {entry_cycle} has insufficient target windows")
    cycles = ordered.center_cycle.to_numpy(float)
    calibration = cycles <= entry_cycle + config.target_initialization_cycles
    if calibration.sum() < 5:
        calibration[: min(5, len(calibration))] = True
    source_z = primary.z(ordered)
    priors = primary.progression_prior(ordered)
    ood_ratio = primary.ood_ratio(ordered)
    local_z, local_distance, speed_1000, divergence, volatility, local_reference = _local_dynamics(
        ordered.loc[:, list(primary.feature_names)].to_numpy(float), cycles, calibration
    )
    config_priors = np.vstack([model.progression_prior(ordered) for model in source_models.values()])
    config_dispersion = np.std(config_priors, axis=0)
    residual = np.zeros(len(primary.feature_names), dtype=float)
    frozen_coefficients = primary.coefficients.copy()
    previous_update_cycle = entry_cycle + config.target_initialization_cycles
    update_rows: list[dict[str, object]] = []; output_rows: list[dict[str, object]] = []
    for position in range(len(ordered)):
        cycle = float(cycles[position])
        evidence_cycles = max(0.0, cycle - (entry_cycle + config.target_initialization_cycles))
        lambda_t = min(config.lambda_max, config.lambda_max * evidence_cycles / max(config.lambda_ramp_cycles, 1e-9))
        residual_score = float(local_z[position] @ residual)
        adapted_raw = float(source_z[position] @ frozen_coefficients + lambda_t * residual_score)
        progression_adapted = _calibrated(adapted_raw, primary)
        local_progression = float(np.tanh(local_distance[position] / 3.0))
        activity = float(np.tanh((speed_1000[position] + divergence[position] + volatility[position]) / 3.0))
        adapter_norm = float(np.linalg.norm(residual))
        adapter_boundary = adapter_norm / max(config.adapter_max_norm, 1e-9)
        pair_count_before = 0
        update_applied = False; reason = "cadence_not_reached"; step_norm = 0.0; latest_pair_cycle = float("nan")
        if cycle <= entry_cycle + config.target_initialization_cycles:
            reason = "initialization_freeze"
        # A different experiment can legitimately be near the source support edge.
        # Only a severe excursion suppresses an update; the continuous OOD component
        # still raises uncertainty before this hard gate is reached.
        elif ood_ratio[position] > 2.5:
            reason = "ood_suppressed"
        elif volatility[position] > config.high_volatility_gate:
            reason = "high_volatility_suppressed"
        elif cycle - previous_update_cycle < config.target_update_interval_cycles:
            reason = "cadence_not_reached"
        else:
            # Exclude the currently emitted row: the update only receives the arrived prefix before it.
            history = ordered.iloc[:position].copy()
            pairs = temporal_pairs(history, config.source_gap_bins, config.target_update_pair_limit, seed=config.random_seed + position)
            pair_count_before = pairs.count
            if pairs.count == 0:
                reason = "insufficient_target_pairs"
            else:
                gradient = _adapter_gradient(source_z[:position], local_z[:position], frozen_coefficients, residual, pairs.earlier, pairs.later, lambda_t, config.target_l2)
                proposal = residual + config.target_learning_rate * gradient
                candidate, step_norm, clipped = _clip_step(residual, proposal, config)
                latest_pair_cycle = float(history.center_cycle.iloc[pairs.later].max())
                # A rejected/no-op cadence is still an observed decision; retry only
                # after another fixed interval rather than repeatedly at every window.
                previous_update_cycle = cycle
                if np.allclose(candidate, residual, atol=1e-12, rtol=0):
                    reason = "constant_or_regularized"
                else:
                    residual = candidate; update_applied = True
                    reason = "updated_clipped" if clipped else "updated"
        gate_uncertainty = 1.0 if reason in {"ood_suppressed", "high_volatility_suppressed"} else 0.0
        pair_uncertainty = 1.0 / (1.0 + pair_count_before / 30.0)
        uncertainty_components = {
            "uncertainty_feature_config_dispersion": min(1.0, float(config_dispersion[position] * 4.0)),
            "uncertainty_ood": min(1.0, max(0.0, float(ood_ratio[position] - .5) / 2.0)),
            "uncertainty_target_pair_evidence": pair_uncertainty,
            "uncertainty_prior_adapted_disagreement": abs(float(priors[position]) - progression_adapted),
            "uncertainty_adapter_boundary": min(1.0, adapter_boundary),
            "uncertainty_local_volatility_or_gate": min(1.0, volatility[position] / config.high_volatility_gate + gate_uncertainty * .5),
        }
        state_uncertainty = float(np.mean(list(uncertainty_components.values())))
        output_rows.append({
            "dataset": str(ordered.dataset.iloc[position]), "entry_cycle": entry_cycle, "window_id": ordered.window_id.iloc[position], "window_index": ordered.window_index.iloc[position],
            "center_cycle": cycle, "elapsed_time_since_entry": cycle - entry_cycle,
            "progression_prior": float(priors[position]), "progression_adapted": progression_adapted, "progression_score": progression_adapted,
            "target_local_score": local_progression, "elapsed_time_since_entry_score": float((cycle - entry_cycle) / max(cycles[-1] - entry_cycle, 1e-9)),
            "local_distance": float(local_distance[position]), "local_speed_1000": float(speed_1000[position]),
            "local_rate_divergence_100_1000": float(divergence[position]), "local_volatility_500": float(volatility[position]), "activity_score_local": activity, "activity_score": activity,
            "adapter_enabled": int(cycle > entry_cycle + config.target_initialization_cycles), "adapter_update_applied": int(update_applied), "adapter_update_reason": reason,
            "adapter_parameter_norm": adapter_norm, "target_pair_count": pair_count_before, "target_pair_latest_cycle": latest_pair_cycle, "lambda_t": lambda_t,
            "ood_ratio": float(ood_ratio[position]), "state_uncertainty": state_uncertainty, "source_frozen_hash": primary.frozen_hash,
            **uncertainty_components,
        })
        update_rows.append({"dataset": str(ordered.dataset.iloc[position]), "entry_cycle": entry_cycle, "center_cycle": cycle, "adapter_update_applied": int(update_applied),
                            "adapter_update_reason": reason, "adapter_parameter_norm_after": float(np.linalg.norm(residual)), "adapter_step_norm": step_norm,
                            "target_pair_count": pair_count_before, "target_pair_latest_cycle": latest_pair_cycle, "lambda_t": lambda_t, "ood_ratio": float(ood_ratio[position])})
    frozen = bool(np.array_equal(frozen_coefficients, primary.coefficients))
    return OnlineRun(pd.DataFrame(output_rows), pd.DataFrame(update_rows), frozen)


def source_model_hashes(models: dict[str, SourceRanker]) -> dict[str, str]:
    return {name: hashlib.sha256(model.coefficients.astype(np.float64).tobytes()).hexdigest() for name, model in models.items()}
