from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import V32Config


def _student_logpdf(
    value: float, mean: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Student-t predictive density for Normal-Gamma sufficient statistics."""
    df = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / np.maximum(alpha * kappa, 1e-12)
    z = (value - mean) ** 2 / np.maximum(df * scale2, 1e-12)
    return np.asarray(
        [
            math.lgamma((d + 1.0) / 2.0)
            - math.lgamma(d / 2.0)
            - 0.5 * (math.log(d * math.pi) + math.log(s))
            - (d + 1.0) / 2.0 * math.log1p(zz)
            for d, s, zz in zip(df, scale2, z)
        ],
        dtype=float,
    )


def _posterior_update(
    value: float, mean: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    next_kappa = kappa + 1.0
    next_mean = (kappa * mean + value) / next_kappa
    next_alpha = alpha + 0.5
    next_beta = beta + kappa * (value - mean) ** 2 / (2.0 * next_kappa)
    return next_mean, next_kappa, next_alpha, next_beta


def bocpd_confirmed_segments(
    energy: np.ndarray, cycles: np.ndarray, config: V32Config
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Causal BOCPD using distinct new-segment and growth predictives.

    The r=0 branch evaluates the *prior* Student-t predictive.  Every growth
    branch evaluates its own run-length posterior predictive.  This is the
    distinction lost in v3.1, where a common likelihood made P(r=0) exactly
    the constant hazard after normalisation.
    """
    posterior = np.asarray([1.0], dtype=float)
    mean = np.asarray([0.0], dtype=float)
    kappa = np.asarray([1.0], dtype=float)
    alpha = np.asarray([1.0], dtype=float)
    beta = np.asarray([1.0], dtype=float)
    prior_mean = np.asarray([0.0], dtype=float)
    prior_kappa = np.asarray([1.0], dtype=float)
    prior_alpha = np.asarray([1.0], dtype=float)
    prior_beta = np.asarray([1.0], dtype=float)

    high = 0
    start = 0
    segments: list[tuple[int, int]] = []
    rows: list[dict[str, object]] = []
    for index, value in enumerate(np.asarray(energy, dtype=float)):
        # Change-point branch: p(x_t | new segment prior), one scalar shared
        # only after integrating over r_(t-1).
        prior_log_likelihood = float(
            _student_logpdf(float(value), prior_mean, prior_kappa, prior_alpha, prior_beta)[0]
        )
        prior_likelihood = math.exp(float(np.clip(prior_log_likelihood, -700.0, 100.0)))
        # Growth branches: each run length has its own posterior predictive.
        growth_log_likelihood = _student_logpdf(float(value), mean, kappa, alpha, beta)
        growth_likelihood = np.exp(np.clip(growth_log_likelihood, -700.0, 100.0))

        change_mass = float(np.sum(posterior) * config.bocpd_hazard * prior_likelihood)
        growth_mass = posterior * (1.0 - config.bocpd_hazard) * growth_likelihood
        next_posterior = np.r_[change_mass, growth_mass][: config.bocpd_max_run_length + 1]
        normalizer = max(float(next_posterior.sum()), 1e-300)
        next_posterior /= normalizer

        change_params = _posterior_update(float(value), prior_mean, prior_kappa, prior_alpha, prior_beta)
        growth_params = _posterior_update(float(value), mean, kappa, alpha, beta)
        mean = np.r_[change_params[0], growth_params[0]][: len(next_posterior)]
        kappa = np.r_[change_params[1], growth_params[1]][: len(next_posterior)]
        alpha = np.r_[change_params[2], growth_params[2]][: len(next_posterior)]
        beta = np.r_[change_params[3], growth_params[3]][: len(next_posterior)]
        posterior = next_posterior

        cp_posterior = float(posterior[0])
        high = high + 1 if cp_posterior >= config.bocpd_confirmation_posterior else 0
        confirmed = False
        if high >= config.bocpd_confirmation_windows:
            boundary = index - config.bocpd_confirmation_windows + 1
            if boundary - start >= config.minimum_segment_windows:
                segments.append((start, boundary))
                start = boundary + 1
                confirmed = True
            high = 0
        entropy = float(-np.sum(posterior * np.log(np.maximum(posterior, 1e-12))))
        rows.append(
            {
                "window_index": index,
                "center_cycle": float(cycles[index]),
                "innovation_energy": float(value),
                "bocpd_change_posterior": cp_posterior,
                "bocpd_prior_predictive": prior_likelihood,
                "bocpd_growth_predictive_mean": float(np.sum(posterior[1:] * growth_likelihood[: len(posterior) - 1])) if len(posterior) > 1 else 0.0,
                "bocpd_run_length_map": int(posterior.argmax()),
                "bocpd_run_length_entropy": entropy,
                "boundary_confirmed": confirmed,
            }
        )
    if len(energy) - start >= config.minimum_segment_windows:
        segments.append((start, len(energy) - 1))
    return pd.DataFrame(rows), segments


def causal_activity_energy(frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    values = frame.loc[:, list(features)].to_numpy(float)
    delta = np.vstack((np.zeros(values.shape[1]), np.diff(values, axis=0)))
    # A fixed causal robust scale from the first available window block avoids
    # domination by a channel's units and never sees appended future samples.
    calibration = delta[: min(32, len(delta))]
    scale = np.maximum(1.4826 * np.median(np.abs(calibration - np.median(calibration, axis=0)), axis=0), 1e-9)
    return np.sqrt(np.mean((delta / scale) ** 2, axis=1))


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def segment_descriptors(
    frame: pd.DataFrame,
    prediction_error: np.ndarray,
    activity: np.ndarray,
    segments: list[tuple[int, int]],
    features: tuple[str, ...],
) -> pd.DataFrame:
    """One descriptor row per confirmed segment, never per raw window."""
    values = frame.loc[:, list(features)].to_numpy(float)
    rows: list[dict[str, object]] = []
    for segment_id, (start, end) in enumerate(segments):
        part = values[start : end + 1]
        errors = np.asarray(prediction_error[start : end + 1], dtype=float)
        local_activity = np.asarray(activity[start : end + 1], dtype=float)
        half = max(1, len(part) // 2)
        start_volatility = float(np.std(part[:half], axis=0).mean())
        end_volatility = float(np.std(part[-half:], axis=0).mean())
        entry_jump = float(activity[start] - activity[start - 1]) if start > 0 else 0.0
        leave_jump = float(activity[end + 1] - activity[end]) if end + 1 < len(activity) else 0.0
        row: dict[str, object] = {
            "segment_id": segment_id,
            "start_index": int(start),
            "end_index": int(end),
            "start_cycle": float(frame.center_cycle.iloc[start]),
            "end_cycle": float(frame.center_cycle.iloc[end]),
            "window_count": int(len(part)),
            "descriptor_duration": float(frame.center_cycle.iloc[end] - frame.center_cycle.iloc[start]),
            "descriptor_start_mean": float(part[0].mean()),
            "descriptor_end_mean": float(part[-1].mean()),
            "descriptor_net_change": float((part[-1] - part[0]).mean()),
            "descriptor_overall_slope": _slope(part.mean(axis=1)),
            "descriptor_first_half_slope": _slope(part[:half].mean(axis=1)),
            "descriptor_second_half_slope": _slope(part[-half:].mean(axis=1)),
            "descriptor_volatility_start": start_volatility,
            "descriptor_volatility_end": end_volatility,
            "descriptor_volatility_change": end_volatility - start_volatility,
            "descriptor_prediction_error_mean": float(np.mean(errors)),
            "descriptor_prediction_error_trend": _slope(errors),
            "descriptor_entry_jump": entry_jump,
            "descriptor_exit_jump": leave_jump,
            "mean_innovation_energy": float(np.mean(errors)),
            "mean_activity_energy": float(np.mean(local_activity)),
        }
        # Feature-level means/slopes aid interpretation, while the clustering
        # input remains descriptor_ columns only.
        for number, name in enumerate(features):
            row[f"mean_{name}"] = float(part[:, number].mean())
            row[f"slope_{name}"] = _slope(part[:, number])
        rows.append(row)
    return pd.DataFrame(rows)
