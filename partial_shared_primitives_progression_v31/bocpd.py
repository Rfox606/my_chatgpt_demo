from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import V31Config


def _student_logpdf(value: float, mean: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    df = 2 * alpha; scale2 = beta * (kappa + 1) / np.maximum(alpha * kappa, 1e-12); z = (value - mean) ** 2 / np.maximum(df * scale2, 1e-12)
    return np.asarray([math.lgamma((d + 1) / 2) - math.lgamma(d / 2) - .5 * (math.log(d * math.pi) + math.log(s)) - (d + 1) / 2 * math.log1p(zz) for d, s, zz in zip(df, scale2, z)])


def _posterior_update(value: float, mean: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    next_kappa = kappa + 1; next_mean = (kappa * mean + value) / next_kappa; next_alpha = alpha + .5; next_beta = beta + kappa * (value - mean) ** 2 / (2 * next_kappa)
    return next_mean, next_kappa, next_alpha, next_beta


def bocpd_confirmed_segments(energy: np.ndarray, cycles: np.ndarray, config: V31Config) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Causal BOCPD with Student-t predictive and delayed boundary confirmation."""
    probability = np.asarray([1.0]); mean = np.asarray([0.0]); kappa = np.asarray([1.0]); alpha = np.asarray([1.0]); beta = np.asarray([1.0]); high = 0; start = 0; segments: list[tuple[int, int]] = []; rows: list[dict[str, object]] = []
    for index, value in enumerate(np.asarray(energy, dtype=float)):
        likelihood = np.exp(np.clip(_student_logpdf(float(value), mean, kappa, alpha, beta), -700, 100)); change = float(np.sum(probability * likelihood * config.bocpd_hazard)); growth = probability * likelihood * (1 - config.bocpd_hazard)
        next_probability = np.r_[change, growth][:config.bocpd_max_run_length + 1]; total = max(float(next_probability.sum()), 1e-300); next_probability /= total
        prior_mean = np.asarray([0.0]); prior_kappa = np.asarray([1.0]); prior_alpha = np.asarray([1.0]); prior_beta = np.asarray([1.0]); cp_params = _posterior_update(float(value), prior_mean, prior_kappa, prior_alpha, prior_beta)
        growth_params = _posterior_update(float(value), mean, kappa, alpha, beta)
        mean = np.r_[cp_params[0], growth_params[0]][:len(next_probability)]; kappa = np.r_[cp_params[1], growth_params[1]][:len(next_probability)]; alpha = np.r_[cp_params[2], growth_params[2]][:len(next_probability)]; beta = np.r_[cp_params[3], growth_params[3]][:len(next_probability)]; probability = next_probability
        cp_posterior = float(probability[0]); high = high + 1 if cp_posterior >= config.bocpd_confirmation_posterior else 0; confirmed = False
        if high >= config.bocpd_confirmation_windows:
            boundary = index - config.bocpd_confirmation_windows + 1
            if boundary - start >= config.minimum_segment_windows:
                segments.append((start, boundary)); start = boundary + 1; confirmed = True
            high = 0
        entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))
        rows.append({"window_index": index, "center_cycle": float(cycles[index]), "innovation_energy": float(value), "bocpd_change_posterior": cp_posterior, "bocpd_run_length_map": int(probability.argmax()), "bocpd_run_length_entropy": entropy, "boundary_confirmed": confirmed})
    if len(energy) - start >= config.minimum_segment_windows: segments.append((start, len(energy) - 1))
    return pd.DataFrame(rows), segments


def causal_activity_energy(frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    values = frame.loc[:, list(features)].to_numpy(float); delta = np.vstack((np.zeros(values.shape[1]), np.diff(values, axis=0))); return np.sqrt(np.mean(delta ** 2, axis=1))


def segment_descriptors(frame: pd.DataFrame, energy: np.ndarray, activity: np.ndarray, segments: list[tuple[int, int]], features: tuple[str, ...]) -> pd.DataFrame:
    values = frame.loc[:, list(features)].to_numpy(float); rows: list[dict[str, object]] = []
    for segment_id, (start, end) in enumerate(segments):
        part = values[start:end + 1]; coordinate = np.arange(len(part), dtype=float); slope = np.polyfit(coordinate, part, 1)[0] if len(part) >= 2 else np.zeros(part.shape[1])
        row: dict[str, object] = {"segment_id": segment_id, "start_index": start, "end_index": end, "start_cycle": float(frame.center_cycle.iloc[start]), "end_cycle": float(frame.center_cycle.iloc[end]), "window_count": int(end - start + 1), "mean_innovation_energy": float(np.mean(energy[start:end + 1])), "mean_activity_energy": float(np.mean(activity[start:end + 1]))}
        row.update({f"mean_{name}": float(value) for name, value in zip(features, part.mean(axis=0))}); row.update({f"slope_{name}": float(value) for name, value in zip(features, slope)}); rows.append(row)
    return pd.DataFrame(rows)

