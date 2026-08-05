from __future__ import annotations

from collections import deque

import numpy as np

from .config import GradualStateMonitoringV33Config
from .features import CausalFeatureScale


class CausalRobustBaseline:
    """A trailing online median/MAD baseline queried before the new value is added."""

    def __init__(self, capacity: int, minimum: int, eps: float, refresh: int) -> None:
        self.values: deque[float] = deque(maxlen=capacity)
        self.minimum = minimum
        self.eps = eps
        self.refresh = refresh
        self.queries_since_refresh = 0
        self.cached_median = np.nan
        self.cached_scale = np.nan

    def score_then_update(self, value: float) -> tuple[float, float, float]:
        score, median, scale = self.score(value)
        if np.isfinite(value):
            self.values.append(float(value))
            self.queries_since_refresh += 1
        return score, median, scale

    def score(self, value: float) -> tuple[float, float, float]:
        if not np.isfinite(value) or len(self.values) < self.minimum:
            return 0.0, np.nan, np.nan
        if not np.isfinite(self.cached_median) or self.queries_since_refresh >= self.refresh:
            history = np.asarray(self.values, dtype=float)
            self.cached_median = float(np.median(history))
            mad = float(np.median(np.abs(history - self.cached_median)))
            iqr = float(np.percentile(history, 75) - np.percentile(history, 25))
            self.cached_scale = max(1.4826 * mad, iqr / 1.349, self.eps)
            self.queries_since_refresh = 0
        return max(0.0, (float(value) - self.cached_median) / self.cached_scale), self.cached_median, self.cached_scale


class CausalEvidenceEngine:
    """Three independent online evidence channels and their configured fusion."""

    def __init__(self, feature_count: int, config: GradualStateMonitoringV33Config) -> None:
        self.config = config
        self.scale = CausalFeatureScale(feature_count, config.history_windows, config.eps)
        self.error_baseline = CausalRobustBaseline(config.evidence_baseline_windows, config.evidence_min_history, config.eps, config.evidence_baseline_refresh)
        self.fast_error_baseline = CausalRobustBaseline(config.evidence_baseline_windows, config.evidence_min_history, config.eps, config.evidence_baseline_refresh)
        self.shift_baseline = CausalRobustBaseline(config.evidence_baseline_windows, config.evidence_min_history, config.eps, config.evidence_baseline_refresh)
        self.velocity_baseline = CausalRobustBaseline(config.evidence_baseline_windows, config.evidence_min_history, config.eps, config.evidence_baseline_refresh)
        # Only the two adjacent 32-window intervals are needed.  A bounded buffer
        # also prevents the online calculation from accidentally becoming O(T²).
        self.history: deque[np.ndarray] = deque(maxlen=64)

    def _raw_distribution_shift(self) -> float:
        if len(self.history) < 64 or not self.scale.frozen:
            return np.nan
        observed = np.vstack(tuple(self.history))
        earlier = observed[-64:-32]
        recent = observed[-32:]
        location = np.mean(np.abs(recent.mean(axis=0) - earlier.mean(axis=0)) / self.scale.scale)
        earlier_std = np.maximum(earlier.std(axis=0), self.config.eps)
        recent_std = np.maximum(recent.std(axis=0), self.config.eps)
        spread = np.mean(np.abs(np.log(recent_std / earlier_std)))
        return float(location + 0.25 * spread)

    def _raw_velocity(self) -> float:
        if len(self.history) < 33 or not self.scale.frozen:
            return np.nan
        return float(np.mean(np.abs((self.history[-1] - self.history[-33]) / self.scale.scale)))

    def step(self, value: np.ndarray, slow_error: float, fast_error: float) -> dict[str, float]:
        """Calculate scores before adding the current evidence to each online baseline."""
        raw_shift = self._raw_distribution_shift()
        raw_velocity = self._raw_velocity()
        prediction_surprise, error_median, error_scale = self.error_baseline.score_then_update(slow_error)
        fast_surprise, fast_median, fast_scale = self.fast_error_baseline.score_then_update(fast_error)
        distribution_shift, shift_median, shift_scale = self.shift_baseline.score_then_update(raw_shift)
        drift_velocity, velocity_median, velocity_scale = self.velocity_baseline.score_then_update(raw_velocity)
        weights = self.config.evidence_weights
        change = (
            weights["prediction_surprise"] * prediction_surprise
            + weights["distribution_shift"] * distribution_shift
            + weights["drift_velocity"] * drift_velocity
        )
        self.history.append(np.asarray(value, dtype=float).copy())
        self.scale.update(value)
        return {
            "prediction_surprise": prediction_surprise,
            "distribution_shift": distribution_shift,
            "drift_velocity": drift_velocity,
            "change_evidence": float(change),
            "fast_prediction_surprise": fast_surprise,
            "raw_distribution_shift": raw_shift,
            "raw_drift_velocity": raw_velocity,
            "prediction_error_online_median": error_median,
            "prediction_error_online_scale": error_scale,
            "fast_error_online_median": fast_median,
            "fast_error_online_scale": fast_scale,
            "distribution_shift_online_median": shift_median,
            "distribution_shift_online_scale": shift_scale,
            "drift_velocity_online_median": velocity_median,
            "drift_velocity_online_scale": velocity_scale,
            "feature_scale_frozen": int(self.scale.frozen),
        }
