from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import GradualStateMonitoringV33Config
from .features import CausalDescriptorScaler


class RecursiveMultiOutputRidge:
    """Predict-then-update RLS predictor with a configurable forgetting speed."""

    def __init__(self, dimensions: int, outputs: int, forgetting: float, initial_covariance: float, min_updates: int) -> None:
        self.dimensions = dimensions + 1
        self.outputs = outputs
        self.forgetting = forgetting
        self.theta = np.zeros((self.dimensions, outputs), dtype=float)
        self.covariance = np.eye(self.dimensions, dtype=float) * initial_covariance
        self.updates = 0
        self.min_updates = min_updates

    def _augment(self, row: np.ndarray) -> np.ndarray:
        return np.concatenate((np.asarray(row, dtype=float), np.ones(1)))

    def predict(self, row: np.ndarray) -> np.ndarray:
        if self.updates < self.min_updates:
            return np.full(self.outputs, np.nan)
        return self._augment(row) @ self.theta

    def update(self, row: np.ndarray, target: np.ndarray) -> None:
        x = self._augment(row)
        px = self.covariance @ x
        gain = px / max(self.forgetting + float(x @ px), 1e-12)
        error = np.asarray(target, dtype=float) - x @ self.theta
        self.theta += np.outer(gain, error)
        self.covariance = (self.covariance - np.outer(gain, x @ self.covariance)) / self.forgetting
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        self.updates += 1


class DualTimescaleForecaster:
    """Parallel slow/fast RLS models; a label becomes available only at its due time."""

    def __init__(self, descriptor_dimensions: int, output_dimensions: int, config: GradualStateMonitoringV33Config) -> None:
        self.config = config
        self.scaler = CausalDescriptorScaler(descriptor_dimensions, config.eps)
        self.slow = RecursiveMultiOutputRidge(descriptor_dimensions, output_dimensions, config.slow_forgetting_factor, config.rls_initial_covariance, config.rls_min_updates)
        self.fast = RecursiveMultiOutputRidge(descriptor_dimensions, output_dimensions, config.fast_forgetting_factor, config.rls_initial_covariance, config.rls_min_updates)
        self.scaled_descriptors: list[np.ndarray] = []
        self.slow_pending: list[np.ndarray] = []
        self.fast_pending: list[np.ndarray] = []

    def step(self, descriptor: np.ndarray, target: np.ndarray, index: int) -> dict[str, object]:
        x = self.scaler.transform(descriptor)
        slow_prediction = self.slow.predict(x)
        fast_prediction = self.fast.predict(x)
        self.scaled_descriptors.append(x.copy())
        self.slow_pending.append(slow_prediction.copy())
        self.fast_pending.append(fast_prediction.copy())
        due_origin = index - self.config.dual_horizon
        slow_error = np.nan
        fast_error = np.nan
        if due_origin >= 0:
            due_slow = self.slow_pending[due_origin]
            due_fast = self.fast_pending[due_origin]
            if np.isfinite(due_slow).all():
                slow_error = float(np.mean(np.abs(target - due_slow)))
            if np.isfinite(due_fast).all():
                fast_error = float(np.mean(np.abs(target - due_fast)))
            self.slow.update(self.scaled_descriptors[due_origin], target)
            self.fast.update(self.scaled_descriptors[due_origin], target)
        self.scaler.update(descriptor)
        difference = float(np.mean(np.abs(slow_prediction - fast_prediction))) if np.isfinite(slow_prediction).all() and np.isfinite(fast_prediction).all() else np.nan
        return {
            "dual_horizon": self.config.dual_horizon,
            "due_origin_index": due_origin if due_origin >= 0 else np.nan,
            "slow_prediction": slow_prediction,
            "fast_prediction": fast_prediction,
            "slow_prediction_error": slow_error,
            "fast_prediction_error": fast_error,
            "fast_slow_prediction_difference": difference,
        }


def run_dual_timescale(descriptors: np.ndarray, values: np.ndarray, config: GradualStateMonitoringV33Config) -> pd.DataFrame:
    engine = DualTimescaleForecaster(descriptors.shape[1], values.shape[1], config)
    rows: list[dict[str, object]] = []
    for index, (descriptor, target) in enumerate(zip(descriptors, values)):
        result = engine.step(descriptor, target, index)
        row: dict[str, object] = {"input_index": index}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                row[f"{key}_mean"] = float(np.mean(value)) if np.isfinite(value).all() else np.nan
                for component, item in enumerate(value):
                    row[f"{key}_{component}"] = float(item)
            else:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _make_model(model_name: str, config: GradualStateMonitoringV33Config) -> object:
    if model_name == "Ridge":
        return Pipeline((
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=config.ridge_alpha)),
        ))
    if model_name == "RBF_Ridge":
        return Pipeline((
            ("scale", StandardScaler()),
            ("rbf", RBFSampler(gamma=config.rbf_gamma, n_components=config.rbf_components, random_state=config.random_seed)),
            ("ridge", Ridge(alpha=config.ridge_alpha)),
        ))
    raise ValueError(f"Unknown learned benchmark model: {model_name}")


def run_causal_benchmark_forecasts(descriptors: np.ndarray, values: np.ndarray, config: GradualStateMonitoringV33Config, feature_names: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Forecast all horizons from causal, repeatedly refit Ridge and RBF-Ridge models.

    At origin t, a training pair is eligible only when its target had arrived strictly
    before t.  Target values appended after the sequence never alter earlier forecasts.
    """
    length = len(values)
    rows: list[dict[str, object]] = []
    learned_names = ("Ridge", "RBF_Ridge")
    cache: dict[tuple[str, int], object | None] = {(name, horizon): None for name in learned_names for horizon in config.horizons}
    for origin in range(length):
        for horizon in config.horizons:
            target_index = origin + horizon
            persistence = values[origin].copy()
            rows.append({
                "input_index": origin, "target_index": target_index, "horizon": horizon,
                "model": "Persistence", "prediction": persistence,
            })
            upper = origin - horizon - 1  # strict predict-then-update cutoff
            lower = max(0, upper - config.benchmark_training_window + 1)
            eligible = upper - lower + 1
            for name in learned_names:
                key = (name, horizon)
                refit = origin % config.benchmark_refit_interval == 0 or cache[key] is None
                if refit and eligible >= config.benchmark_min_train:
                    model = _make_model(name, config)
                    # Learn the horizon change rather than the absolute level.
                    # Persistence is therefore the explicit zero-change baseline,
                    # while the learned models only need to explain systematic drift.
                    model.fit(
                        descriptors[lower:upper + 1],
                        values[lower + horizon:upper + horizon + 1] - values[lower:upper + 1],
                    )
                    cache[key] = model
                model = cache[key]
                prediction = values[origin] + model.predict(descriptors[origin:origin + 1])[0] if model is not None else np.full(values.shape[1], np.nan)
                rows.append({
                    "input_index": origin, "target_index": target_index, "horizon": horizon,
                    "model": name, "prediction": prediction,
                })
    output = pd.DataFrame(rows)
    valid = output.target_index.to_numpy(int) < length
    actual = np.full((len(output), values.shape[1]), np.nan)
    actual[valid] = values[output.loc[valid, "target_index"].to_numpy(int)]
    prediction = np.vstack(output.prediction.to_numpy())
    output["mae"] = np.mean(np.abs(prediction - actual), axis=1)
    output["prediction_available"] = np.isfinite(prediction).all(axis=1).astype(int)
    output["target_available"] = valid.astype(int)
    names = feature_names or tuple(f"feature_{index}" for index in range(values.shape[1]))
    for component, name in enumerate(names):
        output[f"prediction_{name}"] = prediction[:, component]
        output[f"actual_{name}"] = actual[:, component]
    output = output.drop(columns="prediction")
    return output


def run_source_initialised_forecasts(source_descriptors: np.ndarray, source_values: np.ndarray, target_descriptors: np.ndarray, target_values: np.ndarray, config: GradualStateMonitoringV33Config, feature_names: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Evaluate a prior-experiment initial model on the target's early causal prefix.

    The source sequence is wholly earlier than the target experiment.  No target
    observations after an origin are used in the fitted scaler or predictor.
    """
    rows: list[dict[str, object]] = []
    names = ("Ridge", "RBF_Ridge")
    feature_names = feature_names or tuple(f"feature_{index}" for index in range(target_values.shape[1]))
    for horizon in config.horizons:
        upper = len(source_values) - horizon
        lower = max(0, upper - config.source_initialisation_training_window)
        if upper - lower < config.benchmark_min_train:
            continue
        for name in names:
            model = _make_model(name, config)
            model.fit(
                source_descriptors[lower:upper],
                source_values[lower + horizon:upper + horizon] - source_values[lower:upper],
            )
            for origin in range(min(config.source_initialisation_early_windows, len(target_values))):
                target_index = origin + horizon
                if target_index >= len(target_values):
                    continue
                prediction = target_values[origin] + model.predict(target_descriptors[origin:origin + 1])[0]
                actual = target_values[target_index]
                row: dict[str, object] = {
                    "input_index": origin, "target_index": target_index, "horizon": horizon,
                    "model": f"SourceInit_{name}", "mae": float(np.mean(np.abs(prediction - actual))),
                    "prediction_available": 1, "target_available": 1,
                }
                for component, feature in enumerate(feature_names):
                    row[f"prediction_{feature}"] = float(prediction[component])
                    row[f"actual_{feature}"] = float(actual[component])
                rows.append(row)
    return pd.DataFrame(rows)
