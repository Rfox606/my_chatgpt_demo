from __future__ import annotations

"""Cycle-scale causal forecasts and a Safe Gate against the best static baseline."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import ContinuousStateV41Config
from .data import assert_label_free


TARGETS = ("D_state", "V500_norm", "A_state", "residual_change_score")
INPUTS = (
    "D_state", "V100_norm", "V500_norm", "V1000_norm", "direction_consistency",
    "A_state", "state_volatility", "baseline_outlier_fraction", "source_support_oos", "residual_change_score", "abrupt_cusum",
)
STATIC_MODELS = ("Zero_Delta", "Local_Linear", "Kalman", "Frozen_Ridge")
ALL_MODELS = (*STATIC_MODELS, "Online_RLS", "Safe_Gate")


def _due_indices(cycles: np.ndarray, horizon: int) -> np.ndarray:
    return np.searchsorted(cycles, cycles + horizon, side="left")


def _feature_vector(values: np.ndarray, position: int, history: int) -> np.ndarray | None:
    if position + 1 < history:
        return None
    sample = values[position - history + 1:position + 1]
    if not np.isfinite(sample).all():
        return None
    return np.r_[1.0, sample.mean(axis=0), sample.std(axis=0), sample[-1] - sample[0]]


def _local_linear(cycles: np.ndarray, response: np.ndarray, position: int, history: int, horizon: int) -> float:
    start = max(0, position - history + 1)
    x = cycles[start:position + 1]
    y = response[start:position + 1]
    good = np.isfinite(y)
    if good.sum() < 3:
        return float(response[position])
    slope = float(np.median(np.diff(y[good]) / np.maximum(np.diff(x[good]), 1e-9)))
    return float(response[position] + slope * horizon)


@dataclass
class LocalTrendKalman:
    level: float | None = None
    trend_per_cycle: float = 0.0
    covariance: np.ndarray | None = None
    last_cycle: float | None = None

    def update(self, observation: float, cycle: float) -> None:
        if not np.isfinite(observation):
            return
        if self.level is None:
            self.level, self.last_cycle, self.covariance = float(observation), float(cycle), np.diag([1.0, 1.0])
            return
        assert self.covariance is not None and self.last_cycle is not None
        dt = max(float(cycle) - self.last_cycle, 1.0)
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        process = np.diag([1e-4 * dt * dt, 1e-7 * dt])
        state = transition @ np.array([self.level, self.trend_per_cycle])
        predicted = transition @ self.covariance @ transition.T + process
        design = np.array([1.0, 0.0])
        gain = predicted @ design / (float(design @ predicted @ design) + 1e-3)
        state += gain * (float(observation) - float(design @ state))
        self.covariance = (np.eye(2) - np.outer(gain, design)) @ predicted
        self.level, self.trend_per_cycle, self.last_cycle = float(state[0]), float(state[1]), float(cycle)

    def forecast(self, horizon: int, fallback: float) -> float:
        return float(self.level + self.trend_per_cycle * horizon) if self.level is not None else float(fallback)


@dataclass
class ScalarRLS:
    theta: np.ndarray
    covariance: np.ndarray
    frozen_theta: np.ndarray
    forgetting: float
    theta_norm_max: float
    covariance_max: float
    delta_clip: float

    def predict(self, vector: np.ndarray) -> float:
        return float(np.clip(vector @ self.theta, -self.delta_clip, self.delta_clip))

    def update(self, vector: np.ndarray, observed: float, clip: float, gain_max: float) -> None:
        x = vector.reshape(-1, 1)
        denominator = self.forgetting + float((x.T @ self.covariance @ x).item())
        gain = (self.covariance @ x / max(denominator, 1e-12)).reshape(-1)
        norm = float(np.linalg.norm(gain))
        if norm > gain_max:
            gain *= gain_max / norm
        residual = float(np.clip(observed - vector @ self.theta, -clip, clip))
        self.theta += gain * residual
        norm_theta = float(np.linalg.norm(self.theta))
        if norm_theta > self.theta_norm_max:
            self.theta *= self.theta_norm_max / norm_theta
        covariance = (self.covariance - np.outer(gain, vector) @ self.covariance) / self.forgetting
        self.covariance = np.clip(np.nan_to_num((covariance + covariance.T) / 2.0, nan=0.0, posinf=self.covariance_max, neginf=-self.covariance_max), -self.covariance_max, self.covariance_max)


def train_frozen_models(source_states: pd.DataFrame, config: ContinuousStateV41Config) -> dict[tuple[str, int], Ridge | None]:
    """Fit frozen Ridge models from source-only, label-free trajectories."""
    assert_label_free(source_states)
    states = source_states.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    stride = max(float(states.nominal_stride_cycles.iloc[0]), config.eps)
    step = max(1, int(round(config.forecast_issue_stride_cycles / stride)))
    states = states.iloc[::step].reset_index(drop=True)
    values = states.loc[:, list(INPUTS)].to_numpy(float)
    cycles = states.center_cycle.to_numpy(float)
    vectors = [_feature_vector(values, index, config.forecast_history_windows) for index in range(len(states))]
    models: dict[tuple[str, int], Ridge | None] = {}
    for horizon in config.forecast_horizons_cycles:
        due = _due_indices(cycles, horizon)
        for target in TARGETS:
            response = states[target].to_numpy(float)
            x_rows: list[np.ndarray] = []
            y_rows: list[float] = []
            for position, future in enumerate(due):
                if future >= len(states) or vectors[position] is None:
                    continue
                if not np.isfinite(response[position]) or not np.isfinite(response[future]):
                    continue
                x_rows.append(vectors[position])
                y_rows.append(float(response[future] - response[position]))
            models[(target, horizon)] = Ridge(alpha=1.0, fit_intercept=False).fit(np.vstack(x_rows), y_rows) if len(x_rows) >= 20 else None
    return models


def _rolling_mae(history: list[tuple[float, float]], current_cycle: float, span: int) -> tuple[float, int]:
    values = [error for due_cycle, error in history if due_cycle >= current_cycle - span]
    return (float(np.mean(values)), len(values)) if values else (float("inf"), 0)


def safe_gate_select(
    static_scores: dict[str, tuple[float, int]],
    online_score: tuple[float, int],
    minimum_observations: int,
) -> tuple[str, str, bool]:
    """Select Online RLS only after comparing it with *every* static baseline."""
    best_static = min(STATIC_MODELS, key=lambda name: static_scores[name][0])
    best_mae, best_count = static_scores[best_static]
    online_mae, online_count = online_score
    use_online = bool(online_count >= minimum_observations and best_count >= minimum_observations and online_mae <= best_mae)
    return ("Online_RLS" if use_online else best_static), best_static, use_online


def run_online_forecasts(
    states: pd.DataFrame,
    frozen_models: dict[tuple[str, int], Ridge | None],
    protocol_id: str,
    config: ContinuousStateV41Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Predict first, then update RLS only after the scheduled outcome is observed."""
    assert_label_free(states)
    frame = states.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    cycles = frame.center_cycle.to_numpy(float)
    stride = max(float(frame.nominal_stride_cycles.iloc[0]), config.eps)
    issue_step = max(1, int(round(config.forecast_issue_stride_cycles / stride)))
    inputs = frame.loc[:, list(INPUTS)].to_numpy(float)
    vectors = [_feature_vector(inputs, index, config.forecast_history_windows) for index in range(len(frame))]
    prediction_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []
    for horizon in config.forecast_horizons_cycles:
        due = _due_indices(cycles, horizon)
        for target in TARGETS:
            response = frame[target].to_numpy(float)
            model = frozen_models.get((target, horizon))
            frozen_theta = model.coef_.astype(float) if model is not None else np.zeros(len(INPUTS) * 3 + 1)
            rls = ScalarRLS(frozen_theta.copy(), np.eye(len(frozen_theta)) * config.rls_initial_covariance, frozen_theta.copy(), config.rls_forgetting_factor,
                            config.rls_theta_norm_max, config.rls_covariance_max, config.forecast_delta_clip)
            kalman = LocalTrendKalman()
            histories = {name: [] for name in (*STATIC_MODELS, "Online_RLS")}
            pending: dict[int, list[dict[str, object]]] = {}
            for index in range(len(frame)):
                # Outcomes become available before the next prediction at this cycle.
                for issued in pending.pop(index, []):
                    observed = float(response[index])
                    if not np.isfinite(observed):
                        continue
                    for name in histories:
                        histories[name].append((float(cycles[index]), abs(observed - float(issued[f"{name}_prediction"]))))
                    vector = issued["_vector"]
                    origin_value = float(issued["_origin_value"])
                    if vector is not None and np.isfinite(origin_value):
                        rls.update(vector, observed - origin_value, clip=8.0, gain_max=config.rls_gain_norm_max)
                        update_rows.append({"protocol_id": protocol_id, "output_name": target, "horizon_cycles": horizon,
                                            "prediction_origin_cycle": float(issued["prediction_origin_cycle"]),
                                            "target_due_cycle": float(issued["target_due_cycle"]),
                                            "due_observation_cycle": float(cycles[index]), "rls_update_cycle": float(cycles[index])})
                # Kalman is updated with the same current observation already
                # available to Zero Delta, Local Linear, Ridge, and RLS inputs.
                kalman.update(float(response[index]), float(cycles[index]))
                if index % issue_step == 0 and not bool(frame.is_restart_guard.iloc[index]) and vectors[index] is not None and np.isfinite(response[index]):
                    vector = vectors[index]
                    static_mae = {name: _rolling_mae(histories[name], float(cycles[index]), config.forecast_rolling_window_cycles) for name in STATIC_MODELS}
                    best_static = min(STATIC_MODELS, key=lambda name: static_mae[name][0])
                    best_static_mae, best_static_count = static_mae[best_static]
                    online_mae, online_count = _rolling_mae(histories["Online_RLS"], float(cycles[index]), config.forecast_rolling_window_cycles)
                    selected, best_static, use_online = safe_gate_select(static_mae, (online_mae, online_count), config.safe_gate_min_observations)
                    zero = float(response[index])
                    local = _local_linear(cycles, response, index, config.forecast_history_windows, horizon)
                    kalman_value = kalman.forecast(horizon, zero)
                    frozen_delta = float(np.clip(model.predict(vector.reshape(1, -1))[0], -config.forecast_delta_clip, config.forecast_delta_clip)) if model is not None else 0.0
                    online_delta = rls.predict(vector)
                    candidates = {
                        "Zero_Delta": zero,
                        "Local_Linear": local,
                        "Kalman": kalman_value,
                        "Frozen_Ridge": zero + frozen_delta,
                        "Online_RLS": zero + online_delta,
                    }
                    future = int(due[index])
                    if future < len(frame):
                        row: dict[str, object] = {
                            "protocol_id": protocol_id, "target_dataset": str(frame.dataset.iloc[0]), "output_name": target,
                            "horizon_cycles": horizon, "prediction_origin_cycle": float(cycles[index]),
                            "target_due_cycle": float(cycles[index] + horizon), "due_observation_cycle": float(cycles[future]),
                            "observed_value": float(response[future]), "safe_gate_selected_model": selected,
                            "safe_gate_compared_against": best_static, "safe_gate_best_static_rolling_mae": best_static_mae,
                            "safe_gate_best_static_observations": best_static_count, "online_rolling_mae": online_mae,
                            "online_rolling_observations": online_count, "online_model_updated_after_observation": 0,
                        }
                        for name, value in candidates.items():
                            row[f"{name}_prediction"] = value
                        row["Safe_Gate_prediction"] = candidates[selected]
                        row["_vector"] = vector
                        row["_origin_value"] = zero
                        pending.setdefault(future, []).append(row)
                        prediction_rows.append(row)
    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        return predictions, pd.DataFrame(), pd.DataFrame(update_rows)
    internal = ("_vector", "_origin_value")
    metrics_rows: list[dict[str, object]] = []
    for (name, horizon), group in predictions.groupby(["output_name", "horizon_cycles"]):
        observed = group.observed_value.to_numpy(float)
        for model_name in ALL_MODELS:
            predicted = group[f"{model_name}_prediction"].to_numpy(float)
            finite = np.isfinite(observed) & np.isfinite(predicted)
            error = predicted[finite] - observed[finite]
            metrics_rows.append({"protocol_id": protocol_id, "target_dataset": str(group.target_dataset.iloc[0]), "output_name": name,
                                 "horizon_cycles": int(horizon), "model": model_name, "n_predictions": int(finite.sum()),
                                 "MAE": float(np.mean(np.abs(error))) if len(error) else np.nan,
                                 "RMSE": float(np.sqrt(np.mean(error * error))) if len(error) else np.nan,
                                 "evaluation_unit": "cycles", "forecast_issue_stride_cycles": config.forecast_issue_stride_cycles,
                                 "rolling_window_cycles": config.forecast_rolling_window_cycles})
    predictions = predictions.drop(columns=[column for column in internal if column in predictions], errors="ignore")
    assert_label_free(predictions)
    return predictions, pd.DataFrame(metrics_rows), pd.DataFrame(update_rows)
