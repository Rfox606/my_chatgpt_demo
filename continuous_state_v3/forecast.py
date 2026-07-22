from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import ContinuousStateV3Config
from .data import assert_label_free


TARGETS = ("D_state", "V50_norm", "instability_score", "S_severe_candidate")
INPUTS = ("D_state", "V20_norm", "V50_norm", "V100_norm", "direction_consistency", "A_state", "state_volatility_20", "state_volatility_50", "instability_score", "S_severe_candidate", "weighted_oos")


def _finite(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0., posinf=0., neginf=0.)


def _vector(values: np.ndarray, position: int, history: int) -> np.ndarray | None:
    if position + 1 < history:
        return None
    piece = _finite(values[position - history + 1:position + 1])
    return np.r_[1., piece.mean(axis=0), piece.std(axis=0), piece[-1] - piece[0]]


def _due_indices(cycles: np.ndarray, horizon: int) -> np.ndarray:
    return np.searchsorted(cycles, cycles + horizon, side="left")


def train_frozen_models(source_states: pd.DataFrame, config: ContinuousStateV3Config) -> dict[tuple[str, int], Ridge | None]:
    assert_label_free(source_states)
    states = source_states.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    values = _finite(states.loc[:, list(INPUTS)].to_numpy(float)); cycles = states.center_cycle.to_numpy(float)
    models: dict[tuple[str, int], Ridge | None] = {}
    for horizon in config.forecast_horizons_cycles:
        due = _due_indices(cycles, horizon)
        vectors = [_vector(values, index, config.forecast_history_windows) for index in range(len(states))]
        for target in TARGETS:
            outcome = states[target].to_numpy(float)
            x_rows, y_rows = [], []
            for index, later in enumerate(due):
                if later >= len(states) or vectors[index] is None or not np.isfinite(outcome[index]) or not np.isfinite(outcome[later]):
                    continue
                x_rows.append(vectors[index]); y_rows.append(outcome[later] - outcome[index])
            models[(target, horizon)] = Ridge(alpha=1.0, fit_intercept=False).fit(np.vstack(x_rows), np.asarray(y_rows)) if len(x_rows) >= 20 else None
    return models


@dataclass
class ScalarRLS:
    theta: np.ndarray
    covariance: np.ndarray
    frozen_theta: np.ndarray
    forgetting: float
    freeze_until: float = -np.inf

    def predict(self, vector: np.ndarray) -> float:
        return float(vector @ self.theta)

    def reset(self) -> None:
        self.theta = self.frozen_theta.copy(); self.covariance = np.eye(len(self.theta)) * 100.

    def update(self, vector: np.ndarray, observed: float, clip: float, gain_max: float) -> float:
        x = vector.reshape(-1, 1)
        denominator = self.forgetting + (x.T @ self.covariance @ x).item()
        gain = (self.covariance @ x / denominator).reshape(-1)
        norm = float(np.linalg.norm(gain))
        if norm > gain_max:
            gain *= gain_max / norm
        error = float(np.clip(observed - vector @ self.theta, -clip, clip))
        self.theta += gain * error
        self.covariance = (self.covariance - np.outer(gain, vector) @ self.covariance) / self.forgetting
        return error


def run_online_forecasts(target_states: pd.DataFrame, frozen_models: dict[tuple[str, int], Ridge | None], protocol_id: str, config: ContinuousStateV3Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separate output×horizon RLS heads with delayed updates and a guarded ensemble."""
    assert_label_free(target_states)
    states = target_states.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    values = _finite(states.loc[:, list(INPUTS)].to_numpy(float)); cycles = states.center_cycle.to_numpy(float)
    due = {horizon: _due_indices(cycles, horizon) for horizon in config.forecast_horizons_cycles}
    output_values = {target: states[target].to_numpy(float) for target in TARGETS}
    online: dict[tuple[str, int], ScalarRLS | None] = {}
    alpha = {(target, horizon): 0. for target in TARGETS for horizon in config.forecast_horizons_cycles}
    recent: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {(target, horizon): [] for target in TARGETS for horizon in config.forecast_horizons_cycles}
    for key, model in frozen_models.items():
        if model is None:
            online[key] = None
        else:
            theta = np.asarray(model.coef_).reshape(-1).copy()
            online[key] = ScalarRLS(theta.copy(), np.eye(len(theta)) * config.rls_initial_covariance, theta, config.rls_forgetting_factor)
    pending: dict[tuple[str, int, int], list[int]] = {}
    rows: list[dict[str, object]] = []
    volatility_history: list[float] = []
    for position in range(len(states)):
        volatility_history.append(float(states.state_volatility_20.iloc[position]))
        volatility_gate = max(3. * float(np.median(volatility_history[-200:])), config.eps)
        # Observation first reaches the system at t; update only predictions that became due now.
        for target in TARGETS:
            actual_values = output_values[target]
            for horizon in config.forecast_horizons_cycles:
                key = (target, horizon); model = online[key]
                for row_index in pending.pop((target, horizon, position), []):
                    row = rows[row_index]; origin = int(row["_origin"])
                    observed = actual_values[position] - actual_values[origin]
                    row["observation_available"] = int(np.isfinite(observed)); row["observed_delta"] = observed
                    if not np.isfinite(observed):
                        continue
                    frozen_error = abs(float(row["frozen_prediction"]) - observed)
                    online_error = abs(float(row["online_prediction"]) - observed)
                    safe_error = abs(float(row["safe_prediction"]) - observed)
                    history = recent[key]; history.append((frozen_error, online_error, safe_error, observed))
                    if len(history) > config.ensemble_window:
                        history.pop(0)
                    can_update = bool(model is not None and states.is_restart_guard.iloc[position] == 0 and states.is_restart_guard.iloc[origin] == 0 and states.state_volatility_20.iloc[position] <= volatility_gate and not (target == "S_severe_candidate" and not states.severe_direction_available.iloc[position]))
                    if can_update and cycles[position] >= model.freeze_until:
                        residuals = [item[1] for item in history]
                        median = float(np.median(residuals)) if residuals else 0.
                        scale = max(1.4826 * float(np.median(np.abs(np.asarray(residuals) - median))) if residuals else 0., config.eps)
                        model.update(np.asarray(row["_vector"]), float(observed), 3. * scale, config.rls_gain_norm_max)
                        row["online_model_updated_after_observation"] = 1
                    if len(history) >= config.ensemble_window:
                        frozen_mae = float(np.mean([item[0] for item in history])); online_mae = float(np.mean([item[1] for item in history]))
                        frozen_rmse = float(np.sqrt(np.mean([item[0] ** 2 for item in history]))); online_rmse = float(np.sqrt(np.mean([item[1] ** 2 for item in history])))
                        if online_mae <= .95 * frozen_mae and online_rmse <= 1.05 * frozen_rmse:
                            alpha[key] = min(1., alpha[key] + config.ensemble_alpha_step)
                        elif online_mae > 1.10 * frozen_mae or online_rmse > 1.20 * frozen_rmse:
                            alpha[key] = 0.
                            if model is not None:
                                model.reset(); model.freeze_until = float(cycles[position] + config.ensemble_freeze_cycles)
                            row["online_reset"] = 1; row["online_freeze_until"] = float(cycles[position] + config.ensemble_freeze_cycles)
        vector = _vector(values, position, config.forecast_history_windows)
        if vector is None:
            continue
        for target in TARGETS:
            current = float(states[target].iloc[position])
            for horizon in config.forecast_horizons_cycles:
                key = (target, horizon); frozen = frozen_models[key]; model = online[key]
                if target == "S_severe_candidate" and not states.severe_direction_available.iloc[position]:
                    frozen_prediction = online_prediction = safe_prediction = np.nan
                elif frozen is None or not np.isfinite(current):
                    frozen_prediction = online_prediction = safe_prediction = np.nan
                else:
                    frozen_prediction = float(frozen.predict(vector.reshape(1, -1))[0])
                    online_prediction = float(model.predict(vector)) if model is not None else frozen_prediction
                    safe_prediction = (1. - alpha[key]) * frozen_prediction + alpha[key] * online_prediction
                row = {"protocol_id": protocol_id, "target_dataset": states.dataset.iloc[0], "output_name": target, "prediction_origin_window": int(states.window_index.iloc[position]), "prediction_origin_cycle": float(cycles[position]), "horizon_cycles": horizon, "target_due_cycle": float(cycles[position] + horizon), "prediction_available": int(np.isfinite(safe_prediction)), "frozen_prediction": frozen_prediction, "online_prediction": online_prediction, "safe_prediction": safe_prediction, "ensemble_alpha": alpha[key], "observation_available": 0, "observed_delta": np.nan, "online_model_updated_after_observation": 0, "online_reset": 0, "online_freeze_until": model.freeze_until if model is not None else np.nan, "_origin": position, "_vector": vector}
                rows.append(row)
                if due[horizon][position] < len(states) and np.isfinite(safe_prediction):
                    pending.setdefault((target, horizon, int(due[horizon][position])), []).append(len(rows) - 1)
    predictions = pd.DataFrame(rows)
    public = predictions.drop(columns=["_origin", "_vector"], errors="ignore")
    metrics: list[dict[str, object]] = []
    for (output, horizon), group in public.loc[public.observation_available.eq(1)].groupby(["output_name", "horizon_cycles"]):
        for name, column in (("Frozen_Ridge", "frozen_prediction"), ("Robust_Online_RLS", "online_prediction"), ("Safe_Ensemble", "safe_prediction")):
            error = group[column].to_numpy(float) - group.observed_delta.to_numpy(float)
            metrics.append({"protocol_id": protocol_id, "output_name": output, "horizon_cycles": horizon, "model": name, "prediction_count": len(group), "MAE": float(np.mean(np.abs(error))), "RMSE": float(np.sqrt(np.mean(error ** 2))), "direction_accuracy": float(np.mean(np.sign(group[column].to_numpy(float)) == np.sign(group.observed_delta.to_numpy(float))))})
    alpha_table = public.groupby(["protocol_id", "output_name", "horizon_cycles"], as_index=False).agg(final_ensemble_alpha=("ensemble_alpha", "last"), online_reset_count=("online_reset", "sum"), online_freeze_until=("online_freeze_until", "max")) if not public.empty else pd.DataFrame()
    return public, pd.DataFrame(metrics), alpha_table
