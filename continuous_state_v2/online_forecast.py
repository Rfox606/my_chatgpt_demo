from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import ContinuousStateV2Config
from .data import assert_label_free


STATE_INPUTS = ("P_common", "BD", "B_terminal", "P_RS20", "P_RS50", "BD_RS20", "BD_RS50", "B_RS20", "TES", "weighted_oos_common")


def _vector(frame: pd.DataFrame, position: int, history: int) -> np.ndarray | None:
    if position + 1 < history:
        return None
    values = frame.loc[position - history + 1:position, list(STATE_INPUTS)].to_numpy(float)
    if not np.isfinite(values).all():
        return None
    mean, std = values.mean(axis=0), values.std(axis=0)
    slope = values[-1] - values[0]
    return np.r_[1., mean, std, slope]


def _due_indices(cycles: np.ndarray, horizon: int) -> np.ndarray:
    return np.searchsorted(cycles, cycles + horizon, side="left")


@dataclass
class RLS:
    theta: np.ndarray
    covariance: np.ndarray
    forgetting: float

    def predict(self, vector: np.ndarray) -> np.ndarray:
        return vector @ self.theta

    def update(self, vector: np.ndarray, observed: np.ndarray) -> None:
        x = vector.reshape(-1, 1)
        gain = (self.covariance @ x) / (self.forgetting + (x.T @ self.covariance @ x).item())
        error = observed - vector @ self.theta
        self.theta += gain @ error.reshape(1, -1)
        self.covariance = (self.covariance - gain @ x.T @ self.covariance) / self.forgetting


def train_frozen_predictors(source_states: pd.DataFrame, config: ContinuousStateV2Config) -> dict[int, Ridge]:
    assert_label_free(source_states)
    cycles = source_states.center_cycle.to_numpy(float)
    inputs = source_states.loc[:, list(STATE_INPUTS)].to_numpy(float)
    states = source_states.loc[:, ["P_common", "BD", "B_terminal"]].to_numpy(float)
    result: dict[int, Ridge] = {}
    for horizon in config.forecast_horizons_cycles:
        due = _due_indices(cycles, horizon)
        x_rows, y_rows = [], []
        for index, later in enumerate(due):
            if index + 1 < config.forecast_history_windows or later >= len(source_states):
                continue
            history = inputs[index - config.forecast_history_windows + 1:index + 1]
            if not np.isfinite(history).all():
                continue
            vector = np.r_[1., history.mean(axis=0), history.std(axis=0), history[-1] - history[0]]
            x_rows.append(vector)
            y_rows.append(states[later] - states[index])
        if not x_rows:
            raise ValueError("No source forecast samples")
        result[horizon] = Ridge(alpha=1.0, fit_intercept=False).fit(np.vstack(x_rows), np.vstack(y_rows))
    return result


def run_online_forecasts(target_states: pd.DataFrame, frozen: dict[int, Ridge], direction_id: str, config: ContinuousStateV2Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_label_free(target_states)
    states = target_states.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    cycles = states.center_cycle.to_numpy(float)
    due_by_horizon = {h: _due_indices(cycles, h) for h in config.forecast_horizons_cycles}
    online = {h: RLS(model.coef_.T.copy(), np.eye(model.coef_.shape[1]) * config.rls_initial_covariance, config.rls_forgetting_factor) for h, model in frozen.items()}
    pending: dict[tuple[int, int], list[int]] = {}
    rows: list[dict[str, object]] = []
    for t in range(len(states)):
        # At the arrival of t, only now reveal its already pre-update state to forecasts due here.
        for horizon in config.forecast_horizons_cycles:
            for row_index in pending.pop((horizon, t), []):
                origin = int(rows[row_index]["_origin_position"])
                observed = states.loc[t, ["P_common", "BD", "B_terminal"]].to_numpy(float) - states.loc[origin, ["P_common", "BD", "B_terminal"]].to_numpy(float)
                rows[row_index].update({"observation_available": 1, "observed_delta_P": observed[0], "observed_delta_BD": observed[1], "observed_delta_B": observed[2]})
                online[horizon].update(np.asarray(rows[row_index]["_vector"]), observed)
                rows[row_index]["online_model_updated_after_observation"] = 1
        vector = _vector(states, t, config.forecast_history_windows)
        if vector is None:
            continue
        for horizon in config.forecast_horizons_cycles:
            due = int(due_by_horizon[horizon][t])
            f0 = frozen[horizon].predict(vector.reshape(1, -1))[0]
            f1 = online[horizon].predict(vector)
            row: dict[str, object] = {"direction_id": direction_id, "target_dataset": str(states.dataset.iloc[0]), "prediction_origin_window": int(states.window_index.iloc[t]), "prediction_origin_cycle": float(cycles[t]), "horizon_cycles": horizon, "target_due_cycle": float(cycles[t] + horizon), "prediction_available": 1, "pred_delta_P": float(f1[0]), "pred_delta_BD": float(f1[1]), "pred_delta_B": float(f1[2]), "frozen_pred_delta_P": float(f0[0]), "frozen_pred_delta_BD": float(f0[1]), "frozen_pred_delta_B": float(f0[2]), "observation_available": 0, "observed_delta_P": np.nan, "observed_delta_BD": np.nan, "observed_delta_B": np.nan, "online_model_updated_after_observation": 0, "_origin_position": t, "_due_position": due, "_vector": vector}
            rows.append(row)
            if due < len(states):
                pending.setdefault((horizon, due), []).append(len(rows) - 1)
    # Predictions whose due time is later than the target run remain unavailable by definition.
    public = pd.DataFrame(rows)
    metrics: list[dict[str, object]] = []
    if not public.empty:
        observed = public.loc[public.observation_available.eq(1)]
        for (direction, horizon), group in observed.groupby(["direction_id", "horizon_cycles"]):
            for model, prefix in (("Frozen", "frozen_pred"), ("Online_RLS", "pred")):
                row: dict[str, object] = {"direction_id": direction, "horizon_cycles": horizon, "model": model, "prediction_count": len(group)}
                for name, truth in (("P", "observed_delta_P"), ("BD", "observed_delta_BD"), ("B", "observed_delta_B")):
                    error = group[f"{prefix}_delta_{name}"].to_numpy(float) - group[truth].to_numpy(float)
                    row[f"MAE_{name}"] = float(np.mean(np.abs(error)))
                    row[f"RMSE_{name}"] = float(np.sqrt(np.mean(error ** 2)))
                    predicted = group[f"{prefix}_delta_{name}"].to_numpy(float)
                    actual = group[truth].to_numpy(float)
                    row[f"direction_accuracy_{name}"] = float(np.mean(np.sign(predicted) == np.sign(actual)))
                metrics.append(row)
    return public.drop(columns=["_origin_position", "_due_position", "_vector"], errors="ignore"), pd.DataFrame(metrics)
