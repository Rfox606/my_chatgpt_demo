from __future__ import annotations

"""Causal baselines, frozen models, delayed RLS, and the safe ensemble."""

import gc
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import ContinuousStateV31Config
from .data import assert_label_free


TARGETS = ("D_state", "V50_norm", "instability_score", "S_severe_candidate")
INPUTS = (
    "D_state", "V20_norm", "V50_norm", "V100_norm", "direction_consistency",
    "A_state", "state_volatility_20", "state_volatility_50", "weighted_oos",
)
MODEL_COLUMNS = {
    "Zero_Delta": "zero_prediction",
    "Local_Linear": "local_linear_prediction",
    "Kalman_Trend": "kalman_prediction",
    "Frozen_Ridge": "frozen_prediction",
    "Robust_Online_RLS": "online_prediction",
    "Safe_Ensemble": "safe_prediction",
}


def _due_indices(cycles: np.ndarray, horizon: int) -> np.ndarray:
    return np.searchsorted(cycles, cycles + horizon, side="left")


def _feature_vector(values: np.ndarray, position: int, history: int) -> np.ndarray | None:
    """Build a finite causal summary; missing state values are never zero-filled."""
    if position + 1 < history:
        return None
    sample = values[position - history + 1:position + 1]
    if not np.isfinite(sample).all():
        return None
    return np.r_[1.0, sample.mean(axis=0), sample.std(axis=0), sample[-1] - sample[0]]


def _available(states: pd.DataFrame, target: str, position: int) -> bool:
    if target == "instability_score":
        return bool(states.plateau_locked.iloc[position]) and np.isfinite(states[target].iloc[position])
    if target == "S_severe_candidate":
        return bool(states.severe_direction_available.iloc[position]) and np.isfinite(states[target].iloc[position])
    return bool(np.isfinite(states[target].iloc[position]))


def _recent_valid_indices(states: pd.DataFrame, position: int, valid_cycles: float) -> np.ndarray:
    stride = float(states.nominal_stride_cycles.iloc[position])
    selected: list[int] = []
    total = 0.0
    for index in range(position, -1, -1):
        if bool(states.is_restart_guard.iloc[index]):
            continue
        selected.append(index)
        total += stride
        if total >= valid_cycles:
            break
    return np.asarray(selected[::-1], dtype=int)


def _median_slope(cycles: np.ndarray, values: np.ndarray, eps: float) -> float:
    if len(values) < 3:
        return 0.0
    # This is the pre-registered ``median slope`` option.  Unlike all-pairs
    # Theil-Sen, its O(n) causal update is practical for every online window.
    slope = np.diff(values) / np.maximum(np.diff(cycles), eps)
    slope = slope[np.isfinite(slope)]
    return float(np.median(slope)) if len(slope) else 0.0


@dataclass
class LocalTrendKalman:
    """A small causal local-level/local-trend filter for one scalar output."""

    level: float | None = None
    trend_per_cycle: float = 0.0
    covariance: np.ndarray | None = None
    last_cycle: float | None = None

    def update(self, observation: float, cycle: float) -> None:
        if not np.isfinite(observation):
            return
        if self.level is None:
            self.level, self.last_cycle = float(observation), float(cycle)
            self.covariance = np.diag([1.0, 1.0])
            return
        assert self.covariance is not None and self.last_cycle is not None
        dt = max(float(cycle) - self.last_cycle, 1.0)
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        process = np.diag([1e-4 * dt * dt, 1e-7 * dt])
        state = transition @ np.array([self.level, self.trend_per_cycle])
        predicted_covariance = transition @ self.covariance @ transition.T + process
        design = np.array([1.0, 0.0])
        gain = predicted_covariance @ design / (float(design @ predicted_covariance @ design) + 1e-3)
        state += gain * (float(observation) - float(design @ state))
        self.covariance = (np.eye(2) - np.outer(gain, design)) @ predicted_covariance
        self.level, self.trend_per_cycle = float(state[0]), float(state[1])
        self.last_cycle = float(cycle)

    def delta(self, horizon: int) -> float:
        return float(self.trend_per_cycle * horizon) if self.level is not None else 0.0


@dataclass
class ScalarRLS:
    theta: np.ndarray
    covariance: np.ndarray
    frozen_theta: np.ndarray
    forgetting: float

    def predict(self, vector: np.ndarray) -> float:
        return float(vector @ self.theta)

    def reset_to_frozen(self, covariance: float) -> None:
        self.theta = self.frozen_theta.copy()
        self.covariance = np.eye(len(self.theta)) * covariance

    def update(self, vector: np.ndarray, observed: float, clip: float, gain_max: float) -> None:
        x = vector.reshape(-1, 1)
        denominator = self.forgetting + float((x.T @ self.covariance @ x).item())
        gain = (self.covariance @ x / max(denominator, 1e-12)).reshape(-1)
        norm = float(np.linalg.norm(gain))
        if norm > gain_max:
            gain *= gain_max / norm
        residual = float(np.clip(observed - vector @ self.theta, -clip, clip))
        self.theta += gain * residual
        self.covariance = (self.covariance - np.outer(gain, vector) @ self.covariance) / self.forgetting


@dataclass
class EnsembleHead:
    state: str = "PROBATION"
    alpha: float = 0.0
    freeze_until_cycle: float = -np.inf
    reset_episode_count: int = 0
    reset_episode_id: int = 0
    freeze_start_cycle: float = np.nan
    comparison: list[tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if self.comparison is None:
            self.comparison = []


def train_frozen_models(source_states: pd.DataFrame, config: ContinuousStateV31Config) -> dict[tuple[str, int], Ridge | None]:
    """Train F0 only from finite, available, source-only observations."""
    assert_label_free(source_states)
    states = source_states.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    source_stride = max(float(states.nominal_stride_cycles.iloc[0]), config.eps)
    source_step = max(1, int(round(config.forecast_issue_stride_cycles / source_stride)))
    states = states.iloc[::source_step].reset_index(drop=True)
    values = states.loc[:, list(INPUTS)].to_numpy(float)
    cycles = states.center_cycle.to_numpy(float)
    vectors = [_feature_vector(values, position, config.forecast_history_windows) for position in range(len(states))]
    models: dict[tuple[str, int], Ridge | None] = {}
    for horizon in config.forecast_horizons_cycles:
        due = _due_indices(cycles, horizon)
        for target in TARGETS:
            features, outcomes = [], []
            response = states[target].to_numpy(float)
            for position, later in enumerate(due):
                if later >= len(states) or vectors[position] is None:
                    continue
                if not _available(states, target, position) or not _available(states, target, int(later)):
                    continue
                features.append(vectors[position])
                outcomes.append(float(response[later] - response[position]))
            models[(target, horizon)] = (
                Ridge(alpha=1.0, fit_intercept=False).fit(np.vstack(features), np.asarray(outcomes))
                if len(features) >= 20 else None
            )
    return models


def _errors(history: list[tuple[float, float]], eps: float) -> tuple[float, float, float, float]:
    frozen = np.asarray([item[0] for item in history], dtype=float)
    online = np.asarray([item[1] for item in history], dtype=float)
    f_mae, o_mae = float(np.mean(frozen)), float(np.mean(online))
    f_rmse, o_rmse = float(np.sqrt(np.mean(frozen ** 2))), float(np.sqrt(np.mean(online ** 2)))
    return max(f_mae, eps), max(o_mae, eps), max(f_rmse, eps), max(o_rmse, eps)


def _metrics_from_predictions(predictions: pd.DataFrame, protocol_id: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    observed = predictions.loc[predictions.observation_available.eq(1)].copy()
    for (output, horizon), group in observed.groupby(["output_name", "horizon_cycles"]):
        actual = group.observed_delta.to_numpy(float)
        for model, column in MODEL_COLUMNS.items():
            values = group[column].to_numpy(float)
            usable = np.isfinite(values) & np.isfinite(actual)
            if not usable.any():
                continue
            error = values[usable] - actual[usable]
            rows.append({
                "protocol_id": protocol_id,
                "output_name": output,
                "horizon_cycles": int(horizon),
                "model": model,
                "prediction_count": int(usable.sum()),
                "MAE": float(np.mean(np.abs(error))),
                "RMSE": float(np.sqrt(np.mean(error ** 2))),
                "direction_accuracy": float(np.mean(np.sign(values[usable]) == np.sign(actual[usable]))),
            })
    return pd.DataFrame(rows)


def _segment_metrics(predictions: pd.DataFrame, protocol_id: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    observed = predictions.loc[predictions.observation_available.eq(1)].copy()
    for (output, horizon), group in observed.groupby(["output_name", "horizon_cycles"]):
        group = group.sort_values("due_observation_cycle").reset_index(drop=True)
        for segment, indices in zip(("early", "middle", "late"), np.array_split(np.arange(len(group)), 3), strict=True):
            if not len(indices):
                continue
            part = group.iloc[indices]
            actual = part.observed_delta.to_numpy(float)
            for model, column in MODEL_COLUMNS.items():
                values = part[column].to_numpy(float)
                usable = np.isfinite(values) & np.isfinite(actual)
                if not usable.any():
                    continue
                error = values[usable] - actual[usable]
                rows.append({"protocol_id": protocol_id, "output_name": output, "horizon_cycles": int(horizon),
                             "segment": segment, "model": model, "prediction_count": int(usable.sum()),
                             "MAE": float(np.mean(np.abs(error))), "RMSE": float(np.sqrt(np.mean(error ** 2))),
                             "direction_accuracy": float(np.mean(np.sign(values[usable]) == np.sign(actual[usable])))})
    return pd.DataFrame(rows)


def _rolling_metrics(predictions: pd.DataFrame, protocol_id: str, window: int, export_stride: int = 1) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    observed = predictions.loc[predictions.observation_available.eq(1)].copy()
    for (output, horizon), group in observed.groupby(["output_name", "horizon_cycles"]):
        group = group.sort_values("due_observation_cycle").copy()
        actual = group.observed_delta.to_numpy(float)
        for model, column in MODEL_COLUMNS.items():
            values = group[column].to_numpy(float)
            usable = np.isfinite(values) & np.isfinite(actual)
            if not usable.any():
                continue
            part = group.loc[usable, ["due_observation_cycle"]].copy()
            error = values[usable] - actual[usable]
            part["protocol_id"] = protocol_id; part["output_name"] = output; part["horizon_cycles"] = int(horizon); part["model"] = model
            part["rolling_count"] = np.minimum(np.arange(1, len(part) + 1), window)
            part["rolling_MAE"] = pd.Series(np.abs(error)).rolling(window, min_periods=1).mean().to_numpy()
            part["rolling_RMSE"] = np.sqrt(pd.Series(error ** 2).rolling(window, min_periods=1).mean().to_numpy())
            part["rolling_direction_accuracy"] = pd.Series((np.sign(values[usable]) == np.sign(actual[usable])).astype(float)).rolling(window, min_periods=1).mean().to_numpy()
            if export_stride > 1 and len(part):
                keep = (np.arange(len(part)) % export_stride == export_stride - 1)
                keep[-1] = True
                part = part.iloc[np.flatnonzero(keep)]
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _regret(predictions: pd.DataFrame, protocol_id: str, export_stride: int = 1) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    observed = predictions.loc[predictions.observation_available.eq(1)].copy()
    baseline_models = ("Zero_Delta", "Local_Linear", "Kalman_Trend", "Frozen_Ridge")
    safe_column = MODEL_COLUMNS["Safe_Ensemble"]
    for (output, horizon), group in observed.groupby(["output_name", "horizon_cycles"]):
        group = group.sort_values("due_observation_cycle")
        actual = group.observed_delta.to_numpy(float); safe = group[safe_column].to_numpy(float)
        for baseline in baseline_models:
            values = group[MODEL_COLUMNS[baseline]].to_numpy(float)
            usable = np.isfinite(actual) & np.isfinite(safe) & np.isfinite(values)
            if not usable.any():
                continue
            part = group.loc[usable, ["due_observation_cycle"]].copy()
            regret = np.abs(safe[usable] - actual[usable]) - np.abs(values[usable] - actual[usable])
            part["protocol_id"] = protocol_id; part["output_name"] = output; part["horizon_cycles"] = int(horizon)
            part["baseline_model"] = baseline; part["instantaneous_regret"] = regret; part["cumulative_regret"] = np.cumsum(regret)
            if export_stride > 1 and len(part):
                keep = (np.arange(len(part)) % export_stride == export_stride - 1)
                keep[-1] = True
                part = part.iloc[np.flatnonzero(keep)]
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_online_forecasts(
    target_states: pd.DataFrame,
    frozen_models: dict[tuple[str, int], Ridge | None],
    protocol_id: str,
    config: ContinuousStateV31Config,
    include_evaluation: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Issue forecasts first and adapt only when their observations actually arrive."""
    assert_label_free(target_states)
    states = target_states.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    native_stride = max(float(states.nominal_stride_cycles.iloc[0]), config.eps)
    issue_step = max(1, int(round(config.forecast_issue_stride_cycles / native_stride)))
    # Forecast decisions, delayed observations, and RLS updates operate at a
    # fixed 50-cycle causal decision cadence.  State monitoring itself remains
    # at every source window in ``state_window_scores_v31.csv``.
    states = states.iloc[::issue_step].reset_index(drop=True)
    cycles = states.center_cycle.to_numpy(float)
    values = states.loc[:, list(INPUTS)].to_numpy(float)
    output_values = {target: states[target].to_numpy(float) for target in TARGETS}
    due = {horizon: _due_indices(cycles, horizon) for horizon in config.forecast_horizons_cycles}
    vectors = [_feature_vector(values, position, config.forecast_history_windows) for position in range(len(states))]
    online: dict[tuple[str, int], ScalarRLS | None] = {}
    heads: dict[tuple[str, int], EnsembleHead] = {}
    kalman = {target: LocalTrendKalman() for target in TARGETS}
    pending: dict[tuple[str, int, int], list[int]] = {}
    rows: list[dict[str, object]] = []
    state_log: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []
    active_episodes: dict[tuple[str, int], int] = {}
    for key, model in frozen_models.items():
        if model is None:
            online[key] = None
        else:
            theta = np.asarray(model.coef_, dtype=float).reshape(-1)
            online[key] = ScalarRLS(theta.copy(), np.eye(len(theta)) * config.rls_initial_covariance, theta.copy(), config.rls_forgetting_factor)
        heads[key] = EnsembleHead()

    def log_transition(key: tuple[str, int], cycle: float, transition: str) -> None:
        target, horizon = key; head = heads[key]
        state_log.append({
            "protocol_id": protocol_id, "target_dataset": str(states.dataset.iloc[0]), "output_name": target,
            "horizon_cycles": horizon, "cycle": float(cycle), "ensemble_state": head.state,
            "ensemble_alpha": head.alpha, "reset_transition": transition, "reset_episode_id": head.reset_episode_id,
            "reset_episode_count": head.reset_episode_count, "freeze_start_cycle": head.freeze_start_cycle,
            "freeze_until_cycle": head.freeze_until_cycle,
        })

    for position in range(len(states)):
        if os.environ.get("CSV31_FORECAST_PROGRESS") and position and position % 500 == 0:
            print(f"csv31 forecast {protocol_id}: {position}/{len(states)}", flush=True)
        current_cycle = float(cycles[position])
        # A frozen head thaws once.  It cannot reset or extend its deadline while frozen.
        for key, head in heads.items():
            if head.state == "FROZEN" and current_cycle >= head.freeze_until_cycle:
                episode_index = active_episodes.pop(key, None)
                if episode_index is not None:
                    episodes[episode_index]["freeze_end_cycle"] = current_cycle
                    episodes[episode_index]["freeze_duration_cycles"] = current_cycle - float(episodes[episode_index]["freeze_start_cycle"])
                head.state = "PROBATION"; head.alpha = 0.0; head.comparison.clear()
                log_transition(key, current_cycle, "FROZEN_TO_PROBATION")

        # Update only forecasts whose requested horizon has now matured.
        for target in TARGETS:
            for horizon in config.forecast_horizons_cycles:
                key = (target, horizon); head = heads[key]; model = online[key]
                for row_index in pending.pop((target, horizon, position), []):
                    row = rows[row_index]; origin = int(row["_origin"])
                    observed = output_values[target][position] - output_values[target][origin]
                    row["due_observation_cycle"] = current_cycle
                    row["observation_available"] = int(np.isfinite(observed))
                    row["observed_delta"] = float(observed) if np.isfinite(observed) else np.nan
                    if not np.isfinite(observed):
                        continue
                    for model_name, column in MODEL_COLUMNS.items():
                        value = float(row[column])
                        row[f"{model_name}_absolute_error"] = abs(value - observed) if np.isfinite(value) else np.nan
                    frozen_error = float(row["Frozen_Ridge_absolute_error"])
                    online_error = float(row["Robust_Online_RLS_absolute_error"])
                    if np.isfinite(frozen_error) and np.isfinite(online_error):
                        head.comparison.append((frozen_error, online_error))
                        if len(head.comparison) > config.ensemble_window:
                            head.comparison.pop(0)
                    can_update = bool(
                        model is not None and head.state != "FROZEN" and not states.is_restart_guard.iloc[origin]
                        and not states.is_restart_guard.iloc[position] and _available(states, target, position)
                    )
                    if can_update:
                        residuals = np.asarray([item[1] for item in head.comparison], dtype=float)
                        centre = float(np.median(residuals)) if len(residuals) else 0.0
                        scale = max(1.4826 * float(np.median(np.abs(residuals - centre))) if len(residuals) else 0.0, config.eps)
                        model.update(np.asarray(row["_vector"], dtype=float), float(observed), 3.0 * scale, config.rls_gain_norm_max)
                        row["online_model_updated_after_observation"] = 1
                        row["rls_update_cycle"] = current_cycle
                    if len(head.comparison) >= config.ensemble_window and head.state != "FROZEN":
                        frozen_mae, online_mae, frozen_rmse, online_rmse = _errors(head.comparison, config.eps)
                        bad = online_mae > 1.10 * frozen_mae or online_rmse > 1.20 * frozen_rmse
                        good = online_mae <= .95 * frozen_mae and online_rmse <= 1.05 * frozen_rmse
                        if bad:
                            previous = head.state
                            head.state = "FROZEN"; head.alpha = 0.0; head.reset_episode_count += 1; head.reset_episode_id = head.reset_episode_count
                            head.freeze_start_cycle = current_cycle; head.freeze_until_cycle = current_cycle + config.ensemble_freeze_cycles
                            if model is not None:
                                model.reset_to_frozen(config.rls_initial_covariance)
                            episodes.append({"protocol_id": protocol_id, "target_dataset": str(states.dataset.iloc[0]), "output_name": target,
                                             "horizon_cycles": horizon, "reset_episode_id": head.reset_episode_id,
                                             "freeze_start_cycle": current_cycle, "freeze_until_cycle": head.freeze_until_cycle,
                                             "freeze_end_cycle": np.nan, "freeze_duration_cycles": np.nan,
                                             "transition_from": previous})
                            active_episodes[key] = len(episodes) - 1
                            row["reset_transition"] = f"{previous}_TO_FROZEN"
                            row["reset_episode_id"] = head.reset_episode_id
                            log_transition(key, current_cycle, f"{previous}_TO_FROZEN")
                        elif head.state == "PROBATION":
                            head.state = "ACTIVE"; head.alpha = 0.0
                            log_transition(key, current_cycle, "PROBATION_TO_ACTIVE")
                        elif head.state == "ACTIVE" and good:
                            head.alpha = min(1.0, head.alpha + config.ensemble_alpha_step)
                    row["ensemble_state_after_observation"] = head.state
                    row["ensemble_alpha_after_observation"] = head.alpha

        # The local-trend baseline receives only the arrived scalar observation.
        for target in TARGETS:
            if _available(states, target, position):
                kalman[target].update(float(output_values[target][position]), current_cycle)

        vector = vectors[position]
        local_slopes: dict[str, float] = {}
        for target in TARGETS:
            if _available(states, target, position) and vector is not None:
                valid = _recent_valid_indices(states, position, config.plateau_reference_valid_cycles)
                y = output_values[target][valid]
                finite = np.isfinite(y)
                local_slopes[target] = _median_slope(cycles[valid][finite], y[finite], config.eps) if finite.sum() >= 3 else 0.0
        for target in TARGETS:
            available = _available(states, target, position) and vector is not None
            for horizon in config.forecast_horizons_cycles:
                key = (target, horizon); model = online[key]; frozen = frozen_models[key]; head = heads[key]
                if available and frozen is not None:
                    zero = 0.0
                    local = local_slopes[target] * horizon
                    kalman_delta = kalman[target].delta(horizon)
                    frozen_delta = float(frozen.predict(vector.reshape(1, -1))[0])
                    online_delta = float(model.predict(vector)) if model is not None else frozen_delta
                    safe = (1.0 - head.alpha) * frozen_delta + head.alpha * online_delta
                else:
                    zero = local = kalman_delta = frozen_delta = online_delta = safe = np.nan
                requested_due = current_cycle + horizon
                row = {
                    "protocol_id": protocol_id, "target_dataset": str(states.dataset.iloc[0]), "output_name": target,
                    "prediction_origin_window": int(states.window_index.iloc[position]), "prediction_origin_cycle": current_cycle,
                    "horizon_cycles": int(horizon), "target_due_cycle": requested_due,
                    "forecast_issue_stride_cycles": int(config.forecast_issue_stride_cycles),
                    "prediction_available": int(np.isfinite(safe)), "zero_prediction": zero,
                    "local_linear_prediction": local, "kalman_prediction": kalman_delta, "frozen_prediction": frozen_delta,
                    "online_prediction": online_delta, "safe_prediction": safe, "ensemble_state": head.state,
                    "ensemble_alpha": head.alpha, "reset_episode_id": head.reset_episode_id,
                    "reset_episode_count": head.reset_episode_count, "freeze_start_cycle": head.freeze_start_cycle,
                    "freeze_until_cycle": head.freeze_until_cycle, "observation_available": 0, "due_observation_cycle": np.nan,
                    "observed_delta": np.nan, "online_model_updated_after_observation": 0, "rls_update_cycle": np.nan,
                    "reset_transition": "", "_origin": position, "_vector": vector,
                }
                rows.append(row)
                later = int(due[horizon][position])
                if later < len(states) and np.isfinite(safe):
                    pending.setdefault((target, horizon, later), []).append(len(rows) - 1)
                # The prediction table already retains state and alpha per origin.
                # The separate log records transitions only, avoiding a duplicate
                # long table in memory for the full stream.

    for key, episode_index in active_episodes.items():
        # A run may end while frozen.  The configured deadline is retained rather than extended.
        episodes[episode_index]["freeze_duration_cycles"] = max(0.0, float(cycles[-1]) - float(episodes[episode_index]["freeze_start_cycle"]))
    predictions = pd.DataFrame(rows)
    # Release the list of dictionaries and private feature-vector arrays before
    # materializing rolling/regret tables for the full experimental stream.
    del rows, pending, vectors, values, output_values
    gc.collect()
    public = predictions.drop(columns=["_origin", "_vector"], errors="ignore")
    del predictions
    gc.collect()
    metrics = _metrics_from_predictions(public, protocol_id) if include_evaluation else pd.DataFrame()
    segment = _segment_metrics(public, protocol_id) if include_evaluation else pd.DataFrame()
    rolling = _rolling_metrics(public, protocol_id, config.rolling_metric_observations, config.rolling_metric_export_stride) if include_evaluation else pd.DataFrame()
    regret = _regret(public, protocol_id, config.rolling_metric_export_stride) if include_evaluation else pd.DataFrame()
    weights = public.loc[:, ["protocol_id", "target_dataset", "output_name", "horizon_cycles", "prediction_origin_cycle", "ensemble_state", "ensemble_alpha", "reset_episode_id", "reset_episode_count", "freeze_start_cycle", "freeze_until_cycle"]]
    log_columns = ("protocol_id", "target_dataset", "output_name", "horizon_cycles", "cycle", "ensemble_state", "ensemble_alpha",
                   "reset_transition", "reset_episode_id", "reset_episode_count", "freeze_start_cycle", "freeze_until_cycle")
    state_log_frame = pd.DataFrame(state_log + [{**episode, "ensemble_state": "EPISODE", "ensemble_alpha": 0.0, "cycle": episode["freeze_start_cycle"], "reset_transition": "EPISODE_SUMMARY"} for episode in episodes])
    if state_log_frame.empty:
        state_log_frame = pd.DataFrame(columns=log_columns)
    episode_columns = ("protocol_id", "target_dataset", "output_name", "horizon_cycles", "reset_episode_id", "freeze_start_cycle",
                       "freeze_until_cycle", "freeze_end_cycle", "freeze_duration_cycles", "transition_from")
    episode_frame = pd.DataFrame(episodes)
    if episode_frame.empty:
        episode_frame = pd.DataFrame(columns=episode_columns)
    return public, metrics, segment, rolling, regret, state_log_frame, episode_frame, weights
