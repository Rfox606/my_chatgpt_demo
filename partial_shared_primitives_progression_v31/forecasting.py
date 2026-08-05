from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import V31Config
from .data import robust_scale


@dataclass(frozen=True)
class FrozenSourceModel:
    weights: dict[int, np.ndarray]
    location: np.ndarray
    scale: np.ndarray
    source_dataset: str
    train_windows: int


def _input(values: np.ndarray, index: int, location: np.ndarray, scale: np.ndarray) -> np.ndarray:
    prior = (values[index - 1] - location) / scale; lag = max(0, index - 4); delta = (values[index - 1] - values[lag]) / scale
    return np.r_[1.0, prior, delta]


def fit_source_frozen(source: pd.DataFrame, config: V31Config) -> FrozenSourceModel:
    values = source.loc[:, list(config.features)].to_numpy(float); train = min(config.source_train_windows, len(values)); location, scale = robust_scale(values[:train])
    weights: dict[int, np.ndarray] = {}
    for horizon in config.horizons:
        rows = []; targets = []
        for index in range(config.history_windows, train - horizon):
            rows.append(_input(values, index, location, scale)); targets.append((values[index + horizon] - values[index]) / scale)
        x = np.asarray(rows); y = np.asarray(targets); regular = config.ridge_alpha * np.eye(x.shape[1]); weights[horizon] = np.linalg.solve(x.T @ x + regular, x.T @ y)
    return FrozenSourceModel(weights, location, scale, str(source.dataset.iloc[0]), train)


@dataclass
class OnlineRegressor:
    weights: dict[int, np.ndarray]
    location: np.ndarray
    scale: np.ndarray
    support: dict[int, int]

    def predict(self, x: np.ndarray, horizon: int) -> np.ndarray: return x @ self.weights[horizon]

    def update(self, x: np.ndarray, target: np.ndarray, horizon: int, rate: float) -> None:
        # Normalized LMS is the stable online realization of the frozen
        # pre-registered learning rate.  It does not add a tuned clipping
        # threshold and it only uses the label that has just arrived.
        prediction = self.predict(x, horizon)
        residual = target - prediction
        denominator = max(1.0, float(x @ x))
        proposal = self.weights[horizon] + (rate / denominator) * np.outer(x, residual)
        if np.isfinite(proposal).all():
            self.weights[horizon] = proposal
        self.support[horizon] += 1


def _scratch(values: np.ndarray, config: V31Config) -> OnlineRegressor:
    # The target scale is fixed from its first arrived history window only; the
    # adapter warmup is an update gate, never a look-ahead calibration period.
    calibration = values[:config.history_windows]; location, scale = robust_scale(calibration); size = 1 + 2 * values.shape[1]
    return OnlineRegressor({h: np.zeros((size, values.shape[1])) for h in config.horizons}, location, scale, {h: 0 for h in config.horizons})


def run_target_transfer(source: FrozenSourceModel, target: pd.DataFrame, config: V31Config, entry_cycle: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict target replay: predict first, then update only when each delayed label arrives."""
    active = target.loc[target.center_cycle.ge(entry_cycle)].sort_values(["center_cycle", "window_index"]).reset_index(drop=True); values = active.loc[:, list(config.features)].to_numpy(float)
    if len(values) <= config.history_windows + max(config.horizons): return pd.DataFrame(), pd.DataFrame()
    adapter = OnlineRegressor({h: item.copy() for h, item in source.weights.items()}, source.location.copy(), source.scale.copy(), {h: 0 for h in config.horizons}); scratch = _scratch(values, config)
    pending: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}; rows: list[dict[str, object]] = []; gate_rows: list[dict[str, object]] = []; excess_count = 0; gate_active = False
    for current in range(config.history_windows, len(values)):
        # Arrived labels update the target-only adapter and scratch model after their forecasts were emitted.
        current_errors: list[tuple[float, float, int]] = []
        for horizon in config.horizons:
            key = (current - horizon, horizon)
            if key in pending:
                x_source, x_target, pred_adapter, pred_scratch = pending.pop(key); observed = values[current] - values[current - horizon]
                adapter_error = float(np.mean((pred_adapter * source.scale - observed) ** 2)); scratch_error = float(np.mean((pred_scratch * scratch.scale - observed) ** 2)); current_errors.append((adapter_error, scratch_error, horizon))
                if not gate_active:
                    adapter.update(x_source, observed / source.scale, horizon, config.adapter_learning_rate)
                scratch.update(x_target, observed / scratch.scale, horizon, config.adapter_learning_rate)
                rows.extend([
                    {"dataset": active.dataset.iloc[0], "entry_cycle": entry_cycle, "observed_index": current, "origin_index": current - horizon, "center_cycle": float(active.center_cycle.iloc[current]), "horizon": horizon, "model": "Source_Frozen", "squared_error": float(np.mean(((x_source @ source.weights[horizon]) * source.scale - observed) ** 2)), "prediction_available": True, "adapter_gate_active": gate_active},
                    {"dataset": active.dataset.iloc[0], "entry_cycle": entry_cycle, "observed_index": current, "origin_index": current - horizon, "center_cycle": float(active.center_cycle.iloc[current]), "horizon": horizon, "model": "Target_From_Scratch", "squared_error": scratch_error, "prediction_available": True, "adapter_gate_active": gate_active},
                    {"dataset": active.dataset.iloc[0], "entry_cycle": entry_cycle, "observed_index": current, "origin_index": current - horizon, "center_cycle": float(active.center_cycle.iloc[current]), "horizon": horizon, "model": "Source_Plus_Adapter_Gated", "squared_error": scratch_error if gate_active else adapter_error, "prediction_available": True, "adapter_gate_active": gate_active},
                ])
        if current_errors and current >= config.adapter_warmup_windows:
            adapter_loss = float(np.mean([item[0] for item in current_errors])); scratch_loss = float(np.mean([item[1] for item in current_errors])); excess_count = excess_count + 1 if adapter_loss > (1 + config.negative_transfer_excess) * scratch_loss else 0
            if excess_count >= config.negative_transfer_confirmations: gate_active = True
            gate_rows.append({"dataset": active.dataset.iloc[0], "entry_cycle": entry_cycle, "observed_index": current, "center_cycle": float(active.center_cycle.iloc[current]), "adapter_loss": adapter_loss, "scratch_loss": scratch_loss, "consecutive_excess": excess_count, "negative_transfer_gate_active": gate_active, "adapter_support": int(sum(adapter.support.values())), "scratch_support": int(sum(scratch.support.values()))})
        # Forecasts at current use only target history strictly before current.
        x_source = _input(values, current, source.location, source.scale); x_target = _input(values, current, scratch.location, scratch.scale)
        for horizon in config.horizons:
            if current + horizon < len(values): pending[(current, horizon)] = (x_source, x_target, adapter.predict(x_source, horizon), scratch.predict(x_target, horizon))
    return pd.DataFrame(rows), pd.DataFrame(gate_rows)


def gate_a_summary(records: pd.DataFrame, gates: pd.DataFrame, config: V31Config) -> dict[str, object]:
    if records.empty: return {"status": "FAIL", "reason": "no_target_predictions"}
    metrics = records.groupby(["model", "horizon"], as_index=False).squared_error.mean(); metrics["mae"] = np.sqrt(metrics.squared_error)
    frozen = float(metrics.loc[metrics.model.eq("Source_Frozen"), "mae"].mean()); adapter = float(metrics.loc[metrics.model.eq("Source_Plus_Adapter_Gated"), "mae"].mean()); scratch = float(metrics.loc[metrics.model.eq("Target_From_Scratch"), "mae"].mean())
    horizon_coverage = set(metrics.horizon) == set(config.horizons); gate_active = bool(gates.negative_transfer_gate_active.any()) if not gates.empty else False
    post = records.loc[(records.model == "Source_Plus_Adapter_Gated") & records.adapter_gate_active, "squared_error"]
    scratch_post = records.loc[(records.model == "Target_From_Scratch") & records.adapter_gate_active, "squared_error"]
    post_ratio = float(np.sqrt(post.mean()) / max(np.sqrt(scratch_post.mean()), 1e-12)) if len(post) and len(scratch_post) else np.nan
    improvement = 1 - adapter / max(frozen, 1e-12); negative_ok = (not gate_active) or (np.isfinite(post_ratio) and post_ratio <= 1.05)
    passed = bool(horizon_coverage and improvement >= .01 and negative_ok)
    return {"status": "PASS" if passed else "FAIL", "horizon_coverage": horizon_coverage, "frozen_mae": frozen, "adapter_gated_mae": adapter, "scratch_mae": scratch, "adapter_improvement_vs_frozen": improvement, "negative_transfer_gate_active": gate_active, "post_gate_adapter_to_scratch_mae_ratio": post_ratio, "negative_transfer_ok": negative_ok, "reason": "thresholds_not_met" if not passed else "all_preregistered_conditions_met"}
