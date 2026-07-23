from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import V32Config
from .data import robust_scale


MODEL_NAMES = (
    "Persistence",
    "Source_Frozen",
    "Target_From_Scratch",
    "Source_Adapter",
    "Negative_transfer_Gated_Mixture",
)


def _input(values: np.ndarray, index: int, location: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Causal context available at forecast origin ``index`` only."""
    current = (values[index] - location) / scale
    lag = max(0, index - 4)
    delta = (values[index] - values[lag]) / scale
    return np.r_[1.0, current, delta]


def _digest(weights: dict[int, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for horizon in sorted(weights):
        digest.update(np.asarray(weights[horizon], dtype=np.float64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenSourceModel:
    weights: dict[int, np.ndarray]
    location: np.ndarray
    scale: np.ndarray
    source_dataset: str
    train_windows: int
    parameter_sha256: str


def fit_source_frozen(source: pd.DataFrame, config: V32Config) -> FrozenSourceModel:
    values = source.loc[:, list(config.features)].to_numpy(float)
    train = min(config.source_train_windows, len(values))
    location, scale = robust_scale(values[:train])
    weights: dict[int, np.ndarray] = {}
    size = 1 + 2 * values.shape[1]
    for horizon in config.horizons:
        rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for index in range(config.history_windows, train - horizon):
            rows.append(_input(values, index, location, scale))
            targets.append((values[index + horizon] - values[index]) / scale)
        if rows:
            x = np.asarray(rows)
            y = np.asarray(targets)
            regular = config.ridge_alpha * np.eye(x.shape[1])
            weights[horizon] = np.linalg.solve(x.T @ x + regular, x.T @ y)
        else:
            weights[horizon] = np.zeros((size, values.shape[1]))
    frozen_weights = {h: weight.copy() for h, weight in weights.items()}
    return FrozenSourceModel(
        frozen_weights,
        location.copy(),
        scale.copy(),
        str(source.dataset.iloc[0]),
        train,
        _digest(frozen_weights),
    )


@dataclass
class OnlineRegressor:
    weights: dict[int, np.ndarray]
    location: np.ndarray
    scale: np.ndarray
    support: dict[int, int]

    def clone(self) -> "OnlineRegressor":
        return copy.deepcopy(self)

    def predict_delta(self, x: np.ndarray, horizon: int) -> np.ndarray:
        return x @ self.weights[horizon]

    def update(self, x: np.ndarray, target: np.ndarray, horizon: int, rate: float) -> None:
        prediction = self.predict_delta(x, horizon)
        residual = target - prediction
        denominator = max(1.0, float(x @ x))
        proposal = self.weights[horizon] + (rate / denominator) * np.outer(x, residual)
        if np.isfinite(proposal).all():
            self.weights[horizon] = proposal
        self.support[horizon] += 1


def _scratch(values: np.ndarray, config: V32Config) -> OnlineRegressor:
    # This deliberately reads only the first arrived context block, not the
    # eventual target length nor any future target data.
    calibration = values[: config.history_windows]
    location, scale = robust_scale(calibration)
    size = 1 + 2 * values.shape[1]
    return OnlineRegressor(
        {h: np.zeros((size, values.shape[1])) for h in config.horizons},
        location,
        scale,
        {h: 0 for h in config.horizons},
    )


@dataclass
class _Pending:
    origin: int
    horizon: int
    base: np.ndarray
    x_source: np.ndarray
    x_target: np.ndarray
    predictions: dict[str, np.ndarray]
    adapter_weight: float
    gate_active: bool


class TransferEngine:
    """Predict-then-update replay with independent Adapter and Scratch arms."""

    def __init__(self, source: FrozenSourceModel, values: np.ndarray, config: V32Config) -> None:
        self.source = source
        self.values = values
        self.config = config
        self.adapter = OnlineRegressor(
            {h: weight.copy() for h, weight in source.weights.items()},
            source.location.copy(),
            source.scale.copy(),
            {h: 0 for h in config.horizons},
        )
        self.scratch = _scratch(values, config)
        self.pending: dict[tuple[int, int], _Pending] = {}
        self.excess_count = 0
        self.gate_active = False
        self.adapter_weight = 0.80

    def clone_frozen(self) -> "TransferEngine":
        clone = copy.deepcopy(self)
        return clone

    def _issue(self, origin: int) -> None:
        base = self.values[origin]
        x_source = _input(self.values, origin, self.source.location, self.source.scale)
        x_target = _input(self.values, origin, self.scratch.location, self.scratch.scale)
        for horizon in self.config.horizons:
            adapter = base + self.adapter.predict_delta(x_source, horizon) * self.source.scale
            scratch = base + self.scratch.predict_delta(x_target, horizon) * self.scratch.scale
            frozen = base + (x_source @ self.source.weights[horizon]) * self.source.scale
            mixture = self.adapter_weight * adapter + (1.0 - self.adapter_weight) * scratch
            self.pending[(origin + horizon, horizon)] = _Pending(
                origin,
                horizon,
                base.copy(),
                x_source,
                x_target,
                {
                    "Persistence": base.copy(),
                    "Source_Frozen": frozen,
                    "Target_From_Scratch": scratch,
                    "Source_Adapter": adapter,
                    "Negative_transfer_Gated_Mixture": mixture,
                },
                self.adapter_weight,
                self.gate_active,
            )

    def observe(self, current: int, update: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        records: list[dict[str, object]] = []
        logs: list[dict[str, object]] = []
        due: list[tuple[_Pending, float, float]] = []
        for horizon in self.config.horizons:
            pending = self.pending.pop((current, horizon), None)
            if pending is None:
                continue
            observed_delta = self.values[current] - self.values[pending.origin]
            adapter_delta = pending.predictions["Source_Adapter"] - pending.base
            scratch_delta = pending.predictions["Target_From_Scratch"] - pending.base
            adapter_error = float(np.mean(np.abs(pending.predictions["Source_Adapter"] - self.values[current])))
            scratch_error = float(np.mean(np.abs(pending.predictions["Target_From_Scratch"] - self.values[current])))
            due.append((pending, adapter_error, scratch_error))
            for model, prediction in pending.predictions.items():
                absolute = float(np.mean(np.abs(prediction - self.values[current])))
                records.append(
                    {
                        "observed_index": current,
                        "origin_index": pending.origin,
                        "horizon": horizon,
                        "model": model,
                        "absolute_error": absolute,
                        "squared_error": float(np.mean((prediction - self.values[current]) ** 2)),
                        "prediction_available": True,
                        "adapter_gate_active_at_issue": pending.gate_active,
                        "adapter_weight_at_issue": pending.adapter_weight,
                        "predict_then_update": True,
                    }
                )
            if update:
                # Adapter and Scratch update independently.  The gate never
                # overwrites adapter error or parameters with scratch values.
                self.adapter.update(pending.x_source, observed_delta / self.source.scale, horizon, self.config.adapter_learning_rate)
                self.scratch.update(pending.x_target, observed_delta / self.scratch.scale, horizon, self.config.adapter_learning_rate)
        if due and update:
            adapter_loss = float(np.mean([item[1] for item in due]))
            scratch_loss = float(np.mean([item[2] for item in due]))
            if current >= self.config.adapter_warmup_windows:
                self.excess_count = (
                    self.excess_count + 1
                    if adapter_loss > (1.0 + self.config.negative_transfer_excess) * scratch_loss
                    else 0
                )
                if self.excess_count >= self.config.negative_transfer_confirmations:
                    self.gate_active = True
                    self.adapter_weight = 0.0
                elif not self.gate_active:
                    self.adapter_weight = 0.80
            logs.append(
                {
                    "observed_index": current,
                    "adapter_mae": adapter_loss,
                    "scratch_mae": scratch_loss,
                    "consecutive_excess": self.excess_count,
                    "negative_transfer_gate_active": self.gate_active,
                    "gated_mixture_adapter_weight": self.adapter_weight,
                    "gated_mixture_scratch_weight": 1.0 - self.adapter_weight,
                    "adapter_support": int(sum(self.adapter.support.values())),
                    "scratch_support": int(sum(self.scratch.support.values())),
                    "source_frozen_parameter_sha256": self.source.parameter_sha256,
                    "source_frozen_parameters_updated": False,
                    "adapter_and_scratch_independent": True,
                }
            )
        if current + min(self.config.horizons) < len(self.values):
            self._issue(current)
        return records, logs


def run_target_transfer(
    source: FrozenSourceModel,
    target: pd.DataFrame,
    config: V32Config,
    entry_cycle: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = target.loc[target.center_cycle.ge(entry_cycle)].sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    values = active.loc[:, list(config.features)].to_numpy(float)
    if len(values) <= config.history_windows + max(config.horizons):
        return pd.DataFrame(), pd.DataFrame()
    engine = TransferEngine(source, values, config)
    records: list[dict[str, object]] = []
    logs: list[dict[str, object]] = []
    # The target-only calibration block has arrived before the first model is
    # instantiated logically; no target value beyond it is read at startup.
    for current in range(config.history_windows - 1, len(values)):
        current_records, current_logs = engine.observe(current, update=True)
        for row in current_records:
            row.update({"dataset": active.dataset.iloc[0], "entry_cycle": entry_cycle, "center_cycle": float(active.center_cycle.iloc[current])})
        for row in current_logs:
            row.update({"dataset": active.dataset.iloc[0], "entry_cycle": entry_cycle, "center_cycle": float(active.center_cycle.iloc[current])})
        records.extend(current_records)
        logs.extend(current_logs)
    return pd.DataFrame(records), pd.DataFrame(logs)


def prefix_freeze_metrics(
    source: FrozenSourceModel, target: pd.DataFrame, config: V32Config
) -> pd.DataFrame:
    """Freeze each target model at 10/20/40/60/80%, then score its future."""
    active = target.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    values = active.loc[:, list(config.features)].to_numpy(float)
    rows: list[dict[str, object]] = []
    for fraction in config.evaluation_prefixes:
        prefix_end = max(config.history_windows, int(np.floor(len(values) * fraction)) - 1)
        prefix_end = min(prefix_end, len(values) - max(config.horizons) - 1)
        engine = TransferEngine(source, values, config)
        for current in range(config.history_windows - 1, prefix_end + 1):
            engine.observe(current, update=True)
        frozen = engine.clone_frozen()
        held: list[dict[str, object]] = []
        for current in range(prefix_end + 1, len(values)):
            current_rows, _ = frozen.observe(current, update=False)
            held.extend(current_rows)
        table = pd.DataFrame(held)
        if table.empty:
            continue
        for (model, horizon), group in table.groupby(["model", "horizon"], sort=True):
            rows.append(
                {
                    "dataset": active.dataset.iloc[0],
                    "prefix_fraction": fraction,
                    "prefix_end_index": prefix_end,
                    "prefix_end_cycle": float(active.center_cycle.iloc[prefix_end]),
                    "evaluation": "frozen_model_common_future",
                    "model": model,
                    "horizon": int(horizon),
                    "mae": float(group.absolute_error.mean()),
                    "prediction_count": int(len(group)),
                    "model_updated_during_evaluation": False,
                }
            )
        for model, group in table.groupby("model", sort=True):
            by_horizon = group.groupby("horizon").absolute_error.mean()
            if set(by_horizon.index) == set(config.horizons):
                weighted = sum(by_horizon[h] * weight for h, weight in zip(config.horizons, (1.0, 0.5, 0.25))) / 1.75
                rows.append(
                    {
                        "dataset": active.dataset.iloc[0],
                        "prefix_fraction": fraction,
                        "prefix_end_index": prefix_end,
                        "prefix_end_cycle": float(active.center_cycle.iloc[prefix_end]),
                        "evaluation": "frozen_model_common_future",
                        "model": model,
                        "horizon": "weighted",
                        "mae": float(weighted),
                        "prediction_count": int(len(group)),
                        "model_updated_during_evaluation": False,
                    }
                )
    return pd.DataFrame(rows)


def gate_a_summary(prefix_metrics: pd.DataFrame, weight_log: pd.DataFrame, config: V32Config) -> dict[str, object]:
    if prefix_metrics.empty:
        return {"status": "FAIL", "reason": "no_target_predictions"}
    weighted = prefix_metrics.loc[prefix_metrics.horizon.eq("weighted")].copy()
    pivot = weighted.pivot(index="prefix_fraction", columns="model", values="mae")
    required = set(MODEL_NAMES)
    coverage = required.issubset(pivot.columns) and set(config.evaluation_prefixes).issubset(pivot.index)
    adapter_better_frozen = (pivot["Source_Adapter"] < pivot["Source_Frozen"]).mean() if coverage else 0.0
    adapter_close_scratch = (pivot["Source_Adapter"] <= 1.02 * pivot["Target_From_Scratch"]).mean() if coverage else 0.0
    gated_weight = float(weight_log.gated_mixture_adapter_weight.mean()) if not weight_log.empty else 0.0
    gate_active = bool(weight_log.negative_transfer_gate_active.any()) if not weight_log.empty else False
    transfer_evidence = bool(coverage and adapter_better_frozen >= 0.6 and adapter_close_scratch >= 0.6)
    return {
        "status": "PASS" if transfer_evidence else "FAIL",
        "reason": "adapter_has_prefix_transfer_evidence" if transfer_evidence else "adapter_prefix_transfer_conditions_not_met",
        "prefix_coverage": coverage,
        "adapter_better_than_source_frozen_fraction": float(adapter_better_frozen),
        "adapter_close_to_scratch_fraction": float(adapter_close_scratch),
        "negative_transfer_gate_active": gate_active,
        "mean_gated_mixture_adapter_weight": gated_weight,
        "gated_mixture_is_not_adapter": True,
        "migration_claim_permitted": transfer_evidence,
    }
