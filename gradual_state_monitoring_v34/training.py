from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gradual_state_monitoring_v33.data import MAIN_FEATURES

from .config import GradualStateMonitoringV34Config
from .data import source_soft_boundary_labels
from .model import GradualStateTCN, set_deterministic_seed


@dataclass(frozen=True)
class RobustLocationScale:
    median: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.median) / self.scale


@dataclass
class SourceTrainingResult:
    model: GradualStateTCN
    normalizer: RobustLocationScale
    training_log: pd.DataFrame
    train_end: int
    training_origins: np.ndarray


def robust_location_scale(values: np.ndarray, calibration_windows: int, eps: float) -> RobustLocationScale:
    calibration = np.asarray(values[:calibration_windows], dtype=float)
    if len(calibration) < calibration_windows:
        raise ValueError(f"Need at least {calibration_windows} calibration windows")
    median = np.median(calibration, axis=0)
    mad = np.median(np.abs(calibration - median), axis=0) * 1.4826
    return RobustLocationScale(median, np.maximum(mad, eps))


def legal_training_origins(train_end: int, config: GradualStateMonitoringV34Config) -> np.ndarray:
    """Return origins satisfying origin + max(horizons) < train_end exactly."""
    start = config.history_windows - 1
    stop = train_end - config.source_prediction_horizon
    origins = np.arange(start, max(start, stop), dtype=int)
    if len(origins) and not bool(np.all(origins + config.source_prediction_horizon < train_end)):
        raise AssertionError("A source training target escaped the declared training interval")
    return origins


def _subsample(origins: np.ndarray, maximum: int) -> np.ndarray:
    if len(origins) <= maximum:
        return origins
    return origins[np.linspace(0, len(origins) - 1, maximum, dtype=int)]


def _make_examples(values: np.ndarray, labels: np.ndarray, origins: np.ndarray,
                   config: GradualStateMonitoringV34Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history = np.stack([values[index - config.history_windows + 1:index + 1] for index in origins]).astype(np.float32)
    adjacent = np.stack([values[index - config.history_windows + 2:index + 2] for index in origins]).astype(np.float32)
    future_delta = np.stack([values[index + config.source_prediction_horizon] - values[index] for index in origins]).astype(np.float32)
    return history, adjacent, labels[origins].astype(np.float32), future_delta


def _fit(model: GradualStateTCN, values: np.ndarray, soft_labels: np.ndarray, origins: np.ndarray,
         config: GradualStateMonitoringV34Config, seed: int, epochs: int, *, train_all: bool) -> pd.DataFrame:
    if not len(origins):
        raise ValueError("No legal source weak-supervision examples")
    history, adjacent, labels, deltas = _make_examples(values, soft_labels, origins, config)
    dataset = TensorDataset(torch.from_numpy(history), torch.from_numpy(adjacent), torch.from_numpy(labels), torch.from_numpy(deltas))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    if not train_all:
        model.freeze_for_target_adaptation()
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config.learning_rate)
    bce = nn.BCEWithLogitsLoss()
    rows: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(epochs):
        totals = np.zeros(4, dtype=float)
        count = 0
        for batch_history, batch_adjacent, batch_labels, batch_delta in loader:
            optimizer.zero_grad(set_to_none=True)
            z, logit, predicted_delta = model(batch_history)
            boundary = bce(logit, batch_labels)
            positive = batch_labels >= 0.10
            negative = batch_labels <= 1e-12
            if bool(positive.any()) and bool(negative.any()):
                rank = torch.relu(0.05 - (torch.sigmoid(logit[positive]).mean() - torch.sigmoid(logit[negative]).mean()))
            else:
                rank = torch.zeros((), dtype=z.dtype)
            adjacent_z = model.encode(batch_adjacent)
            far = (batch_labels <= 1e-12).float().unsqueeze(1)
            consistency = ((z - adjacent_z).square() * far).sum() / torch.clamp(far.sum() * z.shape[1], min=1.0)
            prediction = torch.nn.functional.mse_loss(predicted_delta, batch_delta)
            loss = (config.loss_boundary_weight * boundary + config.loss_rank_weight * rank +
                    config.loss_consistency_weight * consistency + config.loss_prediction_weight * prediction)
            loss.backward()
            optimizer.step()
            totals += np.array([boundary.item(), rank.item(), consistency.item(), prediction.item()])
            count += 1
        rows.append({
            "epoch": epoch + 1, "examples": int(len(origins)),
            "boundary_loss": float(totals[0] / max(count, 1)), "rank_loss": float(totals[1] / max(count, 1)),
            "consistency_loss": float(totals[2] / max(count, 1)), "prediction_loss": float(totals[3] / max(count, 1)),
        })
    model.eval()
    return pd.DataFrame(rows)


def train_source_model(source: pd.DataFrame, source_segments: pd.DataFrame,
                       config: GradualStateMonitoringV34Config, seed: int) -> SourceTrainingResult:
    """Fit the shared adapter/TCN using only the declared source Stage mapping."""
    set_deterministic_seed(seed)
    raw = source.loc[:, MAIN_FEATURES].to_numpy(float)
    normalizer = robust_location_scale(raw, config.calibration_windows, config.eps)
    values = normalizer.transform(raw)
    soft_labels, _ = source_soft_boundary_labels(source, source_segments, config)
    train_end = int(len(source) * config.source_train_fraction)
    origins = _subsample(legal_training_origins(train_end, config), config.source_max_training_examples)
    model = GradualStateTCN(config).to(config.device)
    log = _fit(model, values, soft_labels, origins, config, seed, config.source_epochs, train_all=True)
    log.insert(0, "dataset", str(source.dataset.iloc[0])); log.insert(1, "seed", seed)
    log["train_end"] = train_end
    log["max_target_index"] = int(origins.max() + config.source_prediction_horizon)
    log["source_stage_weak_supervision"] = 1
    return SourceTrainingResult(model=model, normalizer=normalizer, training_log=log, train_end=train_end, training_origins=origins)


def fit_target_supervised_upper_bound(source_model: GradualStateTCN, target: pd.DataFrame, target_segments: pd.DataFrame,
                                      config: GradualStateMonitoringV34Config, seed: int) -> tuple[GradualStateTCN, RobustLocationScale, pd.DataFrame]:
    """Offline-only target-label upper bound; it is never passed to the online path."""
    set_deterministic_seed(seed + 100_000)
    raw = target.loc[:, MAIN_FEATURES].to_numpy(float)
    normalizer = robust_location_scale(raw, config.calibration_windows, config.eps)
    values = normalizer.transform(raw)
    labels, _ = source_soft_boundary_labels(target, target_segments, config)
    train_end = len(target)
    origins = _subsample(legal_training_origins(train_end, config), config.source_max_training_examples)
    model = source_model.target_copy().to(config.device)
    log = _fit(model, values, labels, origins, config, seed + 100_000, config.upper_bound_epochs, train_all=True)
    log["target_supervised_upper_bound"] = 1
    log["max_target_index"] = int(origins.max() + config.source_prediction_horizon)
    return model, normalizer, log
