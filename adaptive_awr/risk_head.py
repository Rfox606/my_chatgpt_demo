from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import AdaptiveAWRConfig
from .causal_metrics import finite, robust_iqr, sigmoid


RISK_FEATURES: tuple[str, ...] = (
    "AWR_adaptive",
    "BDall_xy_v2",
    "RS50_positive",
    "TES",
    "high_AWR_high_BD_occupancy",
)


def source_split_by_stage(stages: Sequence[int], gap: int) -> tuple[np.ndarray, np.ndarray]:
    stages = np.asarray(stages, dtype=int)
    train = np.zeros(len(stages), dtype=bool)
    validation = np.zeros(len(stages), dtype=bool)
    for stage in sorted(set(stages.tolist())):
        position = np.flatnonzero(stages == stage)
        if position.size < 8:
            continue
        split = int(position.size * 0.70)
        train_end = max(0, split - gap)
        validation_start = min(position.size, split + gap)
        train[position[:train_end]] = True
        validation[position[validation_start:]] = True
    if validation.sum() < 20:
        split = int(len(stages) * 0.70)
        train[:] = False
        validation[:] = False
        train[: max(0, split - gap)] = True
        validation[min(len(stages), split + gap) :] = True
    return train, validation


def source_directions(source: pd.DataFrame, train_mask: np.ndarray, features: Sequence[str]) -> pd.DataFrame:
    """Determine signs using only source training Stage 1 and Stage 5 windows."""
    train = source.loc[train_mask].copy()
    rows = []
    for feature in features:
        early = train.loc[train["stage"] == 1, feature].to_numpy(dtype=float)
        late = train.loc[train["stage"] == 5, feature].to_numpy(dtype=float)
        delta = float(np.nanmedian(finite(late)) - np.nanmedian(finite(early))) if finite(early).size and finite(late).size else 0.0
        sign = int(np.sign(delta))
        if sign == 0:
            sign = 1
        rows.append({"feature_name": feature, "delta_source_train": delta, "direction_sign": sign})
    return pd.DataFrame(rows)


def roc_auc(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    positive = int(y.sum())
    negative = len(y) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    sorted_ranks = np.empty(len(s), dtype=float)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_s[end] == sorted_s[start]:
            end += 1
        sorted_ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = sorted_ranks
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def average_precision(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    cumulative = np.cumsum(y)
    precision = cumulative / (np.arange(len(y)) + 1)
    return float(np.sum(precision[y == 1]) / positives)


@dataclass
class RobustScaler:
    median: np.ndarray
    iqr: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, features: Sequence[str], eps: float) -> "RobustScaler":
        median = []
        widths = []
        for feature in features:
            values = finite(frame[feature].to_numpy(dtype=float))
            median.append(float(np.nanmedian(values)) if values.size else 0.0)
            widths.append(robust_iqr(values, eps))
        return cls(np.asarray(median, dtype=float), np.asarray(widths, dtype=float))

    def transform(self, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
        values = frame.loc[:, features].to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, self.median.reshape(1, -1))
        return (values - self.median.reshape(1, -1)) / self.iqr.reshape(1, -1)


@dataclass
class LogisticRiskHead:
    scaler: RobustScaler
    coefficients: np.ndarray
    success: bool
    message: str
    objective: float

    def logit(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.scaler.transform(frame, RISK_FEATURES)
        return self.coefficients[0] + transformed @ self.coefficients[1:]

    def predict(self, frame: pd.DataFrame, offset: float = 0.0) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.logit(frame) + offset, -35.0, 35.0)))

    def row(self, direction_id: str, source: str, target: str) -> dict[str, object]:
        row: dict[str, object] = {
            "direction_id": direction_id,
            "source_dataset": source,
            "target_dataset": target,
            "optimizer_success": bool(self.success),
            "optimizer_message": self.message,
            "objective": float(self.objective),
            "beta0": float(self.coefficients[0]),
        }
        for name, beta, median, width in zip(RISK_FEATURES, self.coefficients[1:], self.scaler.median, self.scaler.iqr):
            row[f"beta_{name}"] = float(beta)
            row[f"median_{name}"] = float(median)
            row[f"IQR_{name}"] = float(width)
        return row


def fit_risk_head(source: pd.DataFrame, train_mask: np.ndarray, config: AdaptiveAWRConfig) -> LogisticRiskHead:
    selected = source.loc[train_mask & source["stage"].isin([1, 2, 3, 5]).to_numpy()].copy()
    if selected.empty or selected["stage"].eq(5).sum() == 0:
        raise ValueError("Source training split does not contain both late and non-late windows.")
    labels = (selected["stage"].to_numpy(dtype=int) == 5).astype(float)
    scaler = RobustScaler.fit(selected, RISK_FEATURES, config.eps)
    X = scaler.transform(selected, RISK_FEATURES)
    sample_weights = np.where(labels == 1.0, config.positive_class_weight, 1.0)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = theta[0] + X @ theta[1:]
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))
        loss = -np.sum(sample_weights * (labels * np.log(probabilities + config.eps) + (1.0 - labels) * np.log(1.0 - probabilities + config.eps)))
        loss += config.risk_head_l2 * np.sum(theta[1:] ** 2)
        residual = sample_weights * (probabilities - labels)
        gradient = np.empty_like(theta)
        gradient[0] = residual.sum()
        gradient[1:] = X.T @ residual + 2.0 * config.risk_head_l2 * theta[1:]
        return float(loss), gradient

    initial = np.zeros(len(RISK_FEATURES) + 1, dtype=float)
    prevalence = float(np.clip(np.average(labels, weights=sample_weights), config.eps, 1.0 - config.eps))
    initial[0] = np.log(prevalence / (1.0 - prevalence))
    result = minimize(
        fun=lambda theta: objective(theta)[0],
        x0=initial,
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        bounds=[(None, None)] + [(0.0, None)] * len(RISK_FEATURES),
    )
    return LogisticRiskHead(scaler, result.x, bool(result.success), str(result.message), float(result.fun))


def choose_risk_threshold(source: pd.DataFrame, validation_mask: np.ndarray, head: LogisticRiskHead) -> dict[str, float]:
    validation = source.loc[validation_mask].copy()
    validation = validation[validation["stage"].isin([1, 2, 3, 5])].copy()
    scores = head.predict(validation)
    stage = validation["stage"].to_numpy(dtype=int)
    candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    rows = []
    for threshold in candidates:
        high = scores >= threshold
        stage5 = stage == 5
        early = np.isin(stage, [1, 2])
        recall = float(np.mean(high[stage5])) if stage5.any() else 0.0
        precision = float(np.mean(stage5[high])) if high.any() else 0.0
        fpr = float(np.mean(high[early])) if early.any() else 0.0
        rows.append((float(threshold), recall, precision, fpr))
    acceptable = [row for row in rows if row[1] >= 0.85]
    if acceptable:
        chosen = min(acceptable, key=lambda row: (row[3], -row[1], -row[2], row[0]))
        mode = "min_early_fpr_subject_to_stage5_recall_0.85"
    else:
        chosen = max(rows, key=lambda row: (0.7 * row[1] + 0.3 * row[2] - 0.5 * row[3], row[1], -row[3]))
        mode = "fallback_weighted_recall_precision_fpr"
    return {
        "risk_threshold": chosen[0],
        "source_validation_stage5_recall": chosen[1],
        "source_validation_stage5_precision": chosen[2],
        "source_validation_stage1to2_fpr": chosen[3],
        "threshold_selection_mode": mode,
    }


def source_event_logit(value: float, source_reference: Iterable[float], eps: float) -> float:
    ref = finite(source_reference)
    if ref.size == 0 or not np.isfinite(value):
        return 0.0
    return float(max((value - np.nanmedian(ref)) / robust_iqr(ref, eps), 0.0))


def event_risk(tes: float, bd_jump: float, source_tes: Iterable[float], source_bd_jump: Iterable[float], eps: float) -> float:
    return sigmoid(1.5 * source_event_logit(tes, source_tes, eps) + 0.5 * source_event_logit(bd_jump, source_bd_jump, eps))
