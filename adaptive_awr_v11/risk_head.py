from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .causal_metrics import finite, robust_iqr, sigmoid
from .config import AdaptiveAWRV11Config, STAGE_SAMPLE_WEIGHT, STAGE_SOFT_TARGET


RISK_FEATURES: tuple[str, ...] = (
    "AWR_adaptive",
    "BDall_xy_v2",
    "RS50_positive",
    "TES_clean",
    "high_AWR_high_BD_occupancy",
)


def source_split_by_stage(stages: Sequence[int], gap: int) -> tuple[np.ndarray, np.ndarray]:
    stage_array = np.asarray(stages, dtype=int)
    train = np.zeros(len(stage_array), dtype=bool)
    validation = np.zeros(len(stage_array), dtype=bool)
    for stage in sorted(set(stage_array.tolist())):
        indices = np.flatnonzero(stage_array == stage)
        if len(indices) < 8:
            continue
        split = int(len(indices) * 0.70)
        train[indices[: max(0, split - gap)]] = True
        validation[indices[min(len(indices), split + gap) :]] = True
    if validation.sum() < 20:
        split = int(len(stage_array) * 0.70)
        train[:] = False
        validation[:] = False
        train[: max(0, split - gap)] = True
        validation[min(len(stage_array), split + gap) :] = True
    return train, validation


def source_directions(source: pd.DataFrame, train_mask: np.ndarray, features: Sequence[str]) -> pd.DataFrame:
    selected = source.loc[train_mask]
    rows = []
    for feature in features:
        early = finite(selected.loc[selected["stage"] == 1, feature])
        late = finite(selected.loc[selected["stage"] == 5, feature])
        delta = float(np.nanmedian(late) - np.nanmedian(early)) if early.size and late.size else 0.0
        rows.append({"feature_name": feature, "direction_sign": int(np.sign(delta)) or 1, "source_train_delta": delta})
    return pd.DataFrame(rows)


def roc_auc(labels: Iterable[int], values: Iterable[float]) -> float:
    y, x = np.asarray(labels, dtype=int), np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    y, x = y[valid], x[valid]
    pos, neg = int(y.sum()), len(y) - int(y.sum())
    if not pos or not neg:
        return float("nan")
    ranks = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(labels: Iterable[int], values: Iterable[float]) -> float:
    y, x = np.asarray(labels, dtype=int), np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    y, x = y[valid], x[valid]
    if not int(y.sum()):
        return float("nan")
    ordered = y[np.argsort(-x, kind="mergesort")]
    precision = np.cumsum(ordered) / (np.arange(len(ordered)) + 1)
    return float(precision[ordered == 1].mean())


def spearman(values: Iterable[float], stages: Iterable[int]) -> float:
    x, y = np.asarray(values, dtype=float), np.asarray(stages, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    xr, yr = pd.Series(x[valid]).rank().to_numpy(), pd.Series(y[valid]).rank().to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1]) if np.std(xr) and np.std(yr) else float("nan")


@dataclass
class RobustScaler:
    median: np.ndarray
    iqr: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, eps: float) -> "RobustScaler":
        medians, widths = [], []
        for feature in RISK_FEATURES:
            values = finite(frame[feature])
            medians.append(float(np.nanmedian(values)) if values.size else 0.0)
            widths.append(robust_iqr(values, eps))
        return cls(np.asarray(medians, dtype=float), np.asarray(widths, dtype=float))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, RISK_FEATURES].to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, self.median.reshape(1, -1))
        return (values - self.median.reshape(1, -1)) / self.iqr.reshape(1, -1)

    def transform_row(self, values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        array = np.where(np.isfinite(array), array, self.median)
        return (array - self.median) / self.iqr


@dataclass
class SoftRiskHead:
    scaler: RobustScaler
    coefficients: np.ndarray
    l2: float
    success: bool
    message: str
    objective: float

    def logit(self, frame: pd.DataFrame) -> np.ndarray:
        return self.coefficients[0] + self.scaler.transform(frame) @ self.coefficients[1:]

    def logit_row(self, values: Sequence[float]) -> float:
        return float(self.coefficients[0] + self.scaler.transform_row(values) @ self.coefficients[1:])

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.logit(frame), -35.0, 35.0)))

    def parameter_row(self, direction_id: str, source_dataset: str, target_dataset: str) -> dict[str, object]:
        row: dict[str, object] = {
            "direction_id": direction_id,
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "selected_l2": self.l2,
            "optimizer_success": self.success,
            "optimizer_message": self.message,
            "objective": self.objective,
            "beta0": float(self.coefficients[0]),
        }
        for feature, beta, median, width in zip(RISK_FEATURES, self.coefficients[1:], self.scaler.median, self.scaler.iqr):
            row[f"beta_{feature}"] = float(beta)
            row[f"median_{feature}"] = float(median)
            row[f"IQR_{feature}"] = float(width)
        return row


def fit_soft_risk_head(frame: pd.DataFrame, train_mask: np.ndarray, l2: float, config: AdaptiveAWRV11Config) -> SoftRiskHead:
    selected = frame.loc[train_mask & ~frame["is_restart_guard"].astype(bool).to_numpy()].copy()
    if selected.empty:
        raise ValueError("No source training windows remain after restart guard exclusion.")
    y = selected["stage"].map(STAGE_SOFT_TARGET).to_numpy(dtype=float)
    weights = selected["stage"].map(STAGE_SAMPLE_WEIGHT).to_numpy(dtype=float)
    scaler = RobustScaler.fit(selected, config.eps)
    X = scaler.transform(selected)
    weight_sum = float(weights.sum())

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        logit = theta[0] + X @ theta[1:]
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -35.0, 35.0)))
        bce = -(y * np.log(probability + config.eps) + (1.0 - y) * np.log(1.0 - probability + config.eps))
        loss = float(np.sum(weights * bce) / weight_sum + l2 * np.sum(theta[1:] ** 2))
        residual = weights * (probability - y) / weight_sum
        gradient = np.empty_like(theta)
        gradient[0] = residual.sum()
        gradient[1:] = X.T @ residual + 2.0 * l2 * theta[1:]
        return loss, gradient

    result = minimize(
        lambda theta: objective(theta)[0],
        np.zeros(len(RISK_FEATURES) + 1),
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        bounds=[config.beta0_bounds] + [config.beta_bounds] * len(RISK_FEATURES),
    )
    return SoftRiskHead(scaler, result.x, float(l2), bool(result.success), str(result.message), float(result.fun))


def validation_metrics(frame: pd.DataFrame, head: SoftRiskHead, l2: float) -> dict[str, object]:
    selected = frame.loc[~frame["is_restart_guard"].astype(bool)].copy()
    logits = head.logit(selected)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))
    stage = selected["stage"].to_numpy(dtype=int)
    soft = selected["stage"].map(STAGE_SOFT_TARGET).to_numpy(dtype=float)
    weights = selected["stage"].map(STAGE_SAMPLE_WEIGHT).to_numpy(dtype=float)
    early = np.isin(stage, [1, 2])
    stage5 = stage == 5
    stage45 = stage >= 4
    early_threshold = float(np.nanpercentile(logits[early], 90)) if early.any() else np.inf
    stage45_recall = float(np.mean(logits[stage45] >= early_threshold)) if stage45.any() else float("nan")
    brier = float(np.sum(weights * (probabilities - soft) ** 2) / weights.sum())
    stage_spearman = spearman(logits, stage)
    normalized_spearman = (stage_spearman + 1.0) / 2.0 if np.isfinite(stage_spearman) else 0.0
    return {
        "l2": l2,
        "Stage5_AUROC": roc_auc(stage5.astype(int), logits),
        "Stage5_AUPRC": average_precision(stage5.astype(int), logits),
        "Risk_Stage_Spearman": stage_spearman,
        "soft_target_brier": brier,
        "Stage4to5_recall_at_10pct_early_fpr": stage45_recall,
        "selection_score": float(
            0.30 * roc_auc(stage5.astype(int), logits)
            + 0.25 * average_precision(stage5.astype(int), logits)
            + 0.20 * normalized_spearman
            + 0.15 * (1.0 - brier)
            + 0.10 * stage45_recall
        ),
        "max_abs_nonintercept_beta": float(np.max(np.abs(head.coefficients[1:]))),
        "stage4_training_included": True,
        "weighted_mean_loss": head.objective,
    }


def fit_regularization_grid(frame: pd.DataFrame, train_mask: np.ndarray, validation_mask: np.ndarray, config: AdaptiveAWRV11Config) -> tuple[SoftRiskHead, pd.DataFrame, pd.DataFrame]:
    validation = frame.loc[validation_mask].copy()
    rows, candidates = [], []
    for l2 in config.l2_grid:
        head = fit_soft_risk_head(frame, train_mask, l2, config)
        metrics = validation_metrics(validation, head, l2)
        metrics["optimizer_success"] = head.success
        metrics["optimizer_message"] = head.message
        rows.append(metrics)
        candidates.append(head)
    grid = pd.DataFrame(rows).sort_values("l2").reset_index(drop=True)
    best_index = int(grid["selection_score"].to_numpy(dtype=float).argmax())
    return candidates[best_index], grid, validation_metrics(validation, candidates[best_index], candidates[best_index].l2)


def _threshold_stats(logits: np.ndarray, stage: np.ndarray, threshold: float) -> dict[str, float]:
    high = logits >= threshold
    early = np.isin(stage, [1, 2])
    stage5 = stage == 5
    stage45 = stage >= 4
    return {
        "threshold": float(threshold),
        "Stage5_recall": float(np.mean(high[stage5])) if stage5.any() else 0.0,
        "Stage5_precision": float(np.mean(stage5[high])) if high.any() else 0.0,
        "Stage45_recall": float(np.mean(high[stage45])) if stage45.any() else 0.0,
        "Stage1to2_fpr": float(np.mean(high[early])) if early.any() else 0.0,
    }


def select_logit_thresholds(validation: pd.DataFrame, head: SoftRiskHead, config: AdaptiveAWRV11Config) -> dict[str, object]:
    selected = validation.loc[~validation["is_restart_guard"].astype(bool)].copy()
    logits = head.logit(selected)
    stage = selected["stage"].to_numpy(dtype=int)
    candidates = np.unique(logits[np.isfinite(logits)])
    stats = [_threshold_stats(logits, stage, float(value)) for value in candidates]
    high_feasible = [row for row in stats if row["Stage5_recall"] >= 0.85 and row["Stage1to2_fpr"] <= 0.10]
    if high_feasible:
        high = sorted(high_feasible, key=lambda row: (row["Stage1to2_fpr"], -row["Stage5_precision"], -row["threshold"], -row["Stage5_recall"]))[0]
        high_mode = "feasible_stage5_recall_and_early_fpr"
    else:
        high = max(stats, key=lambda row: 0.55 * row["Stage5_recall"] + 0.25 * row["Stage5_precision"] - 0.40 * row["Stage1to2_fpr"])
        high_mode = "fallback_weighted_stage5_recall_precision_early_fpr"
    watch_feasible = [
        row for row in stats
        if row["Stage45_recall"] >= 0.80
        and row["Stage1to2_fpr"] <= 0.20
        and row["threshold"] <= high["threshold"] - 0.25
    ]
    if watch_feasible:
        watch = sorted(watch_feasible, key=lambda row: (row["Stage1to2_fpr"], -row["Stage45_recall"], -row["threshold"]))[0]
        watch_mode = "feasible_stage45_recall_and_early_fpr"
    else:
        lower = [row for row in stats if row["threshold"] <= high["threshold"] - 0.25]
        watch = max(lower or stats, key=lambda row: 0.60 * row["Stage45_recall"] - 0.25 * row["Stage1to2_fpr"] + 0.15 * row["Stage5_precision"])
        watch_mode = "fallback_constrained_below_high"
    return {
        "watch_logit_threshold": watch["threshold"],
        "high_logit_threshold": high["threshold"],
        "watch_probability_equivalent": sigmoid(watch["threshold"]),
        "high_probability_equivalent": sigmoid(high["threshold"]),
        "watch_Stage45_recall": watch["Stage45_recall"],
        "watch_Stage1to2_fpr": watch["Stage1to2_fpr"],
        "high_Stage5_recall": high["Stage5_recall"],
        "high_Stage5_precision": high["Stage5_precision"],
        "high_Stage1to2_fpr": high["Stage1to2_fpr"],
        "high_selection_mode": high_mode,
        "watch_selection_mode": watch_mode,
    }
