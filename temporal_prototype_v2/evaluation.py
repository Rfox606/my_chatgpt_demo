from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    ranks = rankdata(score)
    positives = y == 1
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) / (positives.sum() * (~positives).sum()))


def _average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ordered = y[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / positives)


def _macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    scores = []
    for value in range(1, 6):
        tp = np.sum((y == value) & (pred == value))
        fp = np.sum((y != value) & (pred == value))
        fn = np.sum((y == value) & (pred != value))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def _spearman(y: np.ndarray, score: np.ndarray) -> float:
    if len(y) < 2 or np.std(y) == 0 or np.std(score) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(y), rankdata(score))[0, 1])


def calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            result += mask.mean() * abs(probability[mask].mean() - y[mask].mean())
    return float(result)


def stage_metrics(stage: Iterable[int], posterior: np.ndarray, confidence: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(list(stage), dtype=int)
    posterior = np.asarray(posterior, dtype=float)
    expected = posterior @ np.arange(1, 6)
    pred = posterior.argmax(axis=1) + 1
    stage5 = (y == 5).astype(int)
    stage45 = y >= 4
    mask45 = np.isin(y, [4, 5])
    early = np.isin(y, [1, 2])
    high = pred >= 4
    probability5 = posterior[:, 4]
    return {
        "Stage5_AUROC": _binary_auc(stage5, probability5),
        "Stage5_AUPRC": _average_precision(stage5, probability5),
        "Stage45_AUROC": _binary_auc((y >= 5).astype(int)[mask45], probability5[mask45]) if mask45.any() else float("nan"),
        "Stage45_AUPRC": _average_precision((y >= 5).astype(int)[mask45], probability5[mask45]) if mask45.any() else float("nan"),
        "Ordinal_MAE": float(np.mean(np.abs(pred - y))),
        "ordinal_macro_F1": _macro_f1(y, pred),
        "risk_stage_spearman": _spearman(y, expected),
        "Stage5_recall": float(np.mean(high[y == 5])) if np.any(y == 5) else float("nan"),
        "Stage1to2_false_high_rate": float(np.mean(high[early])) if early.any() else float("nan"),
        "ECE": calibration_error(stage5, probability5),
        "Brier": float(np.mean((probability5 - stage5) ** 2)),
        "mean_posterior_entropy": float(np.mean(-np.sum(posterior * np.log(np.clip(posterior, 1e-8, 1)), axis=1))),
        "mean_confidence": float(np.mean(confidence)) if confidence is not None else float(np.mean(np.max(posterior, axis=1))),
    }


def source_selection_score(metrics: dict[str, float]) -> float:
    values = metrics.copy()
    for key, value in values.items():
        if not np.isfinite(value):
            values[key] = 0.0
    return float(
        0.30 * values["Stage5_AUROC"]
        + 0.20 * values["Stage5_AUPRC"]
        + 0.20 * values["ordinal_macro_F1"]
        + 0.15 * ((values["risk_stage_spearman"] + 1) / 2)
        + 0.15 * (1 - min(values["Ordinal_MAE"] / 4, 1))
    )


def decile_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(frame)
    for decile in range(10):
        lower, upper = int(n * decile / 10), int(n * (decile + 1) / 10)
        part = frame.iloc[lower:upper]
        metrics = stage_metrics(part["stage"], part[[f"stage_posterior_{i}" for i in range(1, 6)]].to_numpy(), part["posterior_confidence"])
        rows.append({"decile": f"{decile * 10}-{(decile + 1) * 10}%", "count": len(part), **metrics,
                     "accepted_pseudo_state_count": int(part["accepted"].sum()),
                     "prototype_update_count": int(part["prototype_updated"].sum())})
    return pd.DataFrame(rows)
