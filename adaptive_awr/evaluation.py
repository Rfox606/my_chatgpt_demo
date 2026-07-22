from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .risk_head import average_precision, roc_auc


def spearman(values: Iterable[float], stages: Iterable[int]) -> float:
    x = np.asarray(list(values), dtype=float)
    y = np.asarray(list(stages), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    x_rank = pd.Series(x[valid]).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y[valid]).rank(method="average").to_numpy(dtype=float)
    if np.nanstd(x_rank) == 0.0 or np.nanstd(y_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def first_stable_high(frame: pd.DataFrame, threshold: float, required_windows: int) -> tuple[float, float]:
    run = 0
    for row in frame.sort_values("window_index").itertuples(index=False):
        if float(row.evaluation_score) >= threshold:
            run += 1
        else:
            run = 0
        if run >= required_windows:
            return float(row.window_index - required_windows + 1), float(row.center_cycle)
    return float("nan"), float("nan")


def false_alarm_rate_per_1000_cycles(frame: pd.DataFrame, threshold: float) -> float:
    early = frame[frame["stage"].isin([1, 2])].sort_values("window_index")
    if early.empty:
        return float("nan")
    high = early["evaluation_score"].to_numpy(dtype=float) >= threshold
    starts = int(np.sum(high & np.r_[True, ~high[:-1]]))
    cycle_range = float(early["end_cycle"].max() - early["start_cycle"].min() + 1.0)
    return float(starts / max(cycle_range, 1.0) * 1000.0)


def recall_at_early_fpr(frame: pd.DataFrame, target_fpr: float = 0.10) -> float:
    early = frame.loc[frame["stage"].isin([1, 2]), "evaluation_score"].to_numpy(dtype=float)
    late = frame.loc[frame["stage"] == 5, "evaluation_score"].to_numpy(dtype=float)
    early = early[np.isfinite(early)]
    late = late[np.isfinite(late)]
    if early.size == 0 or late.size == 0:
        return float("nan")
    threshold = float(np.nanpercentile(early, (1.0 - target_fpr) * 100.0))
    return float(np.mean(late >= threshold))


def evaluate_scored_target(
    frame: pd.DataFrame,
    *,
    direction_id: str,
    source_dataset: str,
    target_dataset: str,
    model: str,
    threshold: float,
    update_count: int = 0,
    freeze_count: int = 0,
    rollback_count: int = 0,
) -> dict[str, object]:
    """This is the only evaluation function that consumes target stages."""
    ordered = frame.sort_values("window_index").copy()
    score = ordered["evaluation_score"].to_numpy(dtype=float)
    stage = ordered["stage"].to_numpy(dtype=int)
    is_stage5 = stage == 5
    early = np.isin(stage, [1, 2])
    high = score >= threshold
    stage5_recall = float(np.mean(high[is_stage5])) if is_stage5.any() else float("nan")
    early_fpr = float(np.mean(high[early])) if early.any() else float("nan")
    high_idx, high_cycle = first_stable_high(ordered, threshold, 3)
    stage5_rows = ordered[ordered["stage"] == 5]
    stage5_start = float(stage5_rows["start_cycle"].min()) if not stage5_rows.empty else float("nan")
    lead = stage5_start - high_cycle if np.isfinite(stage5_start) and np.isfinite(high_cycle) else float("nan")
    return {
        "direction_id": direction_id,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "model": model,
        "risk_threshold": float(threshold),
        "Stage5_AUROC": roc_auc(is_stage5.astype(int), score),
        "Stage5_AUPRC": average_precision(is_stage5.astype(int), score),
        "Stage5_AUPRC_baseline": float(np.mean(is_stage5)),
        "Stage5_Recall": stage5_recall,
        "Stage1to2_FPR": early_fpr,
        "Recall_at_10pct_Stage1to2_FPR": recall_at_early_fpr(ordered),
        "Risk_Stage_Spearman": spearman(score, stage),
        "first_stable_high_window": high_idx,
        "first_stable_high_cycle": high_cycle,
        "Stage5_start_cycle": stage5_start,
        "detection_lead_cycles_relative_to_Stage5": lead,
        "early_false_alarms_per_1000_cycles": false_alarm_rate_per_1000_cycles(ordered, threshold),
        "online_update_count": int(update_count),
        "adapter_freeze_count": int(freeze_count),
        "adapter_rollback_count": int(rollback_count),
    }


def add_stage5_suppression(summary: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    suppression: dict[str, float] = {}
    for direction_id, group in scores.groupby("direction_id", sort=True):
        b1 = group[(group["model"] == "B1") & (group["stage"] == 5)]["final_risk"]
        b4 = group[(group["model"] == "B4") & (group["stage"] == 5)]["final_risk"]
        suppression[str(direction_id)] = float(np.nanmedian(b1) - np.nanmedian(b4)) if len(b1) and len(b4) else float("nan")
    out["Stage5_risk_suppression_B1_minus_B4"] = out["direction_id"].map(suppression)
    out["adaptation_safety_failure"] = out["Stage5_risk_suppression_B1_minus_B4"] > 0.10
    return out


def adaptation_acceptance(summary: pd.DataFrame) -> list[dict[str, object]]:
    """Report the requested acceptance criteria without tuning to make them pass."""
    b0 = summary[summary["model"] == "B0"].set_index("direction_id")
    b4 = summary[summary["model"] == "B4"].set_index("direction_id")
    shared = sorted(set(b0.index).intersection(b4.index))
    if not shared:
        return [{"criterion": "B0/B4 bidirectional results present", "observed": "missing", "status": "FAIL"}]
    b0_common = b0.loc[shared]
    b4_common = b4.loc[shared]
    checks = [
        (
            "B4 worst AUROC drop from B0 <= 0.02",
            float(b4_common["Stage5_AUROC"].min() - b0_common["Stage5_AUROC"].min()),
            lambda value: value >= -0.02,
        ),
        (
            "B4 worst AUPRC drop from B0 <= 0.03",
            float(b4_common["Stage5_AUPRC"].min() - b0_common["Stage5_AUPRC"].min()),
            lambda value: value >= -0.03,
        ),
        (
            "B4 Stage5 Recall is not below B0 in either direction",
            float((b4_common["Stage5_Recall"] - b0_common["Stage5_Recall"]).min()),
            lambda value: value >= 0.0,
        ),
        (
            "B4 Stage1-2 FPR increase <= 0.05 in either direction",
            float((b4_common["Stage1to2_FPR"] - b0_common["Stage1to2_FPR"]).max()),
            lambda value: value <= 0.05,
        ),
        (
            "Stage5 risk suppression <= 0.10 in either direction",
            float(b4_common["Stage5_risk_suppression_B1_minus_B4"].max()),
            lambda value: value <= 0.10,
        ),
    ]
    return [
        {"criterion": name, "observed": observed, "status": "PASS" if predicate(observed) else "FAIL"}
        for name, observed, predicate in checks
    ]


def dataframe_markdown(frame: pd.DataFrame, columns: Sequence[str] | None = None, max_rows: int = 50) -> str:
    show = frame.loc[:, list(columns)].head(max_rows) if columns is not None else frame.head(max_rows)
    if show.empty:
        return "_No rows._"
    headers = [str(column) for column in show.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in show.itertuples(index=False, name=None):
        formatted = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted.append("" if not np.isfinite(value) else f"{value:.4f}")
            else:
                formatted.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)
