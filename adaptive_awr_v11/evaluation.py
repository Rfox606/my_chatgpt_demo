from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .causal_metrics import sigmoid
from .config import STAGE_SOFT_TARGET
from .risk_head import average_precision, roc_auc, spearman


def first_cycle(frame: pd.DataFrame, mask: np.ndarray) -> float:
    matched = frame.loc[mask]
    return float(matched["center_cycle"].iloc[0]) if not matched.empty else float("nan")


def episode_count(mask: np.ndarray) -> int:
    return int(np.sum(mask & np.r_[True, ~mask[:-1]])) if len(mask) else 0


def high_episodes_per_1000_cycles(frame: pd.DataFrame, high: np.ndarray) -> float:
    early = frame[frame["stage"].isin([1, 2])].copy()
    if early.empty:
        return float("nan")
    local = high[frame["stage"].isin([1, 2]).to_numpy()]
    span = float(early["end_cycle"].max() - early["start_cycle"].min() + 1.0)
    return float(episode_count(local) * 1000.0 / max(span, 1.0))


def recall_at_fpr(frame: pd.DataFrame) -> float:
    early = frame.loc[frame["stage"].isin([1, 2]), "evaluation_logit"].to_numpy(dtype=float)
    late = frame.loc[frame["stage"] == 5, "evaluation_logit"].to_numpy(dtype=float)
    if not len(early) or not len(late):
        return float("nan")
    return float(np.mean(late >= np.nanpercentile(early, 90)))


def soft_brier(frame: pd.DataFrame) -> float:
    soft = frame["stage"].map(STAGE_SOFT_TARGET).to_numpy(dtype=float)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(frame["evaluation_logit"].to_numpy(dtype=float), -35.0, 35.0)))
    return float(np.mean((probabilities - soft) ** 2))


def evaluate_target(
    frame: pd.DataFrame,
    *,
    direction_id: str,
    source_dataset: str,
    target_dataset: str,
    model: str,
    watch_threshold: float,
    high_threshold: float,
    event_log: pd.DataFrame | None = None,
) -> dict[str, object]:
    """The sole v1.1 evaluation boundary: target stage is consumed here only."""
    ordered = frame.sort_values("window_index").reset_index(drop=True)
    stage = ordered["stage"].to_numpy(dtype=int)
    score = ordered["evaluation_logit"].to_numpy(dtype=float)
    high, watch = score >= high_threshold, score >= watch_threshold
    early = np.isin(stage, [1, 2])
    stage5, stage45 = stage == 5, stage >= 4
    stage5_start = first_cycle(ordered, stage5)
    first_watch, first_high = first_cycle(ordered, watch), first_cycle(ordered, high)
    events = event_log if event_log is not None else pd.DataFrame()
    counts = events["event_type"].value_counts() if not events.empty else pd.Series(dtype=int)
    state_trace = ordered.get("adapter_state", pd.Series([], dtype=str))
    return {
        "direction_id": direction_id,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "model": model,
        "watch_logit_threshold": watch_threshold,
        "high_logit_threshold": high_threshold,
        "Stage5_AUROC": roc_auc(stage5.astype(int), score),
        "Stage5_AUPRC": average_precision(stage5.astype(int), score),
        "Stage5_Recall_at_high": float(np.mean(high[stage5])) if stage5.any() else float("nan"),
        "Stage4to5_Recall_at_watch": float(np.mean(watch[stage45])) if stage45.any() else float("nan"),
        "Stage1to2_FPR_at_high": float(np.mean(high[early])) if early.any() else float("nan"),
        "Stage1to2_FPR_at_watch": float(np.mean(watch[early])) if early.any() else float("nan"),
        "Recall_at_10pct_Stage1to2_FPR": recall_at_fpr(ordered),
        "Risk_Stage_Spearman": spearman(score, stage),
        "soft_target_brier": soft_brier(ordered),
        "first_WATCH_cycle": first_watch,
        "first_HIGH_cycle": first_high,
        "lead_cycles_relative_to_Stage5": stage5_start - first_high if np.isfinite(stage5_start) and np.isfinite(first_high) else float("nan"),
        "false_HIGH_episodes_per_1000_cycles": high_episodes_per_1000_cycles(ordered, high),
        "watch_occupancy": float(np.mean(watch)),
        "high_occupancy": float(np.mean(high)),
        "update_episode_count": episode_count((state_trace == "ACTIVE_UPDATE").to_numpy()) if len(state_trace) else 0,
        "update_window_count": int(np.sum(state_trace == "ACTIVE_UPDATE")),
        "freeze_episode_count": int(counts.get("ENTER_STATE", 0)),
        "frozen_window_count": int(np.sum(state_trace.isin(["FROZEN_EVENT", "FROZEN_RISK", "COOLDOWN"]))) if len(state_trace) else 0,
        "rollback_count": int(counts.get("ROLLBACK", 0)),
        "ONLINE_ADAPTATION_NOT_EXERCISED": bool(np.sum(state_trace == "ACTIVE_UPDATE") == 0) if len(state_trace) else True,
    }


def add_suppression(summary: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    values = {}
    for direction_id, group in scores.groupby("direction_id"):
        r2 = group[(group["model"] == "R2") & (group["stage"] == 5)]["final_risk"]
        r5 = group[(group["model"] == "R5") & (group["stage"] == 5)]["final_risk"]
        values[direction_id] = float(np.nanmedian(r2) - np.nanmedian(r5)) if len(r2) and len(r5) else float("nan")
    output["Stage5_risk_suppression_R2_minus_R5"] = output["direction_id"].map(values)
    return output


def acceptance(summary: pd.DataFrame, thresholds: pd.DataFrame, guard_audit: pd.DataFrame) -> list[dict[str, object]]:
    r0 = summary[summary["model"] == "R0"].set_index("direction_id")
    r5 = summary[summary["model"] == "R5"].set_index("direction_id")
    shared = sorted(set(r0.index).intersection(r5.index))
    rows: list[dict[str, object]] = []
    if not shared:
        return [{"criterion": "R0 and R5 both present", "observed": "missing", "status": "FAIL"}]
    a, b = r0.loc[shared], r5.loc[shared]
    auroc_delta = float(b["Stage5_AUROC"].min() - a["Stage5_AUROC"].min())
    auprc_delta = float(b["Stage5_AUPRC"].min() - a["Stage5_AUPRC"].min())
    checks = [
        ("R5 Stage5 Recall >= 0.85 in both directions", float(b["Stage5_Recall_at_high"].min()), lambda x: x >= 0.85),
        ("R5 Stage1-2 HIGH FPR <= 0.10 in both directions", float(b["Stage1to2_FPR_at_high"].max()), lambda x: x <= 0.10),
        ("R5 lead relative to Stage5 >= 0 where detected", float(b["lead_cycles_relative_to_Stage5"].min(skipna=True)), lambda x: x >= 0.0),
        ("R5 worst AUROC actual drop <= 0.02", max(0.0, -auroc_delta), lambda x: x <= 0.02),
        ("R5 worst AUPRC actual drop <= 0.03", max(0.0, -auprc_delta), lambda x: x <= 0.03),
        ("Stage5 risk suppression R2 minus R5 <= 0.10", float(b["Stage5_risk_suppression_R2_minus_R5"].max()), lambda x: x <= 0.10),
        ("clean TES events in restart guard == 0", float(guard_audit["clean_TES_events_in_guard"].sum()), lambda x: x == 0.0),
        ("clean freeze triggers in restart guard == 0", float(guard_audit["clean_freeze_triggers_in_guard"].sum()), lambda x: x == 0.0),
        ("all non-intercept coefficients <= 5", float(thresholds["max_abs_nonintercept_beta"].max()), lambda x: x <= 5.0),
        ("high probability threshold < 0.99", float(thresholds["high_probability_equivalent"].max()), lambda x: x < 0.99),
        ("watch logit threshold < high logit threshold", bool((thresholds["watch_logit_threshold"] < thresholds["high_logit_threshold"]).all()), lambda x: bool(x)),
        ("source validation Stage5 Recall >= 0.85", float(thresholds["high_Stage5_recall"].min()), lambda x: x >= 0.85),
        ("source validation Stage1-2 HIGH FPR <= 0.10", float(thresholds["high_Stage1to2_fpr"].max()), lambda x: x <= 0.10),
    ]
    for name, observed, predicate in checks:
        rows.append({"criterion": name, "observed": observed, "status": "PASS" if predicate(observed) else "FAIL"})
    rows.extend(
        [
            {"criterion": "signed_delta_R5_minus_R0_worst_AUROC", "observed": auroc_delta, "status": "INFO"},
            {"criterion": "actual_drop_R0_minus_R5_worst_AUROC", "observed": max(0.0, -auroc_delta), "status": "INFO"},
            {"criterion": "signed_delta_R5_minus_R0_worst_AUPRC", "observed": auprc_delta, "status": "INFO"},
            {"criterion": "actual_drop_R0_minus_R5_worst_AUPRC", "observed": max(0.0, -auprc_delta), "status": "INFO"},
        ]
    )
    return rows


def markdown(frame: pd.DataFrame, columns: Sequence[str] | None = None, limit: int = 60) -> str:
    show = frame.loc[:, list(columns)].head(limit) if columns else frame.head(limit)
    if show.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(map(str, show.columns)) + " |", "| " + " | ".join(["---"] * len(show.columns)) + " |"]
    for row in show.itertuples(index=False, name=None):
        cells = []
        for value in row:
            cells.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
