from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GradualStateMonitoringV34Config
from .data import effective_to_actual


def events_with_actual_cycles(events: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    if result.empty:
        result["peak_actual_cycle"] = pd.Series(dtype=float)
        return result
    result["start_actual_cycle"] = np.nan; result["peak_actual_cycle"] = np.nan; result["end_actual_cycle"] = np.nan
    for dataset, indices in result.groupby("dataset", sort=False).groups.items():
        positions = list(indices)
        for source, target in (("start_cycle", "start_actual_cycle"), ("peak_cycle", "peak_actual_cycle"), ("end_cycle", "end_actual_cycle")):
            result.loc[positions, target] = effective_to_actual(result.loc[positions, source].to_numpy(float), str(dataset), segments)
    return result


def match_candidate_events(events: pd.DataFrame, boundaries: pd.DataFrame, tolerance: float,
                           *, direction: str, setting: str, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """Match one event per boundary; every additional alert remains in precision's denominator."""
    local_events = events[(events.setting == setting)].copy()
    local_boundaries = boundaries.copy()
    rows: list[dict[str, object]] = []
    used: set[int] = set()
    for _, boundary in local_boundaries.sort_values(["dataset", "boundary_actual_cycle"], kind="stable").iterrows():
        candidates = local_events[(local_events.dataset == boundary.dataset) &
                                  (np.abs(local_events.peak_actual_cycle - float(boundary.boundary_actual_cycle)) <= tolerance)]
        candidates = candidates.loc[[idx for idx in candidates.index if int(idx) not in used]]
        if len(candidates):
            candidates = candidates.assign(distance=np.abs(candidates.peak_actual_cycle - float(boundary.boundary_actual_cycle)))
            chosen_index = int(candidates.sort_values(["distance", "peak_score"], ascending=[True, False], kind="stable").index[0])
            used.add(chosen_index)
            event = local_events.loc[chosen_index]
            rows.append(_match_row(direction, setting, seed, tolerance, boundary, event, "MATCHED"))
        else:
            rows.append(_match_row(direction, setting, seed, tolerance, boundary, None, "MISSED"))
    for index, event in local_events.iterrows():
        if int(index) not in used:
            rows.append({
                "direction": direction, "setting": setting, "seed": seed, "tolerance_actual_cycles": tolerance,
                "dataset": str(event.dataset), "from_stage": np.nan, "to_stage": np.nan,
                "boundary_cycle": np.nan, "boundary_actual_cycle": np.nan,
                "event_id": int(event.event_id), "event_peak_cycle": float(event.peak_cycle),
                "event_peak_actual_cycle": float(event.peak_actual_cycle), "peak_score": float(event.peak_score),
                "absolute_detection_delay": np.nan, "match_status": "unmatched_candidate_event",
            })
    log = pd.DataFrame(rows)
    matched = int((log.match_status == "MATCHED").sum()) if len(log) else 0
    total_events = int(len(local_events))
    truth = int(len(local_boundaries))
    precision = float(matched / total_events) if total_events else 0.0
    recall = float(matched / truth) if truth else np.nan
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    delays = log.loc[log.match_status == "MATCHED", "absolute_detection_delay"].to_numpy(float) if len(log) else np.empty(0)
    metrics = {
        "direction": direction, "setting": setting, "seed": seed, "tolerance_actual_cycles": tolerance,
        "boundary_hit_rate": recall, "mean_absolute_detection_delay": float(np.mean(delays)) if len(delays) else np.nan,
        "median_absolute_detection_delay": float(np.median(delays)) if len(delays) else np.nan,
        "event_level_precision": precision, "event_level_recall": recall, "event_level_f1": f1,
        "candidate_event_count": total_events, "unmatched_candidate_count": int((log.match_status == "unmatched_candidate_event").sum()),
    }
    return log, metrics


def _match_row(direction: str, setting: str, seed: int, tolerance: float, boundary: pd.Series,
               event: pd.Series | None, status: str) -> dict[str, object]:
    peak = float(event.peak_actual_cycle) if event is not None else np.nan
    value = {
        "direction": direction, "setting": setting, "seed": seed, "tolerance_actual_cycles": tolerance,
        "dataset": str(boundary.dataset), "from_stage": int(boundary.from_stage), "to_stage": int(boundary.to_stage),
        "boundary_cycle": float(boundary.boundary_cycle), "boundary_actual_cycle": float(boundary.boundary_actual_cycle),
        "event_id": int(event.event_id) if event is not None else np.nan,
        "event_peak_cycle": float(event.peak_cycle) if event is not None else np.nan,
        "event_peak_actual_cycle": peak, "peak_score": float(event.peak_score) if event is not None else np.nan,
        "absolute_detection_delay": abs(peak - float(boundary.boundary_actual_cycle)) if event is not None else np.nan,
        "match_status": status,
    }
    return value


def boundary_ranking(scores: pd.DataFrame, events: pd.DataFrame, boundaries: pd.DataFrame,
                     *, direction: str, setting: str, seed: int, config: GradualStateMonitoringV34Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank events without ever using a target Stage value to form their score."""
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    online = scores[(scores.setting == setting) & (scores.phase == "ONLINE")].copy()
    local_events = events[events.setting == setting].copy()
    for dataset, local_boundaries in boundaries.groupby("dataset", sort=True):
        series = online[online.dataset == dataset]
        candidates = local_events[local_events.dataset == dataset].sort_values("peak_score", ascending=False, kind="stable").reset_index(drop=True)
        hit_ranks: list[float] = []
        for _, boundary in local_boundaries.iterrows():
            distance = np.abs(series.actual_cycle.to_numpy(float) - float(boundary.boundary_actual_cycle))
            near = series.iloc[np.flatnonzero(distance <= config.proximity_limit_actual_cycles)]
            if len(near):
                peak_score = float(near.final_transition_score.max())
                percentile = float(np.mean(series.final_transition_score.to_numpy(float) <= peak_score))
            else:
                peak_score = np.nan; percentile = np.nan
            event_distance = np.abs(candidates.peak_actual_cycle.to_numpy(float) - float(boundary.boundary_actual_cycle)) if len(candidates) else np.empty(0)
            positions = np.flatnonzero(event_distance <= config.proximity_limit_actual_cycles)
            rank = int(positions[0] + 1) if len(positions) else np.nan
            if np.isfinite(rank):
                hit_ranks.append(float(rank))
            rows.append({
                "direction": direction, "setting": setting, "seed": seed, "dataset": str(dataset),
                "from_stage": int(boundary.from_stage), "to_stage": int(boundary.to_stage),
                "boundary_actual_cycle": float(boundary.boundary_actual_cycle), "boundary_peak_score": peak_score,
                "boundary_score_percentile": percentile, "boundary_event_rank": rank,
            })
        count = max(len(local_boundaries), 1)
        for top_k in (4, 8, 12):
            summary.append({
                "direction": direction, "setting": setting, "seed": seed, "dataset": str(dataset),
                "metric": f"Top-{top_k} event recall", "value": float(np.sum(np.asarray(hit_ranks) <= top_k) / count),
            })
    return pd.DataFrame(rows), pd.DataFrame(summary)


def proximity_weighted_score(events: pd.DataFrame, boundaries: pd.DataFrame,
                             *, direction: str, setting: str, seed: int,
                             config: GradualStateMonitoringV34Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    local = events[events.setting == setting]
    for _, boundary in boundaries.iterrows():
        candidates = local[local.dataset == boundary.dataset]
        if len(candidates):
            distance = np.abs(candidates.peak_actual_cycle.to_numpy(float) - float(boundary.boundary_actual_cycle))
            value = candidates.peak_score.to_numpy(float) * np.maximum(0.0, 1.0 - distance / config.proximity_limit_actual_cycles)
            best = float(np.max(value)) if len(value) else 0.0
        else:
            best = 0.0
        rows.append({
            "direction": direction, "setting": setting, "seed": seed, "dataset": str(boundary.dataset),
            "boundary_actual_cycle": float(boundary.boundary_actual_cycle), "proximity_weighted_score": best,
        })
    return pd.DataFrame(rows)


def adaptation_metrics(scores: pd.DataFrame, adaptation: pd.DataFrame, *, direction: str, setting: str, seed: int) -> dict[str, object]:
    online = scores[scores.phase == "ONLINE"]
    updates = adaptation.adapter_updated.to_numpy(float) if len(adaptation) else np.empty(0)
    frozen = adaptation.adapter_frozen.to_numpy(float) if len(adaptation) else np.empty(0)
    return {
        "direction": direction, "setting": setting, "seed": seed,
        "candidate_event_count": int(online.candidate_active.max()) if len(online) else 0,
        "transition_score_above_threshold_fraction": float(np.mean(online.transition_score > online.transition_threshold)) if len(online) else np.nan,
        "adapter_update_fraction": float(np.mean(updates)) if len(updates) else 0.0,
        "adapter_frozen_fraction": float(np.mean(frozen)) if len(frozen) else 0.0,
        "adapter_update_count": int(np.sum(updates)), "adapter_frozen_count": int(np.sum(frozen)),
        "stage_internal_event_count": np.nan,  # filled in offline once Stage is attached to event peaks
    }
