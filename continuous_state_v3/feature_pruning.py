from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import ContinuousStateV3Config, FEATURES
from .data import assert_label_free


PHYSICAL_DIRECTNESS = {name: number for name, number in {
    "rs_corrdist_base": 3, "rs_mean": 2, "rs_q05": 2, "rx_corrdist_base": 3,
    "rs_rms": 2, "ry_p2p": 2, "rx_mean": 2, "rx_absmean": 2, "rx_q05": 2,
}.items()}


def source_train(frame: pd.DataFrame, config: ContinuousStateV3Config) -> pd.DataFrame:
    assert_label_free(frame)
    ordered = frame.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    stop = int(len(ordered) * config.source_train_fraction)
    return ordered.iloc[:stop].copy()


def _pair_positions(centers: np.ndarray, low: int, high: int | None, cap: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(centers)
    starts = np.maximum(np.searchsorted(centers, centers + low, side="left"), np.arange(n) + 1)
    ends = np.full(n, n, int) if high is None else np.searchsorted(centers, centers + high, side="left")
    counts = np.maximum(ends - starts, 0)
    cumulative = np.cumsum(counts, dtype=np.int64)
    total = int(cumulative[-1]) if len(cumulative) else 0
    if total == 0:
        return np.empty(0, int), np.empty(0, int)
    rank = np.sort(rng.choice(total, min(cap, total), replace=False))
    earlier = np.searchsorted(cumulative, rank, side="right")
    previous = np.where(earlier == 0, 0, cumulative[earlier - 1])
    return earlier.astype(int), (starts[earlier] + rank - previous).astype(int)


def shared_temporal_pairs(frame: pd.DataFrame, config: ContinuousStateV3Config) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """One fixed source-train pair set for every feature's direction-free AUC."""
    assert_label_free(frame)
    usable = frame.loc[frame.is_restart_guard.eq(0)].sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    rng = np.random.default_rng(config.random_seed)
    left, right, bins = [], [], []
    centers = usable.center_cycle.to_numpy(float)
    for bin_number, (low, high) in enumerate(config.pair_gap_bins):
        earlier, later = _pair_positions(centers, low, high, config.max_pairs_per_gap_bin, rng)
        left.extend(earlier); right.extend(later); bins.extend([bin_number] * len(earlier))
    audit = pd.DataFrame({"earlier_row": left, "later_row": right, "gap_bin": bins})
    return np.asarray(left, int), np.asarray(right, int), usable


def direction_free_auc(values: np.ndarray, earlier: np.ndarray, later: np.ndarray) -> float:
    if not len(earlier):
        return float("nan")
    delta = values[later] - values[earlier]
    y = np.r_[np.ones(len(delta), int), np.zeros(len(delta), int)]
    score = np.r_[delta, -delta]
    auc = float(roc_auc_score(y, score))
    return max(auc, 1. - auc)


def prune_features(frame: pd.DataFrame, protocol_id: str, config: ContinuousStateV3Config) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Source-only correlation prune; all candidates use the same temporal pairs."""
    train = source_train(frame, config)
    earlier, later, usable = shared_temporal_pairs(train, config)
    active = [name for name in FEATURES if name != "rs_absmean"]
    aucs = {name: direction_free_auc(usable[name].to_numpy(float), earlier, later) for name in active}
    correlation = usable.loc[:, active].corr().abs()
    rows: list[dict[str, object]] = [{"protocol_id": protocol_id, "feature_name": "rs_absmean", "kept": 0, "drop_reason": "EXACT_DUPLICATE_OF_rs_mean", "correlated_with": "rs_mean", "abs_correlation": 1., "direction_free_auc": np.nan, "pair_set_id": f"{protocol_id}_source_train"}]
    for offset, left in enumerate(list(active)):
        if left not in active:
            continue
        for right in list(active)[offset + 1:]:
            if left not in active or right not in active:
                continue
            corr = float(correlation.loc[left, right])
            if corr < config.correlation_prune_threshold:
                continue
            if abs(aucs[left] - aucs[right]) >= .005:
                keep, drop = (left, right) if aucs[left] > aucs[right] else (right, left)
                rationale = "HIGH_CORRELATION_KEEP_HIGHER_DIRECTION_FREE_AUC"
            elif PHYSICAL_DIRECTNESS[left] != PHYSICAL_DIRECTNESS[right]:
                keep, drop = (left, right) if PHYSICAL_DIRECTNESS[left] > PHYSICAL_DIRECTNESS[right] else (right, left)
                rationale = "HIGH_CORRELATION_KEEP_MORE_DIRECT_PHYSICAL_FEATURE"
            else:
                keep, drop = sorted((left, right))
                rationale = "HIGH_CORRELATION_ALPHABETICAL_TIEBREAK"
            active.remove(drop)
            rows.append({"protocol_id": protocol_id, "feature_name": drop, "kept": 0, "drop_reason": rationale, "correlated_with": keep, "abs_correlation": corr, "direction_free_auc": aucs[drop], "pair_set_id": f"{protocol_id}_source_train"})
    for feature in active:
        rows.append({"protocol_id": protocol_id, "feature_name": feature, "kept": 1, "drop_reason": "KEPT", "correlated_with": "", "abs_correlation": 0., "direction_free_auc": aucs[feature], "pair_set_id": f"{protocol_id}_source_train"})
    return tuple(active), pd.DataFrame(rows).sort_values(["kept", "feature_name"], ascending=[False, True]).reset_index(drop=True)
