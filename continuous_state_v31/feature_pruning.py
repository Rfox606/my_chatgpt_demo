from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import ContinuousStateV31Config, FEATURES
from .data import assert_label_free


PHYSICAL_DIRECTNESS = {"rs_corrdist_base": 3, "rs_mean": 2, "rs_q05": 2, "rx_corrdist_base": 3,
                       "rs_rms": 2, "ry_p2p": 2, "rx_mean": 2, "rx_absmean": 2, "rx_q05": 2}


def _pair_positions(centers: np.ndarray, low: int, high: int | None, cap: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(centers)
    starts = np.maximum(np.searchsorted(centers, centers + low, side="left"), np.arange(n) + 1)
    ends = np.full(n, n, int) if high is None else np.searchsorted(centers, centers + high, side="left")
    counts = np.maximum(ends - starts, 0); cumulative = np.cumsum(counts, dtype=np.int64)
    total = int(cumulative[-1]) if len(cumulative) else 0
    if total == 0:
        return np.empty(0, int), np.empty(0, int)
    rank = np.sort(rng.choice(total, min(cap, total), replace=False))
    earlier = np.searchsorted(cumulative, rank, side="right")
    previous = np.where(earlier == 0, 0, cumulative[earlier - 1])
    return earlier.astype(int), (starts[earlier] + rank - previous).astype(int)


def _direction_free_auc(values: np.ndarray, earlier: np.ndarray, later: np.ndarray) -> float:
    if not len(earlier):
        return float("nan")
    delta = values[later] - values[earlier]
    auc = float(roc_auc_score(np.r_[np.ones(len(delta), int), np.zeros(len(delta), int)], np.r_[delta, -delta]))
    return max(auc, 1. - auc)


def prune_features(frame: pd.DataFrame, protocol_id: str, config: ContinuousStateV31Config) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Source-train-only pruning, with exactly one temporal pair set for every feature."""
    assert_label_free(frame)
    ordered = frame.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    train = ordered.iloc[:int(len(ordered) * config.source_train_fraction)]
    usable = train.loc[train.is_restart_guard.eq(0)].reset_index(drop=True)
    centers = usable.center_cycle.to_numpy(float); rng = np.random.default_rng(config.random_seed)
    left_all, right_all = [], []
    for low, high in config.pair_gap_bins:
        left, right = _pair_positions(centers, low, high, config.max_pairs_per_gap_bin, rng)
        left_all.extend(left); right_all.extend(right)
    left, right = np.asarray(left_all, int), np.asarray(right_all, int)
    active = [feature for feature in FEATURES if feature != "rs_absmean"]
    auc = {feature: _direction_free_auc(usable[feature].to_numpy(float), left, right) for feature in active}
    corr = usable.loc[:, active].corr().abs()
    rows = [{"protocol_id": protocol_id, "feature_name": "rs_absmean", "kept": 0,
             "drop_reason": "EXACT_DUPLICATE_OF_rs_mean", "correlated_with": "rs_mean", "abs_correlation": 1.,
             "direction_free_auc": np.nan, "pair_set_id": f"{protocol_id}_source_train"}]
    for i, left_name in enumerate(list(active)):
        if left_name not in active:
            continue
        for right_name in list(active)[i + 1:]:
            if left_name not in active or right_name not in active:
                continue
            value = float(corr.loc[left_name, right_name])
            if value < config.correlation_prune_threshold:
                continue
            if abs(auc[left_name] - auc[right_name]) >= .005:
                keep, drop, reason = ((left_name, right_name, "HIGH_CORRELATION_KEEP_HIGHER_DIRECTION_FREE_AUC")
                                      if auc[left_name] > auc[right_name] else (right_name, left_name, "HIGH_CORRELATION_KEEP_HIGHER_DIRECTION_FREE_AUC"))
            elif PHYSICAL_DIRECTNESS[left_name] != PHYSICAL_DIRECTNESS[right_name]:
                keep, drop = ((left_name, right_name) if PHYSICAL_DIRECTNESS[left_name] > PHYSICAL_DIRECTNESS[right_name] else (right_name, left_name))
                reason = "HIGH_CORRELATION_KEEP_MORE_DIRECT_PHYSICAL_FEATURE"
            else:
                keep, drop = sorted((left_name, right_name)); reason = "HIGH_CORRELATION_ALPHABETICAL_TIEBREAK"
            active.remove(drop)
            rows.append({"protocol_id": protocol_id, "feature_name": drop, "kept": 0, "drop_reason": reason,
                         "correlated_with": keep, "abs_correlation": value, "direction_free_auc": auc[drop],
                         "pair_set_id": f"{protocol_id}_source_train"})
    rows.extend({"protocol_id": protocol_id, "feature_name": feature, "kept": 1, "drop_reason": "KEPT",
                 "correlated_with": "", "abs_correlation": 0., "direction_free_auc": auc[feature],
                 "pair_set_id": f"{protocol_id}_source_train"} for feature in active)
    return tuple(active), pd.DataFrame(rows).sort_values(["kept", "feature_name"], ascending=[False, True]).reset_index(drop=True)
