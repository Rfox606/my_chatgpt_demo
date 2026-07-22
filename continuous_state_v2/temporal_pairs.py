from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ContinuousStateV2Config
from .data import assert_label_free


@dataclass(frozen=True)
class PairBatch:
    pairs: pd.DataFrame
    delta_x: np.ndarray
    labels: np.ndarray
    features: tuple[str, ...]

    @property
    def pair_count(self) -> int:
        return len(self.pairs)


def split_source(frame: pd.DataFrame, config: ContinuousStateV2Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert_label_free(frame)
    ordered = frame.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    split = int(len(ordered) * config.source_train_fraction)
    gap_end = min(len(ordered), split + config.source_gap_windows)
    return ordered.iloc[:split].copy(), ordered.iloc[gap_end:].copy(), ordered.iloc[split:gap_end].copy()


def gap_name(low: int, high: int | None) -> str:
    return f"gap_{low}_{high}" if high is not None else f"gap_{low}_plus"


def _positions(centers: np.ndarray, low: int, high: int | None, cap: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(centers)
    starts = np.maximum(np.searchsorted(centers, centers + low, side="left"), np.arange(n) + 1)
    ends = np.full(n, n, int) if high is None else np.searchsorted(centers, centers + high, side="left")
    counts = np.maximum(ends - starts, 0)
    cumulative = np.cumsum(counts, dtype=np.int64)
    total = int(cumulative[-1]) if len(cumulative) else 0
    if not total:
        return np.array([], int), np.array([], int)
    ranks = np.sort(rng.choice(total, min(cap, total), replace=False))
    earlier = np.searchsorted(cumulative, ranks, side="right")
    prior = np.where(earlier == 0, 0, cumulative[earlier - 1])
    return earlier.astype(int), (starts[earlier] + ranks - prior).astype(int)


def sample_pairs(frame: pd.DataFrame, config: ContinuousStateV2Config, seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_label_free(frame)
    usable = frame.loc[frame.is_restart_guard.eq(0)].sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    centers = usable.center_cycle.to_numpy(float)
    rng = np.random.default_rng(config.pair_random_seed if seed is None else seed)
    pieces: list[pd.DataFrame] = []
    for number, (low, high) in enumerate(config.pair_gap_bins):
        early, late = _positions(centers, low, high, config.max_pairs_per_gap_bin, rng)
        if len(early):
            pieces.append(pd.DataFrame({
                "earlier_row": early, "later_row": late,
                "earlier_window_id": usable.iloc[early].window_id.to_numpy(),
                "later_window_id": usable.iloc[late].window_id.to_numpy(),
                "earlier_window_index": usable.iloc[early].window_index.to_numpy(),
                "later_window_index": usable.iloc[late].window_index.to_numpy(),
                "earlier_guard": 0, "later_guard": 0,
                "cycle_gap": centers[late] - centers[early], "gap_bin": gap_name(low, high), "gap_bin_number": number,
            }))
    columns = ["earlier_row", "later_row", "earlier_window_id", "later_window_id", "earlier_window_index", "later_window_index", "earlier_guard", "later_guard", "cycle_gap", "gap_bin", "gap_bin_number"]
    return (pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)), usable


def build_pair_batch(frame: pd.DataFrame, features: tuple[str, ...], config: ContinuousStateV2Config, seed: int | None = None) -> PairBatch:
    assert_label_free(frame)
    pairs, usable = sample_pairs(frame, config, seed)
    if pairs.empty:
        return PairBatch(pairs, np.empty((0, len(features))), np.empty(0, int), features)
    values = usable.loc[:, list(features)].to_numpy(float)
    positive = values[pairs.later_row.to_numpy()] - values[pairs.earlier_row.to_numpy()]
    return PairBatch(pairs, np.vstack([positive, -positive]), np.r_[np.ones(len(positive), int), np.zeros(len(positive), int)], features)
