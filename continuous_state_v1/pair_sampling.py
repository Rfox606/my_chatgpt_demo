from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ContinuousStateV1Config
from .data import assert_label_free


@dataclass(frozen=True)
class PairBatch:
    positive_pairs: pd.DataFrame
    delta_x: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]

    @property
    def pair_count(self) -> int:
        return len(self.positive_pairs)


def split_source_windows(
    frame: pd.DataFrame, config: ContinuousStateV1Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split source windows, leaving an explicit embargo gap."""
    assert_label_free(frame)
    ordered = frame.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    split_at = int(np.floor(len(ordered) * config.source_train_fraction))
    gap_end = min(len(ordered), split_at + config.source_gap_windows)
    train = ordered.iloc[:split_at].copy()
    gap = ordered.iloc[split_at:gap_end].copy()
    validation = ordered.iloc[gap_end:].copy()
    if train.empty or validation.empty:
        raise ValueError("Source time split produced an empty train or validation partition")
    return train, validation, gap


def gap_bin_name(low: int, high: int | None) -> str:
    return f"gap_{low}_{high}" if high is not None else f"gap_{low}_plus"


def _sample_pair_positions(
    centers: np.ndarray, low: int, high: int | None, limit: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample pair ranks without materialising the quadratic pair matrix."""
    n = len(centers)
    starts = np.searchsorted(centers, centers + low, side="left")
    ends = (
        np.full(n, n, dtype=int)
        if high is None
        else np.searchsorted(centers, centers + high, side="left")
    )
    starts = np.maximum(starts, np.arange(n) + 1)
    counts = np.maximum(ends - starts, 0)
    cumulative = np.cumsum(counts, dtype=np.int64)
    total = int(cumulative[-1]) if len(cumulative) else 0
    if total == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    take = min(int(limit), total)
    pair_ranks = np.sort(rng.choice(total, size=take, replace=False))
    earlier = np.searchsorted(cumulative, pair_ranks, side="right")
    prior = np.where(earlier == 0, 0, cumulative[earlier - 1])
    later = starts[earlier] + (pair_ranks - prior)
    return earlier.astype(int), later.astype(int)


def sample_temporal_pairs(
    frame: pd.DataFrame, config: ContinuousStateV1Config, random_seed: int | None = None
) -> pd.DataFrame:
    """Sample valid earlier/later source or diagnostic pairs from non-guard windows."""
    assert_label_free(frame)
    required = {"window_id", "window_index", "center_cycle", "is_restart_guard"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Pair sampling needs columns: {sorted(missing)}")
    usable = frame.loc[frame["is_restart_guard"].astype(int) == 0].copy()
    usable = usable.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    centers = usable["center_cycle"].to_numpy(dtype=float)
    rng = np.random.default_rng(config.pair_random_seed if random_seed is None else random_seed)
    records: list[pd.DataFrame] = []
    for bin_number, (low, high) in enumerate(config.pair_gap_bins):
        earlier, later = _sample_pair_positions(
            centers, low, high, config.max_pairs_per_gap_bin, rng
        )
        if not len(earlier):
            continue
        records.append(
            pd.DataFrame(
                {
                    "earlier_row": earlier,
                    "later_row": later,
                    "earlier_window_id": usable.iloc[earlier]["window_id"].to_numpy(),
                    "later_window_id": usable.iloc[later]["window_id"].to_numpy(),
                    "earlier_window_index": usable.iloc[earlier]["window_index"].to_numpy(),
                    "later_window_index": usable.iloc[later]["window_index"].to_numpy(),
                    "earlier_is_restart_guard": np.zeros(len(earlier), dtype=int),
                    "later_is_restart_guard": np.zeros(len(later), dtype=int),
                    "earlier_center_cycle": centers[earlier],
                    "later_center_cycle": centers[later],
                    "cycle_gap": centers[later] - centers[earlier],
                    "gap_bin": gap_bin_name(low, high),
                    "gap_bin_number": bin_number,
                }
            )
        )
    if not records:
        return pd.DataFrame(
            columns=(
                "earlier_row", "later_row", "earlier_window_id", "later_window_id",
                "earlier_window_index", "later_window_index", "earlier_center_cycle",
                "later_center_cycle", "earlier_is_restart_guard", "later_is_restart_guard",
                "cycle_gap", "gap_bin", "gap_bin_number",
            )
        )
    return pd.concat(records, ignore_index=True)


def build_pair_batch(
    frame: pd.DataFrame,
    config: ContinuousStateV1Config,
    random_seed: int | None = None,
) -> PairBatch:
    """Create balanced mirrored pair differences for the linear rank head."""
    assert_label_free(frame)
    pairs = sample_temporal_pairs(frame, config, random_seed=random_seed)
    usable = (
        frame.loc[frame["is_restart_guard"].astype(int) == 0]
        .sort_values(["center_cycle", "window_index"])
        .reset_index(drop=True)
    )
    features = tuple(config.stable_plus_features)
    if pairs.empty:
        return PairBatch(pairs, np.empty((0, len(features))), np.empty(0, dtype=int), features)
    values = usable.loc[:, list(features)].to_numpy(dtype=float)
    positive = values[pairs["later_row"].to_numpy()] - values[pairs["earlier_row"].to_numpy()]
    delta_x = np.vstack([positive, -positive])
    labels = np.concatenate([np.ones(len(positive), dtype=int), np.zeros(len(positive), dtype=int)])
    return PairBatch(pairs, delta_x, labels, features)


def pair_split_check(
    train: PairBatch,
    validation: PairBatch,
    config: ContinuousStateV1Config,
    train_window_ids: set[object] | None = None,
    validation_window_ids: set[object] | None = None,
) -> dict[str, object]:
    """Return a serialisable assertion audit for the chronological pair protocol."""
    train_ids = set(train.positive_pairs.get("earlier_window_id", [])) | set(
        train.positive_pairs.get("later_window_id", [])
    )
    validation_ids = set(validation.positive_pairs.get("earlier_window_id", [])) | set(
        validation.positive_pairs.get("later_window_id", [])
    )
    checks: dict[str, bool] = {
        "train_validation_endpoint_disjoint": not bool(train_ids.intersection(validation_ids)),
        "train_has_pairs": train.pair_count > 0,
        "validation_has_pairs": validation.pair_count > 0,
    }
    if train_window_ids is not None:
        checks["train_endpoints_in_train_partition"] = train_ids.issubset(train_window_ids)
    if validation_window_ids is not None:
        checks["validation_endpoints_in_validation_partition"] = validation_ids.issubset(validation_window_ids)
    for name, batch in (("train", train), ("validation", validation)):
        p = batch.positive_pairs
        valid = np.ones(len(p), dtype=bool)
        for low, high in config.pair_gap_bins:
            mask = p["gap_bin"].eq(gap_bin_name(low, high)).to_numpy() if len(p) else np.array([], dtype=bool)
            if high is None:
                valid[mask] &= p.loc[mask, "cycle_gap"].to_numpy(float) >= low
            else:
                gaps = p.loc[mask, "cycle_gap"].to_numpy(float)
                valid[mask] &= (gaps >= low) & (gaps < high)
        checks[f"{name}_gaps_valid"] = bool(valid.all())
        checks[f"{name}_endpoints_non_guard"] = bool(
            (p["earlier_is_restart_guard"].to_numpy(int) == 0).all()
            and (p["later_is_restart_guard"].to_numpy(int) == 0).all()
        )
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
