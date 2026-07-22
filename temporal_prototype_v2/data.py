from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import TemporalPrototypeConfig


LABEL_COLUMNS = {"stage", "stage_label", "Stage1to5"}


def reject_target_labels(frame: pd.DataFrame) -> None:
    leaked = LABEL_COLUMNS.intersection(frame.columns)
    if leaked:
        raise AssertionError(f"Target label leakage is prohibited: {sorted(leaked)}")


def load_window_table(config: TemporalPrototypeConfig) -> pd.DataFrame:
    long = pd.read_csv(config.z_table_path)
    ids = ["dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle", "stage", "stage_label", "baseline_window"]
    wide = long.pivot_table(index=ids, columns="feature_name", values="z_value", aggfunc="first").reset_index()
    wide.columns.name = None
    state = pd.read_csv(config.state_path)
    # The legacy RS20/50/100 columns are intentionally retained by the upstream file but
    # are empty in v2.  The RS_trend*_v2 values are its causal replacements.
    state = state[["dataset", "window_index", "BDall_xy_v2", "BDshape_v2", "RS_trend20_v2", "RS_trend50_v2", "RS_trend100_v2", "TES_v2"]].copy()
    state = state.rename(columns={"RS_trend20_v2": "RS20", "RS_trend50_v2": "RS50", "RS_trend100_v2": "RS100", "TES_v2": "TES"})
    frame = wide.merge(state, on=["dataset", "window_index"], how="left", validate="one_to_one")
    rows: list[pd.DataFrame] = []
    for _, part in frame.groupby("dataset", sort=True):
        part = part.sort_values("window_index").reset_index(drop=True).copy()
        cycles = part["center_cycle"].to_numpy(float)
        period = np.floor((cycles - 1) / config.known_stop_interval_cycles).astype(int)
        restart = np.r_[False, period[1:] != period[:-1]]
        boundary = period * config.known_stop_interval_cycles
        guard = np.abs(cycles - boundary) <= config.restart_guard_cycles
        part["restart_mask"] = restart.astype(int)
        part["is_restart_guard"] = guard.astype(int)
        part["BD_jump"] = np.r_[0.0, np.abs(np.diff(part["BDall_xy_v2"].to_numpy(float)))]
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def stagewise_source_split(frame: pd.DataFrame, config: TemporalPrototypeConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.zeros(len(frame), dtype=bool)
    gap = np.zeros(len(frame), dtype=bool)
    validation = np.zeros(len(frame), dtype=bool)
    for stage in range(1, 6):
        positions = np.flatnonzero(frame["stage"].to_numpy(int) == stage)
        cutoff = int(np.floor(len(positions) * config.source_train_fraction))
        train[positions[:cutoff]] = True
        gap[positions[cutoff:cutoff + config.source_gap_windows]] = True
        validation[positions[cutoff + config.source_gap_windows:]] = True
    return train, gap, validation


@dataclass(frozen=True)
class SourceScaler:
    features: tuple[str, ...]
    median: np.ndarray
    iqr: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def fit(cls, train: pd.DataFrame, features: Iterable[str], config: TemporalPrototypeConfig) -> "SourceScaler":
        names = tuple(features)
        x = train.loc[:, names].to_numpy(float)
        median = np.nanmedian(x, axis=0)
        q25, q75 = np.nanpercentile(x, [25, 75], axis=0)
        iqr = np.maximum(q75 - q25, 1e-6)
        lower, upper = np.nanpercentile(x, [1, 99], axis=0)
        return cls(names, median, iqr, lower, upper)

    def transform(self, frame: pd.DataFrame, config: TemporalPrototypeConfig) -> tuple[np.ndarray, np.ndarray]:
        x = frame.loc[:, self.features].to_numpy(float)
        missing = np.mean(~np.isfinite(x), axis=1)
        x = np.where(np.isfinite(x), x, self.median)
        x = np.clip(x, self.lower, self.upper)
        z = np.clip((x - self.median) / self.iqr, -config.clip_abs_z, config.clip_abs_z)
        return z.astype(np.float32), missing.astype(np.float32)


def causal_sequences(values: np.ndarray, restart_mask: np.ndarray, length: int) -> np.ndarray:
    """Build left-padded prefixes; a restart starts an independent GRU history."""
    result = np.empty((len(values), length, values.shape[1]), dtype=np.float32)
    segment_start = 0
    for i in range(len(values)):
        if restart_mask[i]:
            segment_start = i
        start = max(segment_start, i - length + 1)
        prefix = values[start:i + 1]
        if len(prefix) < length:
            prefix = np.vstack([np.repeat(prefix[:1], length - len(prefix), axis=0), prefix])
        result[i] = prefix
    return result


def unlabeled_target(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(columns=["stage", "stage_label", "baseline_window"], errors="ignore").copy()
    reject_target_labels(result)
    return result
