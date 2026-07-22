from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OrderedRegimeConfig


FORBIDDEN_LABELS = {"stage", "stage_label", "Stage1to5"}


def reject_target_labels(frame: pd.DataFrame) -> None:
    leaked = FORBIDDEN_LABELS.intersection(frame.columns)
    if leaked:
        raise AssertionError(f"Target labels are forbidden online: {sorted(leaked)}")


def load_table(config: OrderedRegimeConfig) -> pd.DataFrame:
    long = pd.read_csv(config.z_table_path)
    ids = ["dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle", "stage", "stage_label", "baseline_window"]
    wide = long.pivot_table(index=ids, columns="feature_name", values="z_value", aggfunc="first").reset_index()
    wide.columns.name = None
    state = pd.read_csv(config.state_path, usecols=["dataset", "window_index", "BDall_xy_v2", "BDshape_v2", "RS_trend20_v2", "RS_trend50_v2", "RS_trend100_v2", "TES_v2"])
    state = state.rename(columns={"RS_trend20_v2": "RS20", "RS_trend50_v2": "RS50", "RS_trend100_v2": "RS100", "TES_v2": "TES"})
    frame = wide.merge(state, on=["dataset", "window_index"], how="left", validate="one_to_one")
    pieces = []
    for _, item in frame.groupby("dataset", sort=True):
        item = item.sort_values("window_index").reset_index(drop=True).copy()
        period = np.floor((item.center_cycle.to_numpy(float) - 1) / config.known_stop_interval_cycles).astype(int)
        item["restart_mask"] = np.r_[False, period[1:] != period[:-1]].astype(int)
        boundary = period * config.known_stop_interval_cycles
        item["is_restart_guard"] = (np.abs(item.center_cycle.to_numpy(float) - boundary) <= config.restart_guard_cycles).astype(int)
        item["BD_jump"] = np.r_[0.0, np.abs(np.diff(item.BDall_xy_v2.to_numpy(float)))]
        pieces.append(item)
    return pd.concat(pieces, ignore_index=True)


def source_split(frame: pd.DataFrame, config: OrderedRegimeConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.zeros(len(frame), dtype=bool); gap = train.copy(); validation = train.copy()
    for stage in range(1, 6):
        positions = np.flatnonzero(frame.stage.to_numpy(int) == stage)
        cutoff = int(np.floor(len(positions) * config.source_train_fraction))
        train[positions[:cutoff]] = True
        gap[positions[cutoff:cutoff + config.source_gap_windows]] = True
        validation[positions[cutoff + config.source_gap_windows:]] = True
    return train, gap, validation


def causal_sequences(values: np.ndarray, restart: np.ndarray, length: int) -> np.ndarray:
    output = np.empty((len(values), length, values.shape[1]), dtype=np.float32)
    start_segment = 0
    for index in range(len(values)):
        if restart[index]:
            start_segment = index
        start = max(start_segment, index - length + 1)
        prefix = values[start:index + 1]
        if len(prefix) < length:
            prefix = np.vstack([np.repeat(prefix[:1], length - len(prefix), axis=0), prefix])
        output[index] = prefix
    return output


def target_unlabeled(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(columns=["stage", "stage_label", "baseline_window"], errors="ignore").copy()
    reject_target_labels(result)
    return result
