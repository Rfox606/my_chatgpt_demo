from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURE_CONFIGS, CrossExperimentAdaptiveConfig


FORBIDDEN_EXACT = frozenset({"stage", "stage1to5", "sa", "sq", "sz", "sku", "morphology", "wear_debris_count", "debris_count"})
REQUIRED_COLUMNS = frozenset({"dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle"})


def assert_formal_frame(frame: pd.DataFrame) -> None:
    """Reject labels and post-hoc measurements at every formal-model boundary."""
    normalised = {str(column).strip().lower() for column in frame.columns}
    leaked = sorted(FORBIDDEN_EXACT.intersection(normalised))
    if leaked:
        raise AssertionError(f"Forbidden label, morphology, or debris fields reached CEAP v1: {leaked}")


def required_features(config_name: str) -> tuple[str, ...]:
    try:
        return FEATURE_CONFIGS[config_name]
    except KeyError as exc:
        raise ValueError(f"Unknown predeclared feature configuration: {config_name}") from exc


def load_windows(config: CrossExperimentAdaptiveConfig) -> pd.DataFrame:
    frame = pd.read_csv(config.input_path)
    assert_formal_frame(frame)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Window table is missing required columns: {sorted(missing)}")
    for name in config.feature_configs:
        missing_features = set(required_features(name)).difference(frame.columns)
        if missing_features:
            raise ValueError(f"{name} has unavailable fixed features: {sorted(missing_features)}")
    frame = frame.loc[:, list(REQUIRED_COLUMNS | set().union(*(set(required_features(name)) for name in config.feature_configs)))].copy()
    for column in ["start_cycle", "end_cycle", "center_cycle", *required_features(config.primary_feature_config)]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame.loc[:, [*required_features(config.primary_feature_config), "center_cycle"]].to_numpy(float)).all():
        raise ValueError("Formal CEAP input requires finite direct force features and cycle index")
    return frame.sort_values(["dataset", "center_cycle", "window_index"]).reset_index(drop=True)


def source_train_validation_split(frame: pd.DataFrame, config: CrossExperimentAdaptiveConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological source split with a time embargo between train and validation."""
    ordered = frame.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    pivot = float(np.quantile(ordered.center_cycle.to_numpy(float), config.source_train_fraction))
    train = ordered.loc[ordered.center_cycle < pivot - config.source_embargo_cycles].copy()
    validation = ordered.loc[ordered.center_cycle > pivot + config.source_embargo_cycles].copy()
    embargo = ordered.loc[(ordered.center_cycle >= pivot - config.source_embargo_cycles) & (ordered.center_cycle <= pivot + config.source_embargo_cycles)].copy()
    if len(train) < 20 or len(validation) < 20:
        raise ValueError("Source chronological split leaves too few training or validation windows")
    return train, validation, embargo


def stable_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TemporalPairs:
    earlier: np.ndarray
    later: np.ndarray
    gaps: np.ndarray

    @property
    def count(self) -> int:
        return int(len(self.earlier))


def temporal_pairs(
    frame: pd.DataFrame,
    gap_bins: tuple[tuple[float, float], ...],
    maximum_per_bin: int,
    *,
    seed: int,
) -> TemporalPairs:
    """Create only source/target historical time-order pairs; cycle is never a feature."""
    ordered = frame.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    cycles = ordered.center_cycle.to_numpy(float)
    rng = np.random.default_rng(seed)
    earlier_parts: list[np.ndarray] = []; later_parts: list[np.ndarray] = []; gap_parts: list[np.ndarray] = []
    for lower, upper in gap_bins:
        left: list[int] = []; right: list[int] = []
        for later in range(len(ordered)):
            first = int(np.searchsorted(cycles, cycles[later] - upper, side="left"))
            last = int(np.searchsorted(cycles, cycles[later] - lower, side="right"))
            if last <= first:
                continue
            candidates = np.arange(first, last, dtype=int)
            # One sampled earlier window per later point avoids dense-window multiplicity bias.
            left.append(int(rng.choice(candidates))); right.append(later)
        if not left:
            continue
        chosen = np.arange(len(left))
        if len(chosen) > maximum_per_bin:
            chosen = np.sort(rng.choice(chosen, size=maximum_per_bin, replace=False))
        early = np.asarray(left, dtype=int)[chosen]; late = np.asarray(right, dtype=int)[chosen]
        earlier_parts.append(early); later_parts.append(late); gap_parts.append(cycles[late] - cycles[early])
    if not earlier_parts:
        return TemporalPairs(np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float))
    return TemporalPairs(np.concatenate(earlier_parts), np.concatenate(later_parts), np.concatenate(gap_parts))
