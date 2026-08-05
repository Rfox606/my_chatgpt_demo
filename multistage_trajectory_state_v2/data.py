from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURE_CONFIGS, MultiStageTrajectoryConfig


FORBIDDEN_COLUMNS = frozenset({
    "stage", "stage1to5", "sa", "sq", "sz", "sku", "morphology", "wear_debris", "wear_debris_count", "debris_count",
})
REQUIRED_COLUMNS = ("dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle")


def assert_label_free(frame: pd.DataFrame) -> None:
    columns = {str(name).strip().lower() for name in frame.columns}
    forbidden = sorted(columns.intersection(FORBIDDEN_COLUMNS))
    if forbidden:
        raise AssertionError(f"Labels, morphology, or debris reached formal v2 input: {forbidden}")


def all_features(config: MultiStageTrajectoryConfig) -> tuple[str, ...]:
    return tuple(dict.fromkeys(feature for name in config.feature_configs for feature in FEATURE_CONFIGS[name]))


def load_windows(config: MultiStageTrajectoryConfig) -> pd.DataFrame:
    frame = pd.read_csv(config.input_path)
    assert_label_free(frame)
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required raw-window columns: {sorted(missing)}")
    needed = all_features(config)
    missing_features = set(needed).difference(frame.columns)
    if missing_features:
        raise ValueError(f"Missing predeclared force features: {sorted(missing_features)}")
    result = frame.loc[:, [*REQUIRED_COLUMNS, *needed]].copy()
    for column in ("window_id", "window_index", "start_cycle", "end_cycle", "center_cycle", *needed):
        result[column] = pd.to_numeric(result[column], errors="raise")
    if not np.isfinite(result.loc[:, ["center_cycle", *needed]].to_numpy(float)).all():
        raise ValueError("Formal v2 input must be finite")
    return result.sort_values(["dataset", "center_cycle", "window_index"]).reset_index(drop=True)


def robust_scale(values: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    location = np.median(values, axis=0)
    mad = np.median(np.abs(values - location), axis=0)
    iqr = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
    scale = np.maximum.reduce((1.4826 * mad, iqr / 1.349, np.full(values.shape[1], eps)))
    return location, scale


def equal_time_sample(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    ordered = frame.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    if len(ordered) <= maximum:
        return ordered
    positions = np.unique(np.rint(np.linspace(0, len(ordered) - 1, maximum)).astype(int))
    return ordered.iloc[positions].reset_index(drop=True)


def input_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
