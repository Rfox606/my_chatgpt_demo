from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PartialSharedPrimitivesConfig


FORBIDDEN_COLUMNS = frozenset({
    "stage", "stage1to5", "sa", "sq", "sz", "sku", "morphology", "wear_debris", "wear_debris_count", "debris_count",
    "wear", "absolute_wear", "volume_loss", "mass_loss",
})
REQUIRED_COLUMNS = ("dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle")


def assert_label_free(frame: pd.DataFrame) -> None:
    names = {str(column).strip().lower() for column in frame.columns}
    forbidden = sorted(names.intersection(FORBIDDEN_COLUMNS))
    if forbidden:
        raise AssertionError(f"v3 formal input contains forbidden labels/metadata: {forbidden}")


def load_windows(config: PartialSharedPrimitivesConfig) -> pd.DataFrame:
    frame = pd.read_csv(config.input_path)
    assert_label_free(frame)
    missing = set(REQUIRED_COLUMNS).difference(frame.columns) | set(config.feature_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing v3 input columns: {sorted(missing)}")
    result = frame.loc[:, [*REQUIRED_COLUMNS, *config.feature_columns]].copy()
    for column in ("window_id", "window_index", "start_cycle", "end_cycle", "center_cycle", *config.feature_columns):
        result[column] = pd.to_numeric(result[column], errors="raise")
    if not np.isfinite(result.loc[:, ["center_cycle", *config.feature_columns]].to_numpy(float)).all():
        raise ValueError("v3 formal inputs must be finite")
    return result.sort_values(["dataset", "center_cycle", "window_index"]).reset_index(drop=True)


def input_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_location_scale(values: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return np.zeros(values.shape[1]), np.ones(values.shape[1])
    location = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - location), axis=0)
    iqr = (np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)) / 1.349
    return location, np.maximum.reduce((mad, iqr, np.full(values.shape[1], eps)))

