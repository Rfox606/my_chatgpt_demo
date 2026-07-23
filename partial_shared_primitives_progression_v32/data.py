from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import V32Config


FORBIDDEN = frozenset({"stage", "stage1to5", "sa", "sq", "sz", "sku", "morphology", "wear_debris", "wear_debris_count", "debris_count", "wear", "absolute_wear", "mass_loss", "volume_loss"})
REQUIRED = ("dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle")


def assert_formal_input(frame: pd.DataFrame) -> None:
    bad = sorted({str(column).strip().lower() for column in frame.columns}.intersection(FORBIDDEN))
    if bad: raise AssertionError(f"Forbidden v3.2 model input: {bad}")


def load_windows(config: V32Config) -> pd.DataFrame:
    frame = pd.read_csv(config.input_path); assert_formal_input(frame)
    missing = set(REQUIRED).difference(frame.columns) | set(config.features).difference(frame.columns)
    if missing: raise ValueError(f"Missing inputs: {sorted(missing)}")
    result = frame.loc[:, [*REQUIRED, *config.features]].copy()
    for column in (*REQUIRED[1:], *config.features): result[column] = pd.to_numeric(result[column], errors="raise")
    if not np.isfinite(result.loc[:, ["center_cycle", *config.features]].to_numpy(float)).all(): raise ValueError("Inputs must be finite")
    return result.sort_values(["dataset", "center_cycle", "window_index"]).reset_index(drop=True)


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = np.median(values, axis=0); mad = 1.4826 * np.median(np.abs(values - location), axis=0); iqr = (np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)) / 1.349
    return location, np.maximum.reduce((mad, iqr, np.full(values.shape[1], 1e-9)))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
