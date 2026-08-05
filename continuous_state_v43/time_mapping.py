from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ContinuousStateV43Config


STOP_CYCLES_ACTUAL = {
    "Exp1": np.asarray((8000, 16000, 24000, 32000, 40000, 48000), dtype=float),
    "Exp2": np.arange(500, 23501, 500, dtype=float),
}


def _mapping_segments(config: ContinuousStateV43Config) -> tuple[list[dict[str, object]], str]:
    """Use a row-level mapping if present; otherwise use the established public segment map."""
    # A repository search found no row-level actual-cycle index in the state input.
    # This pre-existing configuration is therefore the only mapping read here.
    path = Path(config.cycle_mapping_config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["segments"]), str(path)


def map_effective_to_actual(dataset: str, cycles: np.ndarray, config: ContinuousStateV43Config) -> np.ndarray:
    segments, _ = _mapping_segments(config)
    selected = [row for row in segments if row["dataset"] == dataset]
    if not selected:
        raise ValueError(f"No cycle mapping segments for {dataset}")
    values = np.asarray(cycles, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    for index, row in enumerate(selected):
        left = float(row["effective_start"]); right = float(row["effective_end"])
        # Shared segment endpoints are assigned once, without changing either endpoint value.
        mask = (values >= left) & ((values < right) if index < len(selected) - 1 else (values <= right))
        ratio = (values[mask] - left) / (right - left)
        result[mask] = float(row["actual_start"]) + ratio * (float(row["actual_end"]) - float(row["actual_start"]))
    if np.isnan(result).any():
        raise ValueError(f"Effective cycles outside established {dataset} mapping support")
    return result


def add_actual_cycle_columns(frame: pd.DataFrame, config: ContinuousStateV43Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach traceable effective and actual time fields; neither is a model feature."""
    result = frame.copy()
    for name in ("start", "end", "center"):
        effective = result[f"{name}_cycle"].to_numpy(float)
        result[f"{name}_cycle_effective"] = effective
        result[f"{name}_cycle_actual"] = np.concatenate([
            map_effective_to_actual(dataset, group[f"{name}_cycle"].to_numpy(float), config)
            for dataset, group in result.groupby("dataset", sort=False)
        ])
    # regrouping preserves current dataset blocks, but assign directly by dataset to keep index alignment explicit.
    for dataset, positions in result.groupby("dataset", sort=False).groups.items():
        for name in ("start", "end", "center"):
            result.loc[positions, f"{name}_cycle_actual"] = map_effective_to_actual(dataset, result.loc[positions, f"{name}_cycle_effective"].to_numpy(float), config)
    result["cycle_effective"] = result["center_cycle_effective"]
    result["cycle_actual"] = result["center_cycle_actual"]
    result["cycle_mapping_source"] = "existing_segment_config_fallback"
    result["cycle_mapping_config"] = str(config.cycle_mapping_config_path)
    segments, source = _mapping_segments(config)
    rows = []
    for dataset, group in result.groupby("dataset", sort=True):
        rows.append({
            "dataset": dataset,
            "mapping_source": "existing_segment_config_fallback",
            "mapping_config": source,
            "row_count": int(len(group)),
            "cycle_effective_min": float(group.start_cycle_effective.min()),
            "cycle_effective_max": float(group.end_cycle_effective.max()),
            "cycle_actual_min": float(group.start_cycle_actual.min()),
            "cycle_actual_max": float(group.end_cycle_actual.max()),
            "monotone_actual": bool(np.all(np.diff(group.center_cycle_actual.to_numpy(float)) >= 0)),
            "segment_count": int(sum(row["dataset"] == dataset for row in segments)),
        })
    return result, pd.DataFrame(rows)


def nearest_stop_distance_actual(dataset: str, cycles: np.ndarray) -> np.ndarray:
    stops = STOP_CYCLES_ACTUAL[dataset]
    values = np.asarray(cycles, dtype=float)
    return np.min(np.abs(values[:, None] - stops[None, :]), axis=1)


def stop_buffer_mask(dataset: str, start_actual: np.ndarray, end_actual: np.ndarray, half_width: float) -> np.ndarray:
    """True where an actual-cycle window intersects any stop +/- fixed half width."""
    stops = STOP_CYCLES_ACTUAL[dataset]
    start = np.asarray(start_actual, dtype=float); end = np.asarray(end_actual, dtype=float)
    return np.any((start[:, None] <= stops[None, :] + half_width) & (end[:, None] >= stops[None, :] - half_width), axis=1)
