from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from gradual_state_monitoring_v33.data import MAIN_FEATURES

from .config import GradualStateMonitoringV34Config


ONLINE_COLUMNS = ("dataset", "window_index", "center_cycle", *MAIN_FEATURES)


def load_window_data(path: str | Path) -> pd.DataFrame:
    """Read exactly the allowed online columns from a raw feature artifact."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"V3.4 input does not exist: {source}")
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = sorted(set(ONLINE_COLUMNS).difference(header))
    if missing:
        raise ValueError(f"V3.4 input misses required online columns: {missing}")
    frame = pd.read_csv(source, usecols=list(ONLINE_COLUMNS))
    return validate_online_frame(frame, GradualStateMonitoringV34Config(), exact_columns=True)


def validate_online_frame(frame: pd.DataFrame, config: GradualStateMonitoringV34Config,
                          *, exact_columns: bool = True) -> pd.DataFrame:
    """Enforce the target online information boundary.

    The raw CSV can have auxiliary columns, but ``load_window_data`` projects it
    before entering this guard.  A caller that directly supplies a target frame is
    rejected if it carries anything outside the declared online schema.
    """
    columns = [str(column) for column in frame.columns]
    lower = {column.lower() for column in columns}
    forbidden = sorted(lower.intersection(config.target_forbidden_columns))
    if forbidden:
        raise AssertionError(f"V3.4 target online path rejects label/outcome columns: {forbidden}")
    required = set(config.online_allowed_columns)
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError(f"V3.4 online input misses required columns: {missing}")
    unexpected = sorted(set(columns).difference(required))
    if exact_columns and unexpected:
        raise AssertionError(f"V3.4 target online path accepts only declared columns, not: {unexpected}")
    output = frame.loc[:, list(config.online_allowed_columns)].copy()
    if not np.isfinite(output.loc[:, MAIN_FEATURES].to_numpy(float)).all():
        raise ValueError("V3.4 refuses non-finite online force features")
    if output.dataset.isna().any() or output.window_index.isna().any() or output.center_cycle.isna().any():
        raise ValueError("V3.4 online identifiers must be finite and present")
    return output.sort_values(["dataset", "window_index"], kind="stable").reset_index(drop=True)


def load_stage_segments(path: str | Path, *, dataset: str | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load versioned Stage definitions for a named source or for offline evaluation."""
    mapping_path = Path(path)
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    required = {"dataset", "stage", "effective_start", "effective_end", "actual_start", "actual_end", "note"}
    records = payload.get("segments", [])
    if not records or any(not required.issubset(record) for record in records):
        raise ValueError("Stage mapping must contain explicit versioned segments")
    segments = pd.DataFrame(records)
    if dataset is not None:
        segments = segments[segments.dataset == dataset].copy()
        if segments.empty:
            raise ValueError(f"No Stage segments for {dataset}")
    segments = segments.loc[:, sorted(required)].astype({
        "stage": int, "effective_start": float, "effective_end": float,
        "actual_start": float, "actual_end": float,
    }).sort_values(["dataset", "stage"], kind="stable").reset_index(drop=True)
    provenance: dict[str, object] = {
        "status": "VERSIONED_STAGE_MAPPING",
        "source_path": str(mapping_path),
        "source_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "datasets_loaded": sorted(segments.dataset.unique().tolist()),
    }
    return segments, provenance


def effective_to_actual(cycles: Iterable[float], dataset: str, segments: pd.DataFrame) -> np.ndarray:
    values = np.asarray(list(cycles), dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    local = segments[segments.dataset == dataset].sort_values("stage", kind="stable")
    for _, row in local.iterrows():
        mask = ((values >= float(row.effective_start)) & (values <= float(row.effective_end))
                & ~np.isfinite(output))  # shared endpoints belong to the earlier Stage
        span = max(float(row.effective_end - row.effective_start), 1e-12)
        output[mask] = float(row.actual_start) + ((values[mask] - float(row.effective_start)) *
                                                   float(row.actual_end - row.actual_start) / span)
    return output


def stage_boundaries(segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in segments.groupby("dataset", sort=True):
        ordered = group.sort_values("stage", kind="stable").reset_index(drop=True)
        for index in range(1, len(ordered)):
            before, after = ordered.iloc[index - 1], ordered.iloc[index]
            rows.append({
                "dataset": str(dataset), "from_stage": int(before.stage), "to_stage": int(after.stage),
                "boundary_cycle": float(after.effective_start),
                "boundary_actual_cycle": float(after.actual_start),
            })
    return pd.DataFrame(rows)


def source_soft_boundary_labels(source: pd.DataFrame, source_segments: pd.DataFrame,
                                config: GradualStateMonitoringV34Config) -> tuple[np.ndarray, np.ndarray]:
    """Create source-only Gaussian weak labels and their actual cycle coordinates."""
    dataset = str(source.dataset.iloc[0])
    actual = effective_to_actual(source.center_cycle.to_numpy(float), dataset, source_segments)
    boundaries = stage_boundaries(source_segments)
    values = boundaries.loc[boundaries.dataset == dataset, "boundary_actual_cycle"].to_numpy(float)
    if len(values) == 0:
        return np.zeros(len(source), dtype=float), actual
    distance = np.min(np.abs(actual[:, None] - values[None, :]), axis=1)
    labels = np.exp(-0.5 * (distance / config.source_boundary_sigma_actual_cycles) ** 2)
    labels[distance > config.source_boundary_support_actual_cycles] = 0.0
    return labels.astype(float), actual


def attach_posthoc_stage(online: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Append target labels only in the explicit offline evaluation phase."""
    result = online.copy()
    result["actual_cycle"] = np.nan
    result["stage"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    for dataset, indices in result.groupby("dataset", sort=False).groups.items():
        positions = list(indices)
        cycles = result.loc[positions, "center_cycle"].to_numpy(float)
        result.loc[positions, "actual_cycle"] = effective_to_actual(cycles, str(dataset), segments)
        local = segments[segments.dataset == dataset]
        assigned = np.full(len(positions), np.nan)
        for _, segment in local.iterrows():
            mask = ((cycles >= float(segment.effective_start)) & (cycles <= float(segment.effective_end))
                    & ~np.isfinite(assigned))
            assigned[mask] = int(segment.stage)
        result.loc[positions, "stage"] = pd.array(assigned, dtype="Int64")
    return result
