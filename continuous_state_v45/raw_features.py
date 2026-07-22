from __future__ import annotations

"""Rebuild direct, unnormalised sensitive-phase force features without reading Stage."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from run_weighted_awrcore_models import corr_dist_to_base, phase_columns

from .config import ContinuousStateV45Config


DIRECT_FAMILIES = ("mean", "absmean", "rms", "std", "p2p", "q95", "q05")
RAW_FEATURES = tuple([f"{channel}_{family}" for channel in ("rx", "ry", "rs") for family in DIRECT_FAMILIES])


@dataclass(frozen=True)
class SensitiveCycleWaves:
    dataset: str
    cycle_index: np.ndarray
    waves: dict[str, np.ndarray]
    phase_points_all: int
    sensitive_indices_1based: tuple[int, int]
    sensitive_points: int
    raw_path: str


def _direct_cycle_features(waves: dict[str, np.ndarray]) -> pd.DataFrame:
    values: dict[str, np.ndarray] = {}
    for channel, matrix in waves.items():
        values[f"{channel}_mean"] = matrix.mean(axis=1)
        values[f"{channel}_absmean"] = np.abs(matrix).mean(axis=1)
        values[f"{channel}_rms"] = np.sqrt((matrix * matrix).mean(axis=1))
        values[f"{channel}_std"] = matrix.std(axis=1)
        values[f"{channel}_p2p"] = matrix.max(axis=1) - matrix.min(axis=1)
        values[f"{channel}_q95"] = np.percentile(matrix, 95, axis=1)
        values[f"{channel}_q05"] = np.percentile(matrix, 5, axis=1)
    return pd.DataFrame(values)


def load_sensitive_force_cycles(dataset: str, path: Path, config: ContinuousStateV45Config) -> SensitiveCycleWaves:
    """Mirror the established Fx/Fy/Fz phase selection while deliberately excluding Stage1to5."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    fx_all, fy_all, fz_all = (phase_columns(header, name) for name in ("Fx", "Fy", "Fz"))
    phase_count = min(len(fx_all), len(fy_all), len(fz_all))
    lo = max(1, int(math.ceil(config.sensitive_phase[0] * phase_count)))
    hi = min(phase_count, int(math.floor(config.sensitive_phase[1] * phase_count)))
    selected = np.arange(lo, hi + 1, dtype=int)
    columns = {name: [f"{name}_p{index:04d}" for index in selected] for name in ("Fx", "Fy", "Fz")}
    usecols = ["CycleIndex", *columns["Fx"], *columns["Fy"], *columns["Fz"]]
    missing = sorted(set(usecols).difference(header))
    if missing:
        raise ValueError(f"{path} lacks sensitive-force columns: {missing[:8]}")
    # Stage1to5 is intentionally not in usecols.
    frame = pd.read_csv(path, usecols=usecols)
    fx = frame.loc[:, columns["Fx"]].to_numpy(float)
    fy = frame.loc[:, columns["Fy"]].to_numpy(float)
    fz = frame.loc[:, columns["Fz"]].to_numpy(float)
    valid = np.abs(fz) >= config.eps
    rx = np.divide(fx, fz, out=np.full_like(fx, np.nan), where=valid)
    ry = np.divide(fy, fz, out=np.full_like(fy, np.nan), where=valid)
    rs = np.sqrt(rx * rx + ry * ry)
    if not np.isfinite(np.stack((rx, ry, rs))).all():
        raise ValueError(f"{dataset} contains non-finite sensitive-phase ratios; v4.5 refuses non-causal global imputation")
    return SensitiveCycleWaves(dataset, frame.CycleIndex.to_numpy(int), {"rx": rx, "ry": ry, "rs": rs}, phase_count, (int(lo), int(hi)), int(len(selected)), str(path))


def _window_means(cycles: np.ndarray, features: pd.DataFrame, config: ContinuousStateV45Config, dataset: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = features.to_numpy(float)
    for window_index, start in enumerate(range(0, len(features) - config.window_cycles + 1, config.window_stride_cycles)):
        end = start + config.window_cycles
        row: dict[str, object] = {
            "dataset": dataset, "window_id": int(window_index), "window_index": int(window_index),
            "start_cycle": float(cycles[start]), "end_cycle": float(cycles[end - 1]),
            "center_cycle": float((cycles[start] + cycles[end - 1]) / 2.0),
            "baseline_window_500": int(cycles[end - 1] <= 500),
            "baseline_window_1000": int(cycles[end - 1] <= 1000),
            "baseline_window_2000": int(cycles[end - 1] <= 2000),
        }
        row.update({name: float(value) for name, value in zip(features.columns, values[start:end].mean(axis=0))})
        rows.append(row)
    return pd.DataFrame(rows)


def direct_window_features(raw: SensitiveCycleWaves, config: ContinuousStateV45Config) -> pd.DataFrame:
    """The saved raw table has direct physical features only; corrdist is baseline-defined later."""
    return _window_means(raw.cycle_index, _direct_cycle_features(raw.waves), config, raw.dataset)


def add_baseline_corrdist(raw: SensitiveCycleWaves, direct_windows: pd.DataFrame, baseline_cycles: int, config: ContinuousStateV45Config) -> pd.DataFrame:
    mask = raw.cycle_index <= int(baseline_cycles)
    if int(mask.sum()) < int(baseline_cycles):
        raise ValueError(f"{raw.dataset} has too few cycles for {baseline_cycles}-cycle corrdist baseline")
    cycle_corrdist: dict[str, np.ndarray] = {}
    for channel, matrix in raw.waves.items():
        baseline_wave = matrix[mask].mean(axis=0)
        cycle_corrdist[f"{channel}_corrdist_base"] = corr_dist_to_base(matrix, baseline_wave, config.eps)
    corrdist_windows = _window_means(raw.cycle_index, pd.DataFrame(cycle_corrdist), config, raw.dataset)
    ids = ["dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle", "baseline_window_500", "baseline_window_1000", "baseline_window_2000"]
    return direct_windows.merge(corrdist_windows.loc[:, [*ids, *cycle_corrdist]], on=ids, how="left", validate="one_to_one")


def raw_provenance(raws: dict[str, SensitiveCycleWaves], config: ContinuousStateV45Config) -> dict[str, object]:
    return {
        "status": "PASS", "stage_read": False, "upstream_z_standardisation_used": False, "upstream_z_clip_used": False,
        "raw_feature_definition": "direct Fx/Fz, Fy/Fz and resultant sensitive-phase summaries; no pre-normalisation",
        "sensitive_phase": list(config.sensitive_phase), "window_cycles": config.window_cycles, "window_stride_cycles": config.window_stride_cycles,
        "datasets": {dataset: {"raw_path": raw.raw_path, "cycle_count": int(len(raw.cycle_index)), "phase_points_all": raw.phase_points_all,
                                "sensitive_indices_1based": list(raw.sensitive_indices_1based), "sensitive_points": raw.sensitive_points}
                     for dataset, raw in raws.items()},
    }
