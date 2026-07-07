from __future__ import annotations

import json
import logging
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class WeightedAWRConfig:
    output_dir: str = "outputs_weighted_awrcore_v1"
    v2_dir: str = "outputs_aux_state_metrics_v2"
    raw_files: Dict[str, str] = field(
        default_factory=lambda: {
            "Exp1": "Exp1_original_Fx_Fy_Fz_labels.csv",
            "Exp2": "Exp2_original_Fx_Fy_Fz_labels.csv",
        }
    )
    sensitive_phase: Tuple[float, float] = (0.45, 0.63)
    baseline_cycles: int = 500
    window_k: int = 20
    stride: int = 5
    source_gap_windows: int = 4
    eps: float = 1e-9
    z_clip_min: float = -12.0
    z_clip_max: float = 12.0
    direction_delta_min: float = 0.05
    effect_size_min: float = 0.05
    bootstrap_n: int = 500
    bootstrap_unit: str = "block_window"
    bootstrap_block_size: int = 20
    unstable_direction_threshold: float = 0.70
    enter_weight_stability_threshold: float = 0.60
    redundancy_threshold: float = 0.95
    redundancy_method: str = "pearson_downweight_lower_rank"
    redundancy_downweight: float = 0.5
    m3_alpha_grid_step: float = 0.05
    high_awr_threshold_percentile: float = 95.0
    awr_smoothing_window: int = 5
    rs_horizons: Tuple[int, int, int] = (20, 50, 100)
    rs_min_threshold: float = 0.005
    rs_threshold_percentile: float = 95.0
    rs_rising_consecutive_windows: int = 3
    tes_window: int = 20
    tes_smoothing_window: int = 5
    tes_weight_awr_vol: float = 0.4
    tes_weight_bd_jump: float = 0.4
    tes_weight_shape_jump: float = 0.2
    tes_threshold_percentile: float = 99.5
    tes_threshold_floor: float = 3.0
    tes_prominence_percentile: float = 95.0
    tes_prominence_floor: float = 1.0
    min_event_gap_cycles: int = 1000
    event_neighborhood_windows: int = 2
    boundary_match_tolerance_cycles: int = 500
    bd_default_metric: str = "BDall_xy_v2"
    random_seed: int = 20260707


MAIN_FEATURE_FAMILIES = [
    "mean",
    "absmean",
    "rms",
    "q05",
    "q95",
    "p2p",
    "std",
    "corrdist_base",
    "peak_phase",
    "peak_width",
]
RS_FEATURE_FAMILIES = [
    "mean",
    "absmean",
    "rms",
    "q05",
    "q95",
    "p2p",
    "std",
    "corrdist_base",
]
STABLE_PLUS_FEATURES = [
    "rs_corrdist_base",
    "rs_mean",
    "rs_absmean",
    "rs_q05",
    "rx_corrdist_base",
    "rs_rms",
    "ry_p2p",
    "rx_mean",
    "rx_absmean",
    "rx_q05",
]
MODEL_SCORE_COLS = {
    "M0": "AWR_M0",
    "M1": "AWR_M1",
    "M2": "AWR_M2",
    "M3_equal": "AWR_M3_equal",
    "M3_weighted": "AWR_M3_weighted",
}
REGION_COLORS = {
    "low_AWR_low_BD": "#7c8a99",
    "low_AWR_high_BD": "#d4a72c",
    "high_AWR_high_BD": "#b64b5a",
    "high_AWR_low_BD": "#5a6fb0",
}
MODEL_COLORS = {
    "M0": "#7a7a7a",
    "M1": "#3b6ea8",
    "M2": "#46a67a",
    "M3_equal": "#c76d2a",
    "M3_weighted": "#b64b5a",
}


def setup_dirs(config: WeightedAWRConfig) -> Dict[str, Path]:
    root = Path(config.output_dir)
    dirs = {
        "root": root,
        "results": root / "results",
        "figures": root / "figures",
        "reports": root / "reports",
        "diagnostics": root / "diagnostics",
        "configs": root / "configs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(dirs: Dict[str, Path]) -> None:
    log_path = dirs["root"] / "weighted_awrcore_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def finite_values(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def finite_quantile(values: Iterable[float], percentile: float, default: float = np.nan) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return float(default)
    return float(np.nanpercentile(arr, percentile))


def finite_median(values: Iterable[float], default: float = np.nan) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return float(default)
    return float(np.nanmedian(arr))


def robust_iqr(values: Iterable[float], default: float = np.nan) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return float(default)
    return float(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25))


def sanitize(values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out)] = fill
    return out


def phase_columns(header: List[str], channel: str) -> List[str]:
    cols = [col for col in header if col.startswith(f"{channel}_p")]
    return sorted(cols, key=lambda item: int(item.split("_p", 1)[1]))


def corr_dist_to_base(arr: np.ndarray, base: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(arr, dtype=float)
    base = np.asarray(base, dtype=float)
    base_center = base - np.nanmean(base)
    base_center = np.where(np.isfinite(base_center), base_center, 0.0)
    base_norm = float(np.sqrt(np.dot(base_center, base_center)))
    if base_norm < eps:
        return np.zeros(values.shape[0], dtype=float)
    row_mean = np.nanmean(values, axis=1, keepdims=True)
    centered = values - row_mean
    centered = np.where(np.isfinite(centered), centered, 0.0)
    row_norm = np.sqrt(np.sum(centered * centered, axis=1))
    denom = row_norm * base_norm
    corr = np.divide(
        centered @ base_center,
        denom,
        out=np.zeros(values.shape[0], dtype=float),
        where=denom >= eps,
    )
    return 1.0 - np.clip(corr, -1.0, 1.0)


def peak_phase_width(arr: np.ndarray, selected_phase: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(arr, dtype=float)
    abs_values = np.abs(values)
    safe = np.where(np.isfinite(abs_values), abs_values, -np.inf)
    peak_idx = np.argmax(safe, axis=1)
    peak_amp = safe[np.arange(safe.shape[0]), peak_idx]
    phase = selected_phase[peak_idx]
    phase = np.where(np.isfinite(peak_amp), phase, np.nan)
    threshold = 0.80 * np.maximum(peak_amp, eps)
    width = np.mean(abs_values >= threshold[:, None], axis=1)
    width[~np.isfinite(width)] = np.nan
    return phase, width


def feature_metadata(feature_list: List[str]) -> pd.DataFrame:
    meanings = {
        "mean": "signed average shear ratio in the sensitive phase",
        "absmean": "average shear magnitude in the sensitive phase",
        "rms": "energy-like shear ratio magnitude",
        "q05": "lower-tail sensitive-phase shear ratio",
        "q95": "upper-tail sensitive-phase shear ratio",
        "p2p": "peak-to-peak sensitive-phase span",
        "std": "within-phase dispersion",
        "corrdist_base": "shape departure from the baseline mean waveform",
        "peak_phase": "phase location of the largest absolute response",
        "peak_width": "fraction of sensitive phase near the peak response",
    }
    rows = []
    for name in feature_list:
        channel = name.split("_", 1)[0]
        family = name[len(channel) + 1 :]
        if channel == "rx":
            channel_meaning = "Fx/Fz main shear ratio"
        elif channel == "ry":
            channel_meaning = "Fy/Fz lateral shear ratio"
        else:
            channel_meaning = "resultant shear ratio contrast"
        rows.append(
            {
                "feature_name": name,
                "channel": channel,
                "feature_family": family,
                "physical_meaning": f"{channel_meaning}; {meanings.get(family, family)}",
            }
        )
    return pd.DataFrame(rows)


def all_feature_names() -> List[str]:
    out = []
    for channel in ("rx", "ry"):
        out.extend([f"{channel}_{family}" for family in MAIN_FEATURE_FAMILIES])
    out.extend([f"rs_{family}" for family in RS_FEATURE_FAMILIES])
    return out


def load_cycle_feature_data(dataset: str, path: Path, config: WeightedAWRConfig) -> Dict[str, object]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    fx_all = phase_columns(header, "Fx")
    fy_all = phase_columns(header, "Fy")
    fz_all = phase_columns(header, "Fz")
    n_phase = min(len(fx_all), len(fy_all), len(fz_all))
    lo = max(1, int(math.ceil(config.sensitive_phase[0] * n_phase)))
    hi = min(n_phase, int(math.floor(config.sensitive_phase[1] * n_phase)))
    selected = np.arange(lo, hi + 1, dtype=int)
    cols = {
        "Fx": [f"Fx_p{i:04d}" for i in selected],
        "Fy": [f"Fy_p{i:04d}" for i in selected],
        "Fz": [f"Fz_p{i:04d}" for i in selected],
    }
    usecols = ["CycleIndex", "Stage1to5"] + cols["Fx"] + cols["Fy"] + cols["Fz"]
    missing = sorted(set(usecols) - set(header))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing[:8]}")

    logging.info("Reading sensitive phase columns for %s from %s", dataset, path)
    df = pd.read_csv(path, usecols=usecols)
    fx = df[cols["Fx"]].to_numpy(dtype=np.float64)
    fy = df[cols["Fy"]].to_numpy(dtype=np.float64)
    fz = df[cols["Fz"]].to_numpy(dtype=np.float64)
    valid_den = np.abs(fz) >= config.eps
    rx = np.divide(fx, fz, out=np.full_like(fx, np.nan), where=valid_den)
    ry = np.divide(fy, fz, out=np.full_like(fy, np.nan), where=valid_den)
    rs = np.sqrt(rx * rx + ry * ry)
    phase = selected.astype(float) / float(n_phase)

    features: Dict[str, np.ndarray] = {}
    baseline_n = min(config.baseline_cycles, len(df))
    for channel, arr in {"rx": rx, "ry": ry, "rs": rs}.items():
        base = arr[:baseline_n]
        finite_base_wave = np.nanmean(base, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            features[f"{channel}_mean"] = np.nanmean(arr, axis=1)
            features[f"{channel}_absmean"] = np.nanmean(np.abs(arr), axis=1)
            features[f"{channel}_rms"] = np.sqrt(np.nanmean(arr * arr, axis=1))
            features[f"{channel}_std"] = np.nanstd(arr, axis=1)
            features[f"{channel}_p2p"] = np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)
            features[f"{channel}_q95"] = np.nanpercentile(arr, 95, axis=1)
            features[f"{channel}_q05"] = np.nanpercentile(arr, 5, axis=1)
            features[f"{channel}_corrdist_base"] = corr_dist_to_base(arr, finite_base_wave, config.eps)
        if channel in ("rx", "ry"):
            peak_phase, peak_width = peak_phase_width(arr, phase, config.eps)
            features[f"{channel}_peak_phase"] = peak_phase
            features[f"{channel}_peak_width"] = peak_width

    X = pd.DataFrame(features)
    for col in X.columns:
        values = X[col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        fill = float(np.nanmedian(finite)) if finite.size else 0.0
        values = np.where(np.isfinite(values), values, fill)
        X[col] = values

    return {
        "dataset": dataset,
        "path": str(path),
        "cycle_index": df["CycleIndex"].to_numpy(dtype=int),
        "stage": df["Stage1to5"].to_numpy(dtype=int),
        "X_cycle": X,
        "raw_meta": {
            "dataset": dataset,
            "rows": int(len(df)),
            "phase_points_all": int(n_phase),
            "sensitive_index_1based": [int(lo), int(hi)],
            "sensitive_points": int(len(selected)),
            "near_zero_fz_count": int((~valid_den).sum()),
            "stage_counts": {
                str(k): int(v)
                for k, v in df["Stage1to5"].value_counts(dropna=False).sort_index().items()
            },
        },
    }


def compute_window_features(data: Dict[str, object], config: WeightedAWRConfig) -> pd.DataFrame:
    feature_list = all_feature_names()
    X_cycle = data["X_cycle"][feature_list]
    arr = X_cycle.to_numpy(dtype=float)
    cycle_index = np.asarray(data["cycle_index"], dtype=int)
    stage = np.asarray(data["stage"], dtype=int)
    rows = []
    window_id = 0
    for start in range(0, len(X_cycle) - config.window_k + 1, config.stride):
        end = start + config.window_k
        values = np.nanmean(arr[start:end, :], axis=0)
        row = {
            "dataset": data["dataset"],
            "window_id": int(window_id),
            "window_index": int(window_id),
            "start_cycle": float(cycle_index[start]),
            "end_cycle": float(cycle_index[end - 1]),
            "center_cycle": float((cycle_index[start] + cycle_index[end - 1]) / 2.0),
            "stage": int(stage[end - 1]),
            "stage_label": f"Stage {int(stage[end - 1])}",
            "baseline_window": int(cycle_index[end - 1] <= config.baseline_cycles),
        }
        row.update({feature: float(value) for feature, value in zip(feature_list, values)})
        rows.append(row)
        window_id += 1
    return pd.DataFrame(rows)


def load_raw_or_window_data(config: WeightedAWRConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    feature_frames = []
    raw_meta = {}
    for dataset, file_name in config.raw_files.items():
        path = Path(file_name)
        if not path.exists():
            raise FileNotFoundError(f"Missing raw force file for {dataset}: {path}")
        data = load_cycle_feature_data(dataset, path, config)
        raw_meta[dataset] = data["raw_meta"]
        feature_frames.append(compute_window_features(data, config))
    wide = pd.concat(feature_frames, ignore_index=True)
    meta = feature_metadata(all_feature_names())
    long = wide.melt(
        id_vars=[
            "dataset",
            "window_id",
            "window_index",
            "start_cycle",
            "end_cycle",
            "center_cycle",
            "stage",
            "stage_label",
            "baseline_window",
        ],
        value_vars=all_feature_names(),
        var_name="feature_name",
        value_name="feature_value",
    ).merge(meta, on="feature_name", how="left")
    return wide, long, meta, raw_meta


def robust_baseline_normalize(
    wide: pd.DataFrame,
    meta: pd.DataFrame,
    config: WeightedAWRConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_list = all_feature_names()
    id_cols = [
        "dataset",
        "window_id",
        "window_index",
        "start_cycle",
        "end_cycle",
        "center_cycle",
        "stage",
        "stage_label",
        "baseline_window",
    ]
    z_wide = wide[id_cols].copy()
    diagnostics = []
    for dataset, group in wide.groupby("dataset", sort=True):
        base_mask = group["end_cycle"].to_numpy(dtype=float) <= config.baseline_cycles
        if int(base_mask.sum()) < 20:
            base_mask = np.zeros(len(group), dtype=bool)
            base_mask[: min(100, len(base_mask))] = True
        idx = group.index
        for feature in feature_list:
            values = group[feature].to_numpy(dtype=float)
            ref = finite_values(values[base_mask])
            median_base = float(np.nanmedian(ref)) if ref.size else 0.0
            iqr_base = robust_iqr(ref, default=0.0)
            iqr_used = float(iqr_base + config.eps)
            raw_z = (np.where(np.isfinite(values), values, median_base) - median_base) / iqr_used
            clipped = np.clip(raw_z, config.z_clip_min, config.z_clip_max)
            z_wide.loc[idx, feature] = clipped
            mrow = meta[meta["feature_name"] == feature].iloc[0]
            diagnostics.append(
                {
                    "dataset": dataset,
                    "feature_name": feature,
                    "channel": mrow["channel"],
                    "feature_family": mrow["feature_family"],
                    "median_base": median_base,
                    "IQR_base": float(iqr_base),
                    "IQR_used": iqr_used,
                    "clip_min": float(config.z_clip_min),
                    "clip_max": float(config.z_clip_max),
                    "saturation_rate_low": float(np.nanmean(raw_z <= config.z_clip_min)),
                    "saturation_rate_high": float(np.nanmean(raw_z >= config.z_clip_max)),
                    "missing_rate": float(np.mean(~np.isfinite(values))),
                }
            )
    z_long = z_wide.melt(
        id_vars=id_cols,
        value_vars=feature_list,
        var_name="feature_name",
        value_name="z_value",
    ).merge(meta, on="feature_name", how="left")
    return z_wide, z_long, pd.DataFrame(diagnostics)


def source_split_by_stage(stages: np.ndarray, gap: int) -> Tuple[np.ndarray, np.ndarray]:
    stages = np.asarray(stages, dtype=int)
    train = np.zeros(len(stages), dtype=bool)
    val = np.zeros(len(stages), dtype=bool)
    for stage in sorted(set(stages.tolist())):
        idx = np.where(stages == stage)[0]
        if len(idx) < 20:
            continue
        cut = int(len(idx) * 0.70)
        train_end = max(0, cut - gap)
        val_start = min(len(idx), cut + gap)
        train[idx[:train_end]] = True
        val[idx[val_start:]] = True
    if int(val.sum()) < 20:
        cut = int(len(stages) * 0.70)
        train[: max(0, cut - gap)] = True
        val[min(len(stages), cut + gap) :] = True
    return train, val


def block_resample_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if n <= block_size:
        return rng.integers(0, n, size=n)
    block_count = int(math.ceil(n / float(block_size)))
    starts = rng.integers(0, n - block_size + 1, size=block_count)
    idx = np.concatenate([np.arange(start, start + block_size) for start in starts])
    return idx[:n]


def cliffs_delta(early: np.ndarray, late: np.ndarray) -> float:
    early = finite_values(early)
    late = finite_values(late)
    if early.size == 0 or late.size == 0:
        return float("nan")
    combined = np.concatenate([late, early])
    ranks = pd.Series(combined).rank(method="average").to_numpy(dtype=float)
    n_late = late.size
    n_early = early.size
    rank_late = float(ranks[:n_late].sum())
    u_late = rank_late - n_late * (n_late + 1) / 2.0
    return float((2.0 * u_late / (n_late * n_early)) - 1.0)


def cohen_d(early: np.ndarray, late: np.ndarray) -> float:
    early = finite_values(early)
    late = finite_values(late)
    if early.size < 2 or late.size < 2:
        return float("nan")
    pooled = math.sqrt(
        ((early.size - 1) * np.nanvar(early, ddof=1) + (late.size - 1) * np.nanvar(late, ddof=1))
        / max(early.size + late.size - 2, 1)
    )
    if pooled < 1e-12:
        return 0.0
    return float((np.nanmean(late) - np.nanmean(early)) / pooled)


def bootstrap_direction_stability(
    early_matrix: np.ndarray,
    late_matrix: np.ndarray,
    direction_signs: np.ndarray,
    config: WeightedAWRConfig,
) -> np.ndarray:
    early_matrix = np.asarray(early_matrix, dtype=float)
    late_matrix = np.asarray(late_matrix, dtype=float)
    n_features = early_matrix.shape[1]
    stability = np.zeros(n_features, dtype=float)
    active = direction_signs != 0
    if early_matrix.shape[0] < 5 or late_matrix.shape[0] < 5 or not active.any():
        return stability
    rng = np.random.default_rng(config.random_seed)
    matches = np.zeros(n_features, dtype=float)
    for _ in range(config.bootstrap_n):
        ei = block_resample_indices(early_matrix.shape[0], config.bootstrap_block_size, rng)
        li = block_resample_indices(late_matrix.shape[0], config.bootstrap_block_size, rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            delta = np.nanmedian(late_matrix[li, :], axis=0) - np.nanmedian(early_matrix[ei, :], axis=0)
        boot_sign = np.sign(delta)
        matches += (boot_sign == direction_signs).astype(float)
    stability[active] = matches[active] / float(config.bootstrap_n)
    return stability


def determine_direction_signs(
    direction_id: str,
    source_dataset: str,
    target_dataset: str,
    z_wide: pd.DataFrame,
    meta: pd.DataFrame,
    config: WeightedAWRConfig,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    source = z_wide[z_wide["dataset"] == source_dataset].sort_values("window_index").reset_index(drop=True)
    stages = source["stage"].to_numpy(dtype=int)
    train_mask, val_mask = source_split_by_stage(stages, config.source_gap_windows)
    early_mask = train_mask & (stages == 1)
    late_mask = train_mask & (stages == 5)
    if int(early_mask.sum()) < 20:
        early_mask = train_mask & (stages <= 1)
    if int(late_mask.sum()) < 20:
        late_mask = train_mask & (stages >= 4)

    features = all_feature_names()
    early_matrix = source.loc[early_mask, features].to_numpy(dtype=float)
    late_matrix = source.loc[late_mask, features].to_numpy(dtype=float)
    signs = []
    rows = []
    for pos, feature in enumerate(features):
        early_values = early_matrix[:, pos]
        late_values = late_matrix[:, pos]
        early_median = finite_median(early_values, np.nan)
        late_median = finite_median(late_values, np.nan)
        delta = late_median - early_median
        sign = 0 if (not np.isfinite(delta) or abs(delta) < config.direction_delta_min) else int(np.sign(delta))
        signs.append(sign)
        c_d = cohen_d(early_values, late_values)
        c_delta = cliffs_delta(early_values, late_values)
        abs_effect = abs(c_delta) if np.isfinite(c_delta) else abs(c_d)
        mrow = meta[meta["feature_name"] == feature].iloc[0]
        rows.append(
            {
                "direction_id": direction_id,
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "feature_name": feature,
                "channel": mrow["channel"],
                "feature_family": mrow["feature_family"],
                "early_median": early_median,
                "late_median": late_median,
                "delta_median": float(delta) if np.isfinite(delta) else np.nan,
                "direction_sign": int(sign),
                "cohen_d": float(c_d) if np.isfinite(c_d) else np.nan,
                "cliffs_delta": float(c_delta) if np.isfinite(c_delta) else np.nan,
                "abs_effect_size": float(abs_effect) if np.isfinite(abs_effect) else 0.0,
                "physical_meaning": mrow["physical_meaning"],
            }
        )
    direction_signs = np.asarray(signs, dtype=int)
    stability = bootstrap_direction_stability(early_matrix, late_matrix, direction_signs, config)
    out = pd.DataFrame(rows)
    out["direction_stability"] = stability
    out["is_weak_direction"] = (
        (out["direction_sign"].astype(int) == 0)
        | (out["delta_median"].abs() < config.direction_delta_min)
        | (out["abs_effect_size"] < config.effect_size_min)
    )
    out["is_unstable_direction"] = out["direction_stability"] < config.unstable_direction_threshold
    out["enter_M1"] = (out["direction_sign"].astype(int) != 0) & (~out["is_weak_direction"])
    out["enter_M2"] = (
        out["enter_M1"]
        & (out["direction_stability"] >= config.enter_weight_stability_threshold)
        & (out["abs_effect_size"] >= config.effect_size_min)
    )
    out["enter_M3"] = out["enter_M2"] & out["channel"].isin(["rx", "ry"])
    return out, train_mask, val_mask


def compute_redundancy_factors(
    direction_table: pd.DataFrame,
    source_z: pd.DataFrame,
    train_mask: np.ndarray,
    config: WeightedAWRConfig,
) -> pd.DataFrame:
    table = direction_table.copy()
    table["redundancy_factor"] = 0.0
    table["redundancy_notes"] = ""
    candidates = table[table["enter_M2"]].copy()
    if candidates.empty:
        return table
    candidates["rank_score"] = candidates["abs_effect_size"] * candidates["direction_stability"]
    ordered = candidates.sort_values(["rank_score", "feature_name"], ascending=[False, True])
    signed = {}
    for row in ordered.itertuples(index=False):
        feature = str(row.feature_name)
        sign = int(row.direction_sign)
        signed[feature] = sign * source_z.loc[train_mask, feature].to_numpy(dtype=float)
    kept: List[str] = []
    factors = {str(row.feature_name): 1.0 for row in ordered.itertuples(index=False)}
    notes = {str(row.feature_name): "" for row in ordered.itertuples(index=False)}
    for feature in ordered["feature_name"].astype(str).tolist():
        for prev in kept:
            x = signed[feature]
            y = signed[prev]
            ok = np.isfinite(x) & np.isfinite(y)
            corr = 0.0
            if int(ok.sum()) > 5 and np.nanstd(x[ok]) > 1e-12 and np.nanstd(y[ok]) > 1e-12:
                corr = float(np.corrcoef(x[ok], y[ok])[0, 1])
            if abs(corr) > config.redundancy_threshold:
                factors[feature] *= config.redundancy_downweight
                note = f"downweighted_redundant_with_{prev}_r={corr:.3f}"
                notes[feature] = note if not notes[feature] else notes[feature] + ";" + note
        kept.append(feature)
    for feature, factor in factors.items():
        table.loc[table["feature_name"] == feature, "redundancy_factor"] = float(factor)
        table.loc[table["feature_name"] == feature, "redundancy_notes"] = notes[feature]
    return table


def compute_feature_weights(direction_table: pd.DataFrame) -> pd.DataFrame:
    table = direction_table.copy()
    table["raw_weight"] = 0.0
    active = table["enter_M2"].astype(bool)
    table.loc[active, "raw_weight"] = (
        table.loc[active, "abs_effect_size"]
        * table.loc[active, "direction_stability"]
        * table.loc[active, "redundancy_factor"]
    )
    total = float(table.loc[active, "raw_weight"].sum())
    table["normalized_weight"] = 0.0
    if total > 0:
        table.loc[active, "normalized_weight"] = table.loc[active, "raw_weight"] / total
    table["reason_excluded"] = ""
    table.loc[~table["enter_M1"], "reason_excluded"] = "weak_or_zero_direction"
    table.loc[table["enter_M1"] & ~table["enter_M2"], "reason_excluded"] = "unstable_or_small_effect"
    table.loc[table["redundancy_notes"].astype(str) != "", "reason_excluded"] = table.loc[
        table["redundancy_notes"].astype(str) != "", "redundancy_notes"
    ]
    return table[
        [
            "direction_id",
            "source_dataset",
            "target_dataset",
            "feature_name",
            "channel",
            "feature_family",
            "direction_sign",
            "abs_effect_size",
            "direction_stability",
            "redundancy_factor",
            "raw_weight",
            "normalized_weight",
            "enter_M2",
            "enter_M3",
            "reason_excluded",
            "physical_meaning",
        ]
    ]


def signed_matrix(z: pd.DataFrame, table: pd.DataFrame, active_col: str) -> Tuple[np.ndarray, List[str]]:
    active = table[table[active_col].astype(bool)].copy()
    features = active["feature_name"].astype(str).tolist()
    if not features:
        return np.zeros((len(z), 0), dtype=float), []
    signs = active["direction_sign"].to_numpy(dtype=float)
    return z[features].to_numpy(dtype=float) * signs.reshape(1, -1), features


def compute_M0_AWR(
    z_wide: pd.DataFrame,
    state_v2: pd.DataFrame,
    dirs: Dict[str, Path],
) -> Tuple[pd.Series, pd.DataFrame]:
    key = ["dataset", "window_index"]
    existing = state_v2[key + ["AWR"]].rename(columns={"AWR": "AWR_M0_existing"})
    recomputed = z_wide[key].copy()
    recomputed["AWR_M0_recomputed"] = z_wide[STABLE_PLUS_FEATURES].mean(axis=1)
    merged = recomputed.merge(existing, on=key, how="left")
    rows = []
    for dataset, group in merged.groupby("dataset", sort=True):
        x = group["AWR_M0_recomputed"].to_numpy(dtype=float)
        y = group["AWR_M0_existing"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        corr = float(np.corrcoef(x[ok], y[ok])[0, 1]) if int(ok.sum()) > 2 else np.nan
        rows.append(
            {
                "dataset": dataset,
                "windows": int(len(group)),
                "matched_existing_AWR": int(ok.sum()),
                "recomputed_vs_existing_corr": corr,
                "median_abs_diff": float(np.nanmedian(np.abs(x[ok] - y[ok]))) if int(ok.sum()) else np.nan,
                "notes": "AWR_M0 uses existing v2 AWR to preserve stable_plus baseline; recomputed column uses v1 feature list under current z config.",
            }
        )
    diag = pd.DataFrame(rows)
    diag.to_csv(dirs["diagnostics"] / "m0_reproduction_check.csv", index=False, encoding="utf-8-sig")
    return merged["AWR_M0_existing"], diag


def compute_M1_AWR(z: pd.DataFrame, direction_table: pd.DataFrame) -> np.ndarray:
    matrix, _ = signed_matrix(z, direction_table, "enter_M1")
    if matrix.shape[1] == 0:
        return np.zeros(len(z), dtype=float)
    return np.nanmean(matrix, axis=1)


def compute_M2_AWR(z: pd.DataFrame, weight_table: pd.DataFrame) -> np.ndarray:
    active = weight_table[weight_table["enter_M2"].astype(bool)].copy()
    if active.empty or float(active["normalized_weight"].sum()) <= 0:
        return np.zeros(len(z), dtype=float)
    features = active["feature_name"].astype(str).tolist()
    signs = active["direction_sign"].to_numpy(dtype=float)
    weights = active["normalized_weight"].to_numpy(dtype=float)
    matrix = z[features].to_numpy(dtype=float) * signs.reshape(1, -1)
    return np.nansum(matrix * weights.reshape(1, -1), axis=1)


def compute_channel_component(z: pd.DataFrame, weight_table: pd.DataFrame, channel: str) -> np.ndarray:
    active = weight_table[(weight_table["enter_M3"].astype(bool)) & (weight_table["channel"].astype(str) == channel)].copy()
    if active.empty:
        return np.zeros(len(z), dtype=float)
    raw = active["raw_weight"].to_numpy(dtype=float)
    if not np.isfinite(raw).any() or float(np.nansum(raw)) <= 0:
        raw = np.ones(len(active), dtype=float)
    weights = raw / float(np.nansum(raw))
    features = active["feature_name"].astype(str).tolist()
    signs = active["direction_sign"].to_numpy(dtype=float)
    matrix = z[features].to_numpy(dtype=float) * signs.reshape(1, -1)
    return np.nansum(matrix * weights.reshape(1, -1), axis=1)


def roc_auc(y: Iterable[float], score: Iterable[float]) -> float:
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks_sorted = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = ranks_sorted
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: Iterable[float], score: Iterable[float]) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    yy = y[order]
    precision = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((precision * yy).sum() / n_pos)


def spearman_corr(x: Iterable[float], y: Iterable[float]) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return float("nan")
    return float(frame["x"].rank().corr(frame["y"].rank()))


def choose_alpha_m3(
    source_scores: pd.DataFrame,
    val_mask: np.ndarray,
    config: WeightedAWRConfig,
) -> Tuple[float, float, float]:
    if int(val_mask.sum()) < 20:
        val_mask = np.ones(len(source_scores), dtype=bool)
    y = (source_scores.loc[val_mask, "stage"].to_numpy(dtype=int) == 5).astype(int)
    hx = source_scores.loc[val_mask, "Hx_M3"].to_numpy(dtype=float)
    hy = source_scores.loc[val_mask, "Hy_M3"].to_numpy(dtype=float)
    best = (float("-inf"), float("-inf"), -999.0, 0.5)
    steps = int(round(1.0 / config.m3_alpha_grid_step))
    for alpha_x in np.linspace(0.0, 1.0, steps + 1):
        score = alpha_x * hx + (1.0 - alpha_x) * hy
        auc = roc_auc(y, score)
        ap = average_precision(y, score)
        key = (
            auc if np.isfinite(auc) else -1.0,
            ap if np.isfinite(ap) else -1.0,
            -abs(float(alpha_x) - 0.5),
            float(alpha_x),
        )
        if key > best[:4]:
            best = key
    alpha_x = float(best[3])
    return alpha_x, float(1.0 - alpha_x), float(best[0])


def compute_M3_AWR(
    z: pd.DataFrame,
    weight_table: pd.DataFrame,
    alpha_x: float,
    alpha_y: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hx = compute_channel_component(z, weight_table, "rx")
    hy = compute_channel_component(z, weight_table, "ry")
    equal = 0.5 * hx + 0.5 * hy
    weighted = alpha_x * hx + alpha_y * hy
    return hx, hy, equal, weighted


def compute_direction_models(
    z_wide: pd.DataFrame,
    state_v2: pd.DataFrame,
    meta: pd.DataFrame,
    dirs: Dict[str, Path],
    config: WeightedAWRConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    directions = [("Exp1_to_Exp2", "Exp1", "Exp2"), ("Exp2_to_Exp1", "Exp2", "Exp1")]
    direction_tables = []
    weight_tables = []
    awr_frames = []
    channel_frames = []
    split_masks: Dict[str, Dict[str, np.ndarray]] = {}
    m0_series, _ = compute_M0_AWR(z_wide, state_v2, dirs)
    m0_lookup = z_wide[["dataset", "window_index"]].copy()
    m0_lookup["AWR_M0"] = m0_series.to_numpy(dtype=float)

    for direction_id, source_dataset, target_dataset in directions:
        direction_table, train_mask, val_mask = determine_direction_signs(
            direction_id, source_dataset, target_dataset, z_wide, meta, config
        )
        source_z = z_wide[z_wide["dataset"] == source_dataset].sort_values("window_index").reset_index(drop=True)
        direction_table = compute_redundancy_factors(direction_table, source_z, train_mask, config)
        weight_table = compute_feature_weights(direction_table)
        direction_tables.append(direction_table)
        weight_tables.append(weight_table)
        split_masks[direction_id] = {"train": train_mask, "val": val_mask}

        temp_scores = {}
        for dataset in (source_dataset, target_dataset):
            z = z_wide[z_wide["dataset"] == dataset].sort_values("window_index").reset_index(drop=True)
            base = z[
                [
                    "dataset",
                    "window_id",
                    "window_index",
                    "start_cycle",
                    "end_cycle",
                    "center_cycle",
                    "stage",
                    "stage_label",
                ]
            ].copy()
            base["direction_id"] = direction_id
            base["source_dataset"] = source_dataset
            base["target_dataset"] = target_dataset
            base = base.merge(m0_lookup, on=["dataset", "window_index"], how="left")
            base["AWR_M1"] = compute_M1_AWR(z, direction_table)
            base["AWR_M2"] = compute_M2_AWR(z, weight_table)
            hx, hy, equal, weighted_placeholder = compute_M3_AWR(z, weight_table, 0.5, 0.5)
            base["Hx_M3"] = hx
            base["Hy_M3"] = hy
            base["AWR_M3_equal"] = equal
            base["AWR_M3_weighted"] = weighted_placeholder
            temp_scores[dataset] = base

        source_scores = temp_scores[source_dataset].sort_values("window_index").reset_index(drop=True)
        alpha_x, alpha_y, source_val_auc = choose_alpha_m3(source_scores, val_mask, config)
        for dataset, base in temp_scores.items():
            base["alpha_x"] = alpha_x
            base["alpha_y"] = alpha_y
            base["AWR_M3_weighted"] = alpha_x * base["Hx_M3"] + alpha_y * base["Hy_M3"]
            awr_frames.append(base)
            channel_frames.append(
                base[
                    [
                        "dataset",
                        "direction_id",
                        "window_id",
                        "window_index",
                        "center_cycle",
                        "stage_label",
                        "stage",
                        "Hx_M3",
                        "Hy_M3",
                        "AWR_M3_equal",
                        "AWR_M3_weighted",
                        "alpha_x",
                        "alpha_y",
                    ]
                ].rename(columns={"Hx_M3": "Hx", "Hy_M3": "Hy"})
            )
        logging.info(
            "%s: enter_M1=%d enter_M2=%d enter_M3=%d alpha_x=%.2f source_val_auc=%.3f",
            direction_id,
            int(direction_table["enter_M1"].sum()),
            int(direction_table["enter_M2"].sum()),
            int(direction_table["enter_M3"].sum()),
            alpha_x,
            source_val_auc,
        )

    awrs = pd.concat(awr_frames, ignore_index=True)
    channel_scores = pd.concat(channel_frames, ignore_index=True)
    return (
        pd.concat(direction_tables, ignore_index=True),
        pd.concat(weight_tables, ignore_index=True),
        channel_scores,
        awrs,
        split_masks,
    )


def compute_awr_thresholds(
    awrs: pd.DataFrame,
    split_masks: Dict[str, Dict[str, np.ndarray]],
    config: WeightedAWRConfig,
) -> pd.DataFrame:
    rows = []
    for direction_id, group in awrs.groupby("direction_id", sort=True):
        source_dataset = str(group["source_dataset"].iloc[0])
        target_dataset = str(group["target_dataset"].iloc[0])
        source = group[group["dataset"] == source_dataset].sort_values("window_index").reset_index(drop=True)
        target = group[group["dataset"] == target_dataset].sort_values("window_index").reset_index(drop=True)
        val_mask = split_masks[direction_id]["val"]
        if len(val_mask) != len(source) or int(val_mask.sum()) < 20:
            val_mask = np.ones(len(source), dtype=bool)
        for model, col in MODEL_SCORE_COLS.items():
            source_values = source.loc[val_mask, col].to_numpy(dtype=float)
            threshold = finite_quantile(source_values, config.high_awr_threshold_percentile, default=np.nan)
            if not np.isfinite(threshold):
                threshold = finite_quantile(source[col], config.high_awr_threshold_percentile, default=0.0)
            rows.append(
                {
                    "direction_id": direction_id,
                    "model_name": model,
                    "threshold_value": float(threshold),
                    "threshold_source": f"source_validation_p{config.high_awr_threshold_percentile:g}",
                    "source_dataset": source_dataset,
                    "target_dataset": target_dataset,
                    "source_high_rate": float(np.nanmean(source[col].to_numpy(dtype=float) >= threshold)),
                    "target_high_rate": float(np.nanmean(target[col].to_numpy(dtype=float) >= threshold)),
                    "notes": "State-region threshold selected from source validation only; not an alarm threshold.",
                }
            )
    return pd.DataFrame(rows)


def evaluate_bidirectional(
    awrs: pd.DataFrame,
    thresholds: pd.DataFrame,
    state_v2: pd.DataFrame,
    bd_thresholds: pd.DataFrame,
    config: WeightedAWRConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bd_cols = ["dataset", "window_index", config.bd_default_metric]
    bd = state_v2[bd_cols].copy()
    default_bd = bd_thresholds[bd_thresholds["bd_metric"] == config.bd_default_metric]
    bd_lookup = {str(row.dataset): float(row.BD_major_threshold) for row in default_bd.itertuples(index=False)}
    rows = []
    for direction_id, group in awrs.groupby("direction_id", sort=True):
        source_dataset = str(group["source_dataset"].iloc[0])
        target_dataset = str(group["target_dataset"].iloc[0])
        target = group[group["dataset"] == target_dataset].sort_values("window_index").reset_index(drop=True)
        target = target.merge(bd, on=["dataset", "window_index"], how="left")
        stage = target["stage"].to_numpy(dtype=int)
        y = (stage == 5).astype(int)
        for model, col in MODEL_SCORE_COLS.items():
            score = target[col].to_numpy(dtype=float)
            th_row = thresholds[(thresholds["direction_id"] == direction_id) & (thresholds["model_name"] == model)].iloc[0]
            threshold = float(th_row["threshold_value"])
            stage1 = score[stage == 1]
            stage5 = score[stage == 5]
            high = score >= threshold
            bd_high = target[config.bd_default_metric].to_numpy(dtype=float) >= bd_lookup.get(target_dataset, np.inf)
            stage5_mask = stage == 5
            occupancy = (
                float(np.mean(high[stage5_mask] & bd_high[stage5_mask])) if int(stage5_mask.sum()) else np.nan
            )
            rows.append(
                {
                    "model_name": model,
                    "direction_id": direction_id,
                    "source_dataset": source_dataset,
                    "target_dataset": target_dataset,
                    "target_AUROC": roc_auc(y, score),
                    "target_AUPRC": average_precision(y, score),
                    "target_AUPRC_baseline": float(np.nanmean(y)),
                    "target_Spearman_stage_AWR": spearman_corr(stage, score),
                    "Stage1_median_AWR": finite_median(stage1, np.nan),
                    "Stage5_median_AWR": finite_median(stage5, np.nan),
                    "ScoreGap": finite_median(stage5, np.nan) - finite_median(stage1, np.nan),
                    "Stage5_high_AWR_rate": float(np.mean(high[stage5_mask])) if int(stage5_mask.sum()) else np.nan,
                    "Stage5_high_AWR_high_BD_occupancy": occupancy,
                    "notes": "Stage5 is used only as a late-state proxy label for evaluation.",
                }
            )
    summary = pd.DataFrame(rows)
    comp_rows = []
    interpretability = {
        "M0": ("baseline", "stable_plus mean-z reference"),
        "M1": ("direction-transparent", "direction-corrected contrast"),
        "M2": ("weighted-transparent", "effect/stability weighted contrast"),
        "M3_equal": ("channel-constrained", "default interpretable Hx/Hy fusion"),
        "M3_weighted": ("channel-constrained candidate", "source-validation channel fusion candidate"),
    }
    for model, group in summary.groupby("model_name", sort=False):
        comp_rows.append(
            {
                "model_name": model,
                "mean_AUROC": float(np.nanmean(group["target_AUROC"])),
                "worst_AUROC": float(np.nanmin(group["target_AUROC"])),
                "mean_AUPRC": float(np.nanmean(group["target_AUPRC"])),
                "worst_AUPRC": float(np.nanmin(group["target_AUPRC"])),
                "mean_Spearman": float(np.nanmean(group["target_Spearman_stage_AWR"])),
                "worst_Spearman": float(np.nanmin(group["target_Spearman_stage_AWR"])),
                "interpretability_level": interpretability[model][0],
                "recommended_role": interpretability[model][1],
            }
        )
    comparison = pd.DataFrame(comp_rows)
    return summary, comparison


def select_model(comparison: pd.DataFrame, weight_table: pd.DataFrame) -> Dict[str, object]:
    comp = comparison.set_index("model_name")
    m0_worst_auc = float(comp.loc["M0", "worst_AUROC"])
    m0_worst_ap = float(comp.loc["M0", "worst_AUPRC"])
    selected = "M0"
    reason = "Weighted variants did not satisfy the conservative worst-direction guardrail against M0."
    strengths = ["Preserves existing stable_plus baseline."]
    weaknesses = ["Direction and channel contributions remain implicit."]
    if "M3_equal" in comp.index:
        m3_auc = float(comp.loc["M3_equal", "worst_AUROC"])
        m3_ap = float(comp.loc["M3_equal", "worst_AUPRC"])
        m2_auc = float(comp.loc["M2", "worst_AUROC"])
        if m3_auc >= m0_worst_auc - 0.01 and m3_ap >= m0_worst_ap - 0.01 and m3_auc >= m2_auc - 0.02:
            selected = "M3_equal"
            reason = "M3_equal keeps worst-direction performance close to M0 while exposing Hx/Hy channel contributions."
            strengths = [
                "Direction signs and feature weights are explicit.",
                "Hx and Hy provide channel-level traceability.",
                "Equal fusion avoids overusing one source-validation direction.",
            ]
            weaknesses = ["It is still a signal-layer state score and needs external physical-loop validation."]
    if selected != "M3_equal" and "M2" in comp.index:
        m2_auc = float(comp.loc["M2", "worst_AUROC"])
        m2_ap = float(comp.loc["M2", "worst_AUPRC"])
        if m2_auc >= m0_worst_auc - 0.01 and m2_ap >= m0_worst_ap - 0.01:
            selected = "M2"
            reason = "M2 meets the conservative M0 guardrail with explicit effect/stability feature weighting."
            strengths = ["Feature direction, effect size, stability, and weight are explicit."]
            weaknesses = ["Channel contribution is less direct than M3."]
    if selected == "M3_equal" and "M3_weighted" in comp.index:
        weighted_gain = float(comp.loc["M3_weighted", "worst_AUROC"] - comp.loc["M3_equal", "worst_AUROC"])
        if weighted_gain > 0.03:
            selected = "M3_weighted"
            reason = "M3_weighted improves worst-direction AUROC by more than 0.03 while retaining nonnegative Hx/Hy fusion."
            strengths = ["Channel weights remain source-validation constrained.", "Hx/Hy decomposition is retained."]
            weaknesses = ["Interpretation is slightly less simple than equal Hx/Hy fusion."]
    top_weight = 0.0
    if not weight_table.empty and "normalized_weight" in weight_table.columns:
        top_weight = float(weight_table["normalized_weight"].max())
    return {
        "selected_model": selected,
        "reason": reason,
        "comparison_to_M0": {
            "M0_worst_AUROC": m0_worst_auc,
            "M0_worst_AUPRC": m0_worst_ap,
            "selected_worst_AUROC": float(comp.loc[selected, "worst_AUROC"]),
            "selected_worst_AUPRC": float(comp.loc[selected, "worst_AUPRC"]),
        },
        "strengths": strengths,
        "weaknesses": weaknesses,
        "top_feature_weight": top_weight,
        "next_steps": [
            "Use FEM/contact morphology/debris observations for external physical-loop checks.",
            "Keep Stage5 as a late-state proxy label rather than a physical endpoint.",
        ],
    }


def add_selected_scores(awrs: pd.DataFrame, selected_model: str, config: WeightedAWRConfig) -> pd.DataFrame:
    out = awrs.copy()
    out["AWR_M3_selected"] = np.where(
        selected_model == "M3_weighted", out["AWR_M3_weighted"], out["AWR_M3_equal"]
    )
    col = MODEL_SCORE_COLS[selected_model]
    out["AWR_selected_model"] = out[col]
    out["AWR_smooth_selected"] = np.nan
    for _, idx in out.groupby(["direction_id", "dataset"], sort=True).groups.items():
        idx = list(idx)
        ordered = out.loc[idx].sort_values("window_index")
        smooth = causal_rolling_median(ordered["AWR_selected_model"], config.awr_smoothing_window)
        out.loc[ordered.index, "AWR_smooth_selected"] = smooth
    return out


def causal_rolling_median(values: Iterable[float], window: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=window, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def causal_rolling_mean(values: Iterable[float], window: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=window, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def causal_rolling_robust_std(values: Iterable[float], window: int, eps: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros(len(arr), dtype=float)
    for idx in range(len(arr)):
        start = max(0, idx - window + 1)
        segment = finite_values(arr[start : idx + 1])
        if segment.size < 3:
            out[idx] = 0.0
            continue
        med = np.nanmedian(segment)
        mad = np.nanmedian(np.abs(segment - med)) * 1.4826
        out[idx] = float(mad if mad >= eps else np.nanstd(segment))
    return out


def z_plus(values: np.ndarray, ref_values: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ref = finite_values(ref_values)
    if ref.size == 0:
        return np.zeros_like(values, dtype=float)
    med = np.nanmedian(ref)
    width = max(robust_iqr(ref, default=0.0), eps)
    z = (values - med) / width
    z[~np.isfinite(z)] = 0.0
    return np.maximum(z, 0.0)


def compute_rs_for_group(values: np.ndarray, horizons: Tuple[int, int, int]) -> Dict[int, np.ndarray]:
    out = {}
    for horizon in horizons:
        window = horizon
        rs = np.full(len(values), np.nan, dtype=float)
        for pos in range(len(values)):
            now_start = pos - window + 1
            prev_start = pos - horizon - window + 1
            prev_end = pos - horizon
            if now_start < 0 or prev_start < 0 or prev_end < 0:
                continue
            now_segment = values[now_start : pos + 1]
            prev_segment = values[prev_start : prev_end + 1]
            if len(now_segment) < window or len(prev_segment) < window:
                continue
            rs[pos] = (np.nanmedian(now_segment) - np.nanmedian(prev_segment)) / float(horizon)
        out[horizon] = rs
    return out


def compute_state_metrics_for_selected_AWR(
    awrs: pd.DataFrame,
    state_v2: pd.DataFrame,
    awr_thresholds: pd.DataFrame,
    bd_thresholds: pd.DataFrame,
    boundaries: pd.DataFrame,
    selected_model: str,
    config: WeightedAWRConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bd_cols = [
        "dataset",
        "window_index",
        "BDx_v2",
        "BDy_v2",
        "BDshape_v2",
        "BDall_xy_v2",
    ]
    bd_cols = [col for col in bd_cols if col in state_v2.columns]
    frame = awrs[
        [
            "dataset",
            "direction_id",
            "source_dataset",
            "target_dataset",
            "window_id",
            "window_index",
            "start_cycle",
            "end_cycle",
            "center_cycle",
            "stage",
            "stage_label",
            "AWR_selected_model",
            "AWR_smooth_selected",
        ]
    ].rename(columns={"AWR_selected_model": "AWR", "AWR_smooth_selected": "AWR_smooth"})
    frame["model_name"] = selected_model
    frame = frame.merge(state_v2[bd_cols], on=["dataset", "window_index"], how="left")
    for col in ("BDx_v2", "BDy_v2", "BDshape_v2", "BDall_xy_v2"):
        if col not in frame.columns:
            frame[col] = np.nan

    rs_threshold_rows = []
    tes_threshold_rows = []
    all_events = []
    event_id = 1
    frame["RS_trend20"] = np.nan
    frame["RS_trend50"] = np.nan
    frame["RS_trend100"] = np.nan
    frame["RS_trend50_rising"] = 0
    frame["RS_trend50_rising_run"] = 0
    frame["TES"] = np.nan
    frame["TES_smooth"] = np.nan
    frame["is_TES_event"] = 0
    frame["TES_event_neighborhood"] = 0
    frame["TES_high_confidence_neighborhood"] = 0

    for (direction_id, dataset), idx in frame.groupby(["direction_id", "dataset"], sort=True).groups.items():
        ordered = frame.loc[list(idx)].sort_values("window_index")
        ordered_idx = ordered.index.tolist()
        awr_smooth = ordered["AWR_smooth"].to_numpy(dtype=float)
        rs = compute_rs_for_group(awr_smooth, config.rs_horizons)
        for horizon, values in rs.items():
            frame.loc[ordered_idx, f"RS_trend{horizon}"] = values

        baseline = ordered["end_cycle"].to_numpy(dtype=float) <= config.baseline_cycles
        if int(baseline.sum()) < 10:
            baseline = np.zeros(len(ordered), dtype=bool)
            baseline[: min(100, len(ordered))] = True
        bd = ordered["BDall_xy_v2"].to_numpy(dtype=float)
        shape = ordered["BDshape_v2"].to_numpy(dtype=float)
        awr_vol = causal_rolling_robust_std(awr_smooth, config.tes_window, config.eps)
        bd_delta = np.maximum(bd - causal_rolling_median(bd, config.tes_window), 0.0)
        shape_jump = np.maximum(shape - causal_rolling_median(shape, config.tes_window), 0.0)
        tes = (
            config.tes_weight_awr_vol * z_plus(awr_vol, awr_vol[baseline], config.eps)
            + config.tes_weight_bd_jump * z_plus(bd_delta, bd_delta[baseline], config.eps)
            + config.tes_weight_shape_jump * z_plus(shape_jump, shape_jump[baseline], config.eps)
        )
        frame.loc[ordered_idx, "TES"] = tes
        frame.loc[ordered_idx, "TES_smooth"] = causal_rolling_mean(tes, config.tes_smoothing_window)

    for direction_id, group in frame.groupby("direction_id", sort=True):
        source_dataset = str(group["source_dataset"].iloc[0])
        source = group[group["dataset"] == source_dataset].sort_values("window_index").reset_index(drop=True)
        source_base = source["end_cycle"].to_numpy(dtype=float) <= config.baseline_cycles
        if int(source_base.sum()) < 10:
            source_base = np.zeros(len(source), dtype=bool)
            source_base[: min(100, len(source))] = True
        rs50_base = source.loc[source_base, "RS_trend50"].to_numpy(dtype=float)
        rs_threshold = max(finite_quantile(rs50_base, config.rs_threshold_percentile, default=0.0), config.rs_min_threshold)
        tes_base = source.loc[source_base, "TES_smooth"].to_numpy(dtype=float)
        tes_threshold_raw = finite_quantile(tes_base, config.tes_threshold_percentile, default=0.0)
        tes_prominence_raw = finite_quantile(tes_base, config.tes_prominence_percentile, default=0.0)
        tes_threshold = max(tes_threshold_raw, config.tes_threshold_floor)
        tes_prominence = max(tes_prominence_raw, config.tes_prominence_floor)
        rs_threshold_rows.append(
            {
                "direction_id": direction_id,
                "source_dataset": source_dataset,
                "horizon": 50,
                "RS_threshold": float(rs_threshold),
                "threshold_source": f"source_baseline_p{config.rs_threshold_percentile:g}_with_floor",
            }
        )
        tes_threshold_rows.append(
            {
                "direction_id": direction_id,
                "source_dataset": source_dataset,
                "TES_threshold": float(tes_threshold),
                "TES_threshold_raw_percentile": float(tes_threshold_raw),
                "TES_threshold_source": f"source_baseline_p{config.tes_threshold_percentile:g}_with_floor",
                "TES_prominence_threshold": float(tes_prominence),
                "TES_prominence_raw_percentile": float(tes_prominence_raw),
                "TES_prominence_source": f"source_baseline_p{config.tes_prominence_percentile:g}_with_floor",
                "min_event_gap_cycles": int(config.min_event_gap_cycles),
                "min_event_gap_windows": int(max(1, round(config.min_event_gap_cycles / config.stride))),
            }
        )

        direction_idx = frame[frame["direction_id"] == direction_id].index
        ordered_all = frame.loc[direction_idx].sort_values(["dataset", "window_index"])
        for dataset, sub in ordered_all.groupby("dataset", sort=True):
            idx_sub = sub.sort_values("window_index").index.tolist()
            values = frame.loc[idx_sub, "RS_trend50"].to_numpy(dtype=float)
            run = 0
            runs = []
            rising = []
            for value in values:
                if np.isfinite(value) and value > rs_threshold:
                    run += 1
                else:
                    run = 0
                runs.append(run)
                rising.append(int(run >= config.rs_rising_consecutive_windows))
            frame.loc[idx_sub, "RS_trend50_rising_run"] = runs
            frame.loc[idx_sub, "RS_trend50_rising"] = rising

            tes_values = frame.loc[idx_sub, "TES_smooth"].to_numpy(dtype=float)
            tes_values = np.where(np.isfinite(tes_values), tes_values, 0.0)
            peaks, props = find_peaks(
                tes_values,
                height=float(tes_threshold),
                prominence=float(tes_prominence),
                distance=int(max(1, round(config.min_event_gap_cycles / config.stride))),
            )
            ds_boundaries = boundaries[boundaries["dataset"].astype(str) == str(dataset)] if not boundaries.empty else pd.DataFrame()
            awr_th = float(
                awr_thresholds[
                    (awr_thresholds["direction_id"] == direction_id)
                    & (awr_thresholds["model_name"] == selected_model)
                ].iloc[0]["threshold_value"]
            )
            for prop_pos, peak_pos in enumerate(peaks):
                row = frame.loc[idx_sub[int(peak_pos)]]
                peak_window = int(row["window_index"])
                near_mask = (frame.loc[idx_sub, "window_index"].astype(int) - peak_window).abs() <= config.event_neighborhood_windows
                near_idx = list(np.asarray(idx_sub)[near_mask.to_numpy()])
                high_awr = bool(float(row["AWR"]) >= awr_th)
                rising = bool((frame.loc[near_idx, "RS_trend50"].to_numpy(dtype=float) > rs_threshold).any())
                if high_awr and rising:
                    event_type = "TES_in_high_AWR_and_rising"
                elif high_awr:
                    event_type = "TES_in_high_AWR"
                elif rising:
                    event_type = "TES_with_rising"
                else:
                    event_type = "TES_only"
                high_conf = event_type != "TES_only"
                nearest_stage_boundary = ""
                delay_to_boundary = np.nan
                matched_boundary = False
                if not ds_boundaries.empty:
                    distances = (ds_boundaries["boundary_cycle"] - float(row["center_cycle"])).abs()
                    nearest = ds_boundaries.loc[distances.idxmin()]
                    delay_to_boundary = float(row["center_cycle"] - nearest["boundary_cycle"])
                    matched_boundary = bool(abs(delay_to_boundary) <= config.boundary_match_tolerance_cycles)
                    nearest_stage_boundary = f"{nearest['boundary_label']}@{int(nearest['boundary_cycle'])}"
                all_events.append(
                    {
                        "event_id": int(event_id),
                        "direction_id": direction_id,
                        "dataset": dataset,
                        "model_name": selected_model,
                        "peak_cycle": float(row["center_cycle"]),
                        "peak_window_index": peak_window,
                        "TES_peak": float(row["TES_smooth"]),
                        "TES_raw_at_peak": float(row["TES"]),
                        "TES_prominence": float(props["prominences"][prop_pos]),
                        "event_type": event_type,
                        "high_confidence_event": bool(high_conf),
                        "AWR_at_peak": float(row["AWR"]),
                        "BDall_at_peak_v2": float(row["BDall_xy_v2"]),
                        "BDx_at_peak_v2": float(row["BDx_v2"]),
                        "BDy_at_peak_v2": float(row["BDy_v2"]),
                        "BDshape_at_peak_v2": float(row["BDshape_v2"]),
                        "RS_trend50_at_peak": float(row["RS_trend50"]) if np.isfinite(row["RS_trend50"]) else np.nan,
                        "nearest_stage_boundary": nearest_stage_boundary,
                        "delay_to_boundary": delay_to_boundary,
                        "matched_boundary": bool(matched_boundary),
                    }
                )
                event_id += 1
                frame.loc[idx_sub[int(peak_pos)], "is_TES_event"] = 1
                frame.loc[near_idx, "TES_event_neighborhood"] = 1
                if high_conf:
                    frame.loc[near_idx, "TES_high_confidence_neighborhood"] = 1

    default_bd = bd_thresholds[bd_thresholds["bd_metric"] == config.bd_default_metric]
    bd_lookup = {str(row.dataset): float(row.BD_major_threshold) for row in default_bd.itertuples(index=False)}
    regions = []
    interpretations = []
    for row in frame.itertuples(index=False):
        awr_th = float(
            awr_thresholds[
                (awr_thresholds["direction_id"] == getattr(row, "direction_id"))
                & (awr_thresholds["model_name"] == selected_model)
            ].iloc[0]["threshold_value"]
        )
        awr_high = bool(float(getattr(row, "AWR")) >= awr_th)
        bd_high = bool(float(getattr(row, config.bd_default_metric)) >= bd_lookup.get(str(getattr(row, "dataset")), np.inf))
        if awr_high and bd_high:
            region = "high_AWR_high_BD"
        elif awr_high:
            region = "high_AWR_low_BD"
        elif bd_high:
            region = "low_AWR_high_BD"
        else:
            region = "low_AWR_low_BD"
        if bool(getattr(row, "TES_event_neighborhood")):
            interp = "transition_event_neighborhood"
        elif awr_high and bool(getattr(row, "RS_trend50_rising")):
            interp = "high_AWR_rising"
        elif awr_high and bd_high:
            interp = "high_AWR_high_BD_state"
        elif awr_high:
            interp = "high_AWR_low_BD_state"
        elif bd_high:
            interp = "low_AWR_high_BD_state"
        else:
            interp = "initial_like_state"
        regions.append(region)
        interpretations.append(interp)
    frame["state_region"] = regions
    frame["state_interpretation"] = interpretations

    events = pd.DataFrame(all_events)
    stage_summary = summarize_stage_state(frame)
    occupancy = summarize_occupancy(frame)
    event_eval = evaluate_events(events, boundaries, frame, config)
    return (
        frame,
        stage_summary,
        events,
        event_eval,
        occupancy,
        pd.DataFrame(rs_threshold_rows),
        pd.DataFrame(tes_threshold_rows),
    )


def summarize_stage_state(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["AWR", "AWR_smooth", "BDall_xy_v2", "BDx_v2", "BDy_v2", "BDshape_v2", "RS_trend50", "TES", "TES_smooth"]
    for key, group in frame.groupby(["direction_id", "dataset", "model_name", "stage"], sort=True):
        direction_id, dataset, model_name, stage = key
        row = {
            "direction_id": direction_id,
            "dataset": dataset,
            "model_name": model_name,
            "stage": int(stage),
            "stage_label": f"Stage {int(stage)}",
            "windows": int(len(group)),
            "cycle_start": float(np.nanmin(group["start_cycle"])),
            "cycle_end": float(np.nanmax(group["end_cycle"])),
            "TES_event_count": int(group["is_TES_event"].sum()),
            "TES_high_confidence_event_count": int(
                ((group["is_TES_event"] == 1) & (group["TES_high_confidence_neighborhood"] == 1)).sum()
            ),
        }
        for metric in metrics:
            row[f"{metric}_median"] = finite_median(group[metric], np.nan)
            row[f"{metric}_P10"] = finite_quantile(group[metric], 10, np.nan)
            row[f"{metric}_P90"] = finite_quantile(group[metric], 90, np.nan)
        for region in REGION_COLORS:
            row[f"{region}_count"] = int((group["state_region"] == region).sum())
            row[f"{region}_rate"] = float((group["state_region"] == region).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_occupancy(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (direction_id, dataset, stage), group in frame.groupby(["direction_id", "dataset", "stage"], sort=True):
        n = len(group)
        for region in REGION_COLORS:
            count = int((group["state_region"] == region).sum())
            rows.append(
                {
                    "direction_id": direction_id,
                    "dataset": dataset,
                    "group_type": "stage",
                    "group_value": int(stage),
                    "state_region": region,
                    "count": count,
                    "rate": float(count / n) if n else 0.0,
                    "cycle_start": float(np.nanmin(group["start_cycle"])),
                    "cycle_end": float(np.nanmax(group["end_cycle"])),
                }
            )
    return pd.DataFrame(rows)


def evaluate_events(events: pd.DataFrame, boundaries: pd.DataFrame, frame: pd.DataFrame, config: WeightedAWRConfig) -> pd.DataFrame:
    rows = []
    for (direction_id, dataset), group in frame.groupby(["direction_id", "dataset"], sort=True):
        ev = events[(events["direction_id"].astype(str) == str(direction_id)) & (events["dataset"].astype(str) == str(dataset))] if not events.empty else pd.DataFrame()
        bd = boundaries[boundaries["dataset"].astype(str) == str(dataset)] if not boundaries.empty else pd.DataFrame()
        matched_boundary_count = 0
        delays = []
        if not ev.empty and not bd.empty:
            for boundary in bd.itertuples(index=False):
                distances = (ev["peak_cycle"] - float(getattr(boundary, "boundary_cycle"))).abs()
                nearest = ev.loc[distances.idxmin()]
                delay = float(nearest["peak_cycle"] - float(getattr(boundary, "boundary_cycle")))
                if abs(delay) <= config.boundary_match_tolerance_cycles:
                    matched_boundary_count += 1
                    delays.append(delay)
        matched_events = int(ev["matched_boundary"].sum()) if not ev.empty and "matched_boundary" in ev.columns else 0
        high_conf = ev[ev["high_confidence_event"].astype(bool)] if not ev.empty else pd.DataFrame()
        rows.append(
            {
                "direction_id": direction_id,
                "dataset": dataset,
                "event_count": int(len(ev)),
                "matched_boundary_count": int(matched_events),
                "EventRecall": float(matched_boundary_count / len(bd)) if len(bd) else np.nan,
                "FalseEvents": int(len(ev) - matched_events),
                "DetectionDelay_median": finite_median(delays, np.nan),
                "DetectionDelay_mean": float(np.nanmean(delays)) if delays else np.nan,
                "high_confidence_event_count": int(len(high_conf)),
                "high_confidence_false_events": int((~high_conf["matched_boundary"].astype(bool)).sum()) if not high_conf.empty else 0,
                "stage_boundary_count": int(len(bd)),
            }
        )
    if rows:
        df = pd.DataFrame(rows)
        for direction_id, group in df.groupby("direction_id", sort=True):
            rows.append(
                {
                    "direction_id": direction_id,
                    "dataset": "ALL",
                    "event_count": int(group["event_count"].sum()),
                    "matched_boundary_count": int(group["matched_boundary_count"].sum()),
                    "EventRecall": finite_median(group["EventRecall"], np.nan),
                    "FalseEvents": int(group["FalseEvents"].sum()),
                    "DetectionDelay_median": finite_median(group["DetectionDelay_median"], np.nan),
                    "DetectionDelay_mean": finite_median(group["DetectionDelay_mean"], np.nan),
                    "high_confidence_event_count": int(group["high_confidence_event_count"].sum()),
                    "high_confidence_false_events": int(group["high_confidence_false_events"].sum()),
                    "stage_boundary_count": int(group["stage_boundary_count"].sum()),
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "No rows."
    shown = frame.head(max_rows).copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4g}")
        else:
            shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(col) for col in shown.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in shown.itertuples(index=False):
        cells = [str(value).replace("|", "/") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    if len(frame) > max_rows:
        lines.append(f"\nShowing first {max_rows} of {len(frame)} rows.")
    return "\n".join(lines)


def add_boundaries(ax: plt.Axes, boundaries: pd.DataFrame, dataset: str) -> None:
    ds = boundaries[boundaries["dataset"].astype(str) == str(dataset)] if not boundaries.empty else pd.DataFrame()
    for row in ds.itertuples(index=False):
        ax.axvline(float(getattr(row, "boundary_cycle")), color="#999999", linestyle="--", linewidth=0.8, alpha=0.55)


def plot_model_comparison(summary: pd.DataFrame, comparison: pd.DataFrame, figures_dir: Path) -> None:
    models = ["M0", "M1", "M2", "M3_equal", "M3_weighted"]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for ax, metric, title in zip(axes, ["target_AUROC", "target_AUPRC"], ["AUROC", "AUPRC"]):
        means = [float(summary[summary["model_name"] == model][metric].mean()) for model in models]
        worst = [float(summary[summary["model_name"] == model][metric].min()) for model in models]
        ax.bar(x - 0.18, means, width=0.35, color="#5a84b8", label="Mean")
        ax.bar(x + 0.18, worst, width=0.35, color="#c76d2a", label="Worst direction")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Bidirectional Model Comparison")
    save_figure(fig, figures_dir / "fig_model_comparison_AUROC_AUPRC")


def plot_awrs_timeseries(awrs: pd.DataFrame, boundaries: pd.DataFrame, figures_dir: Path) -> None:
    targets = {
        "Exp1": "Exp2_to_Exp1",
        "Exp2": "Exp1_to_Exp2",
    }
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=False)
    for ax, (dataset, direction_id) in zip(axes, targets.items()):
        sub = awrs[(awrs["dataset"] == dataset) & (awrs["direction_id"] == direction_id)].sort_values("window_index")
        for model, col in MODEL_SCORE_COLS.items():
            ax.plot(sub["center_cycle"], sub[col], linewidth=1.0, label=model, color=MODEL_COLORS[model], alpha=0.9)
        add_boundaries(ax, boundaries, dataset)
        ax.set_title(f"{dataset}: M0/M1/M2/M3 AWR trajectories")
        ax.set_ylabel("AWR score")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Cycle")
    axes[0].legend(ncol=5, frameon=False, fontsize=8)
    save_figure(fig, figures_dir / "fig_AWR_M0_M1_M2_M3_timeseries")


def plot_feature_directions(direction_table: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.0), sharex=False)
    for ax, (direction_id, group) in zip(axes, direction_table.groupby("direction_id", sort=True)):
        top = group.reindex(group["abs_effect_size"].sort_values(ascending=False).index).head(18)
        colors = np.where(top["direction_sign"].to_numpy(dtype=int) > 0, "#46a67a", "#b64b5a")
        ax.bar(np.arange(len(top)), top["abs_effect_size"], color=colors, alpha=0.85)
        ax.plot(np.arange(len(top)), top["direction_stability"], color="#222222", marker="o", linewidth=1.0, label="Direction stability")
        ax.axhline(0.70, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_xticks(np.arange(len(top)))
        ax.set_xticklabels(top["feature_name"], rotation=55, ha="right", fontsize=8)
        ax.set_ylim(0, max(1.05, float(top[["abs_effect_size", "direction_stability"]].to_numpy().max()) + 0.05))
        ax.set_title(direction_id)
        ax.set_ylabel("Effect / stability")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    save_figure(fig, figures_dir / "fig_feature_direction_stability")


def plot_feature_weights(weight_table: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.0), sharex=False)
    for ax, (direction_id, group) in zip(axes, weight_table.groupby("direction_id", sort=True)):
        top = group[group["normalized_weight"] > 0].sort_values("normalized_weight", ascending=False).head(18)
        colors = top["channel"].map({"rx": "#3b6ea8", "ry": "#c76d2a", "rs": "#7c8a99"}).fillna("#999999")
        ax.bar(np.arange(len(top)), top["normalized_weight"], color=colors)
        ax.set_xticks(np.arange(len(top)))
        ax.set_xticklabels(top["feature_name"], rotation=55, ha="right", fontsize=8)
        ax.set_title(direction_id)
        ax.set_ylabel("Normalized weight")
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "fig_feature_weight_contribution")


def plot_channel_scores(channel_scores: pd.DataFrame, boundaries: pd.DataFrame, figures_dir: Path) -> None:
    targets = {"Exp1": "Exp2_to_Exp1", "Exp2": "Exp1_to_Exp2"}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.5), sharex=False)
    for ax, (dataset, direction_id) in zip(axes, targets.items()):
        sub = channel_scores[(channel_scores["dataset"] == dataset) & (channel_scores["direction_id"] == direction_id)].sort_values("window_index")
        ax.plot(sub["center_cycle"], sub["Hx"], color="#3b6ea8", linewidth=1.1, label="Hx")
        ax.plot(sub["center_cycle"], sub["Hy"], color="#c76d2a", linewidth=1.1, label="Hy")
        ax.plot(sub["center_cycle"], sub["AWR_M3_equal"], color="#222222", linewidth=1.0, alpha=0.75, label="M3 equal")
        add_boundaries(ax, boundaries, dataset)
        ax.set_title(f"{dataset}: channel scores")
        ax.set_ylabel("Channel score")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Cycle")
    save_figure(fig, figures_dir / "fig_channel_scores_Hx_Hy")


def plot_state_map(state_frame: pd.DataFrame, figures_dir: Path) -> None:
    targets = {"Exp1": "Exp2_to_Exp1", "Exp2": "Exp1_to_Exp2"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=False, sharey=False)
    for ax, (dataset, direction_id) in zip(axes, targets.items()):
        sub = state_frame[(state_frame["dataset"] == dataset) & (state_frame["direction_id"] == direction_id)]
        colors = sub["state_region"].map(REGION_COLORS).fillna("#999999")
        ax.scatter(sub["BDall_xy_v2"], sub["AWR"], c=colors, s=8, alpha=0.60, linewidths=0)
        ax.set_title(f"{dataset}: AWR-BD state map")
        ax.set_xlabel("BDall_xy_v2")
        ax.set_ylabel("Selected AWR")
        ax.grid(alpha=0.25)
    save_figure(fig, figures_dir / "fig_AWR_BD_state_map_weighted")


def plot_occupancy(occupancy: pd.DataFrame, figures_dir: Path) -> None:
    targets = {"Exp1": "Exp2_to_Exp1", "Exp2": "Exp1_to_Exp2"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, (dataset, direction_id) in zip(axes, targets.items()):
        sub = occupancy[(occupancy["dataset"] == dataset) & (occupancy["direction_id"] == direction_id)]
        pivot = sub.pivot_table(index="group_value", columns="state_region", values="rate", aggfunc="sum").fillna(0.0)
        bottom = np.zeros(len(pivot), dtype=float)
        x = np.arange(len(pivot))
        for region, color in REGION_COLORS.items():
            values = pivot[region].to_numpy(dtype=float) if region in pivot.columns else np.zeros(len(pivot))
            ax.bar(x, values, bottom=bottom, color=color, label=region)
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{int(v)}" for v in pivot.index])
        ax.set_title(f"{dataset}: state occupancy")
        ax.set_xlabel("Stage proxy")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Rate")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.25), ncol=2)
    save_figure(fig, figures_dir / "fig_state_region_occupancy_weighted")


def plot_tes_alignment(state_frame: pd.DataFrame, events: pd.DataFrame, boundaries: pd.DataFrame, figures_dir: Path) -> None:
    targets = {"Exp1": "Exp2_to_Exp1", "Exp2": "Exp1_to_Exp2"}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.7), sharex=False)
    for ax, (dataset, direction_id) in zip(axes, targets.items()):
        sub = state_frame[(state_frame["dataset"] == dataset) & (state_frame["direction_id"] == direction_id)].sort_values("window_index")
        ax.plot(sub["center_cycle"], sub["TES_smooth"], color="#5a6fb0", linewidth=1.0, label="TES smooth")
        ev = events[(events["dataset"].astype(str) == dataset) & (events["direction_id"].astype(str) == direction_id)] if not events.empty else pd.DataFrame()
        if not ev.empty:
            high = ev[ev["high_confidence_event"].astype(bool)]
            ax.scatter(ev["peak_cycle"], ev["TES_peak"], s=30, marker="o", color="#777777", label="TES event")
            if not high.empty:
                ax.scatter(high["peak_cycle"], high["TES_peak"], s=42, marker="D", color="#b64b5a", label="High-confidence")
        add_boundaries(ax, boundaries, dataset)
        ax.set_title(f"{dataset}: transition event score")
        ax.set_ylabel("TES")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Cycle")
    save_figure(fig, figures_dir / "fig_TES_event_alignment_weighted")


def plot_m0_vs_selected(summary: pd.DataFrame, state_frame: pd.DataFrame, event_eval: pd.DataFrame, selected_model: str, figures_dir: Path) -> None:
    rows = []
    for model in ("M0", selected_model):
        sub = summary[summary["model_name"] == model]
        rows.append({"metric": "AUROC", "model": model, "value": float(sub["target_AUROC"].mean())})
        rows.append({"metric": "AUPRC", "model": model, "value": float(sub["target_AUPRC"].mean())})
        rows.append(
            {
                "metric": "Stage5 high AWR+BD",
                "model": model,
                "value": float(sub["Stage5_high_AWR_high_BD_occupancy"].mean()),
            }
        )
    ev_all = event_eval[event_eval["dataset"].astype(str) == "ALL"] if not event_eval.empty else pd.DataFrame()
    rows.append({"metric": "TES event count", "model": selected_model, "value": float(ev_all["event_count"].sum()) if not ev_all.empty else 0.0})
    rows.append({"metric": "FalseEvents", "model": selected_model, "value": float(ev_all["FalseEvents"].sum()) if not ev_all.empty else 0.0})
    plot_df = pd.DataFrame(rows)
    metrics = plot_df["metric"].drop_duplicates().tolist()
    models = plot_df["model"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(metrics))
    width = 0.35
    for pos, model in enumerate(models):
        vals = []
        for metric in metrics:
            sub = plot_df[(plot_df["metric"] == metric) & (plot_df["model"] == model)]
            vals.append(float(sub["value"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (pos - (len(models) - 1) / 2) * width, vals, width=width, label=model, color=MODEL_COLORS.get(model, "#5a84b8"))
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_title("M0 vs selected model summary")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, figures_dir / "fig_M0_vs_selected_summary")


def generate_figures(
    awrs: pd.DataFrame,
    direction_table: pd.DataFrame,
    weight_table: pd.DataFrame,
    channel_scores: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    state_frame: pd.DataFrame,
    occupancy: pd.DataFrame,
    events: pd.DataFrame,
    event_eval: pd.DataFrame,
    boundaries: pd.DataFrame,
    selected_model: str,
    dirs: Dict[str, Path],
) -> None:
    plot_model_comparison(summary, comparison, dirs["figures"])
    plot_awrs_timeseries(awrs, boundaries, dirs["figures"])
    plot_feature_directions(direction_table, dirs["figures"])
    plot_feature_weights(weight_table, dirs["figures"])
    plot_channel_scores(channel_scores, boundaries, dirs["figures"])
    plot_state_map(state_frame, dirs["figures"])
    plot_occupancy(occupancy, dirs["figures"])
    plot_tes_alignment(state_frame, events, boundaries, dirs["figures"])
    plot_m0_vs_selected(summary, state_frame, event_eval, selected_model, dirs["figures"])


def write_interpretation_report(
    report_path: Path,
    config: WeightedAWRConfig,
    decision: Dict[str, object],
    direction_table: pd.DataFrame,
    weight_table: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    occupancy: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    selected = decision["selected_model"]
    stable_count = int((direction_table["direction_stability"] >= config.unstable_direction_threshold).sum())
    total_dir_features = int(len(direction_table))
    enter_m2 = int(direction_table["enter_M2"].sum())
    enter_m3 = int(direction_table["enter_M3"].sum())
    top_weights = (
        weight_table[weight_table["normalized_weight"] > 0]
        .sort_values("normalized_weight", ascending=False)
        .head(10)[["direction_id", "feature_name", "channel", "normalized_weight", "direction_stability"]]
    )
    comp_text = dataframe_to_markdown(comparison)
    summary_text = dataframe_to_markdown(summary)
    top_text = dataframe_to_markdown(top_weights) if not top_weights.empty else "No weighted features entered M2."
    high_occ = occupancy[
        (occupancy["state_region"] == "high_AWR_high_BD")
        & (occupancy["group_value"].astype(int) == 5)
    ]
    occ_text = dataframe_to_markdown(high_occ) if not high_occ.empty else "No Stage5 high_AWR_high_BD rows."
    event_count = int(len(events))
    high_conf = int(events["high_confidence_event"].sum()) if not events.empty else 0

    text = f"""# Weighted AWR-core signal-layer interpretation report

## Purpose

This run rebuilds AWR-score as a transparent continuous state score before physical closed-loop validation. It keeps the existing M0 `stable_plus` mean-z AWR as a baseline and adds M1/M2/M3 to make feature direction, feature weight, and Fx/Fz versus Fy/Fz channel contribution explicit. It does not retrain Stage1-Stage5 classification, and Stage5 is used only as a late-state proxy label for external evaluation.

## Model definitions

- M0: existing `stable_plus` mean-z AWR read from the v2 state table.
- M1: direction-corrected mean-z AWR, using source-only `direction_sign`.
- M2: direction-corrected AWR weighted by effect size, bootstrap direction stability, and redundancy factor.
- M3: channel-constrained weighted AWR, with Hx from rx=Fx/Fz and Hy from ry=Fy/Fz. `M3_equal` is the transparent default candidate and `M3_weighted` is a source-validation fusion candidate.

## Direction and stability

- Direction rows: {total_dir_features}
- Stable direction rows with stability >= {config.unstable_direction_threshold:.2f}: {stable_count}
- Features entering M2: {enter_m2}
- Features entering M3: {enter_m3}
- Bootstrap: {config.bootstrap_n} block resamples, block size {config.bootstrap_block_size} windows.

## Feature weights

{top_text}

## Bidirectional comparison

{comp_text}

Detailed target-side metrics:

{summary_text}

## Selected model

Selected model: **{selected}**

Reason: {decision["reason"]}

The selection uses worst-direction behavior, interpretability, and channel traceability. It is not a search for the highest one-direction score.

## State occupancy and TES

Stage5 high_AWR_high_BD occupancy rows:

{occ_text}

TES events detected with selected AWR: {event_count}, high-confidence events: {high_conf}.

## Dependency boundary

BD v2 is reused as an AWR-independent baseline deviation layer. RS depends on selected AWR trend, and TES partly depends on selected AWR volatility, so RS/TES should be interpreted as derived signal-layer descriptors rather than independent physical evidence. FEM/contact morphology/debris observations remain the next external validation layer.

## Current conclusion boundary

The outputs describe continuous signal-state structure and channel-level AWR contributions. They are not wear-depth prediction, not a failure-warning result, and not a replacement for physical closed-loop validation.
"""
    report_path.write_text(text, encoding="utf-8")


def config_payload(config: WeightedAWRConfig, selected_model: str) -> Dict[str, object]:
    payload = asdict(config)
    payload["feature_list"] = all_feature_names()
    payload["stable_plus_features_M0"] = STABLE_PLUS_FEATURES
    payload["model_definitions"] = {
        "M0": "existing stable_plus mean-z baseline",
        "M1": "direction-corrected mean-z",
        "M2": "direction-corrected effect/stability/redundancy weighted",
        "M3_equal": "equal Hx/Hy channel-constrained fusion",
        "M3_weighted": "source-validation nonnegative Hx/Hy fusion",
    }
    payload["direction_source"] = "source_dataset_train_only"
    payload["early_definition"] = "source train Stage1 windows"
    payload["late_definition"] = "source train Stage5 windows, fallback Stage4+5 if needed"
    payload["effect_size_metric"] = "abs_cliffs_delta_primary_with_cohen_d_reported"
    payload["weighting_formula"] = "abs_effect_size * direction_stability * redundancy_factor"
    payload["channel_fusion_rule"] = "M3_equal default; M3_weighted alpha selected on source validation only"
    payload["selected_model"] = selected_model
    payload["threshold_rules"] = "source validation P95 for high AWR state regions"
    payload["BD_metric"] = config.bd_default_metric
    payload["RS_parameters"] = {"horizons": list(config.rs_horizons), "threshold_source": "source baseline"}
    payload["TES_parameters"] = {
        "window": config.tes_window,
        "smoothing": config.tes_smoothing_window,
        "weights": [config.tes_weight_awr_vol, config.tes_weight_bd_jump, config.tes_weight_shape_jump],
    }
    return payload


def main() -> None:
    config = WeightedAWRConfig()
    dirs = setup_dirs(config)
    setup_logging(dirs)
    logging.info("Starting weighted AWR-core modeling")

    v2_dir = Path(config.v2_dir)
    state_path = v2_dir / "window_state_scores_v2.csv"
    bd_threshold_path = v2_dir / "bd_thresholds_v2.csv"
    boundary_path = v2_dir / "stage_boundaries_v2.csv"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing v2 state table: {state_path}")
    state_v2 = pd.read_csv(state_path)
    bd_thresholds = pd.read_csv(bd_threshold_path) if bd_threshold_path.exists() else pd.DataFrame()
    boundaries = pd.read_csv(boundary_path) if boundary_path.exists() else pd.DataFrame()
    logging.info("Read v2 results: yes (%s)", state_path)

    feature_wide, feature_long, meta, raw_meta = load_raw_or_window_data(config)
    feature_long.to_csv(dirs["results"] / "window_feature_table.csv", index=False, encoding="utf-8-sig")
    feature_wide.to_csv(dirs["results"] / "window_feature_wide.csv", index=False, encoding="utf-8-sig")
    z_wide, z_long, norm_diag = robust_baseline_normalize(feature_wide, meta, config)
    z_long.to_csv(dirs["results"] / "window_feature_z_table.csv", index=False, encoding="utf-8-sig")
    norm_diag.to_csv(dirs["diagnostics"] / "feature_normalization_diagnostics.csv", index=False, encoding="utf-8-sig")

    direction_table, weight_table, channel_scores, awrs, split_masks = compute_direction_models(
        z_wide, state_v2, meta, dirs, config
    )
    direction_table.to_csv(dirs["results"] / "feature_direction_table.csv", index=False, encoding="utf-8-sig")
    weight_table.to_csv(dirs["results"] / "feature_weight_table.csv", index=False, encoding="utf-8-sig")
    channel_scores.to_csv(dirs["results"] / "channel_score_table.csv", index=False, encoding="utf-8-sig")

    awr_thresholds = compute_awr_thresholds(awrs, split_masks, config)
    awr_thresholds.to_csv(dirs["results"] / "awr_thresholds_weighted.csv", index=False, encoding="utf-8-sig")
    bidirectional_summary, comparison = evaluate_bidirectional(awrs, awr_thresholds, state_v2, bd_thresholds, config)
    bidirectional_summary.to_csv(dirs["results"] / "bidirectional_model_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(dirs["results"] / "model_comparison_summary.csv", index=False, encoding="utf-8-sig")
    decision = select_model(comparison, weight_table)
    selected_model = str(decision["selected_model"])
    (dirs["results"] / "selected_model_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    awrs = add_selected_scores(awrs, selected_model, config)
    awrs.to_csv(dirs["results"] / "window_awrs_M0_M1_M2_M3.csv", index=False, encoding="utf-8-sig")

    (
        state_frame,
        stage_summary,
        events,
        event_eval,
        occupancy,
        rs_thresholds,
        tes_thresholds,
    ) = compute_state_metrics_for_selected_AWR(
        awrs, state_v2, awr_thresholds, bd_thresholds, boundaries, selected_model, config
    )
    state_frame.to_csv(dirs["results"] / "window_state_scores_weighted.csv", index=False, encoding="utf-8-sig")
    stage_summary.to_csv(dirs["results"] / "stage_state_summary_weighted.csv", index=False, encoding="utf-8-sig")
    events.to_csv(dirs["results"] / "tes_events_weighted.csv", index=False, encoding="utf-8-sig")
    event_eval.to_csv(dirs["results"] / "tes_event_eval_summary_weighted.csv", index=False, encoding="utf-8-sig")
    occupancy.to_csv(dirs["results"] / "awr_bd_state_occupancy_weighted.csv", index=False, encoding="utf-8-sig")
    rs_thresholds.to_csv(dirs["results"] / "rs_thresholds_weighted.csv", index=False, encoding="utf-8-sig")
    tes_thresholds.to_csv(dirs["results"] / "tes_thresholds_weighted.csv", index=False, encoding="utf-8-sig")

    write_interpretation_report(
        dirs["reports"] / "weighted_awrcore_interpretation.md",
        config,
        decision,
        direction_table,
        weight_table,
        bidirectional_summary,
        comparison,
        occupancy,
        events,
    )
    config_out = config_payload(config, selected_model)
    config_out["raw_meta"] = raw_meta
    config_out["selected_model_decision"] = decision
    (dirs["configs"] / "weighted_awrcore_config.json").write_text(
        json.dumps(config_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    generate_figures(
        awrs,
        direction_table,
        weight_table,
        channel_scores,
        bidirectional_summary,
        comparison,
        state_frame,
        occupancy,
        events,
        event_eval,
        boundaries,
        selected_model,
        dirs,
    )

    logging.info("Weighted AWR modeling complete.")
    logging.info("Selected model: %s", selected_model)
    logging.info("Output directory: %s", Path(config.output_dir))
    print("Weighted AWR modeling complete.")
    print(f"Selected model: {selected_model}")
    print(f"Output directory: {config.output_dir}/")


if __name__ == "__main__":
    main()
