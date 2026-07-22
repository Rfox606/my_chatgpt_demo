from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import V31Config


def descriptor_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in frame.columns if column.startswith(("mean_", "slope_")))


def _scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = np.median(values, axis=0); mad = 1.4826 * np.median(np.abs(values - location), axis=0); return location, np.maximum(mad, 1e-9)


def _bic(values: np.ndarray, labels: np.ndarray, centres: np.ndarray) -> float:
    sse = float(((values - centres[labels]) ** 2).sum()); n, d, k = len(values), values.shape[1], len(centres); return float(n * d * np.log(max(sse / max(n * d, 1), 1e-12)) + (k * d + k) * np.log(max(n, 2)))


@dataclass(frozen=True)
class SegmentClusterModel:
    centres: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    selected_k: int
    columns: tuple[str, ...]
    provenance: str

    def labels(self, descriptors: pd.DataFrame) -> np.ndarray:
        values = (descriptors.loc[:, list(self.columns)].to_numpy(float) - self.location) / self.scale; return ((values[:, None, :] - self.centres[None, :, :]) ** 2).mean(axis=2).argmin(axis=1)


def fit_segment_clusters(descriptors: pd.DataFrame, candidates: tuple[int, ...], config: V31Config, provenance: str) -> SegmentClusterModel | None:
    if descriptors.empty: return None
    columns = descriptor_columns(descriptors); raw = descriptors.loc[:, list(columns)].to_numpy(float)
    if len(raw) < 2: return None
    location, scale = _scale(raw); values = (raw - location) / scale; valid: list[tuple[float, int, KMeans]] = []
    for k in candidates:
        if k > len(values): continue
        fitted = KMeans(n_clusters=k, n_init=20, random_state=config.random_seed + k).fit(values); counts = np.bincount(fitted.labels_, minlength=k)
        if np.all(counts >= max(1, int(len(values) * config.minimum_cluster_fraction))): valid.append((_bic(values, fitted.labels_, fitted.cluster_centers_), k, fitted))
    if not valid:
        fitted = KMeans(n_clusters=min(2, len(values)), n_init=20, random_state=config.random_seed).fit(values); selected = len(fitted.cluster_centers_)
    else:
        _, selected, fitted = min(valid, key=lambda item: item[0])
    return SegmentClusterModel(fitted.cluster_centers_, location, scale, int(selected), columns, provenance)


def primitive_table(model: SegmentClusterModel | None, source_segments: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if model is None: return pd.DataFrame(columns=["dataset", "row_type", "selected_k"])
    return pd.DataFrame([{ "dataset": dataset, "row_type": "segment_dynamic_primitive", "primitive_id": index, "selected_k": model.selected_k, "descriptor_provenance": "BOCPD_confirmed_segments", "source_segment_rows": len(source_segments), "window_rows_not_used_as_primitives": True, "centre": np.array2string(item, precision=7, separator=",") } for index, item in enumerate(model.centres)])


def private_state_path(target_segments: pd.DataFrame, config: V31Config) -> tuple[pd.DataFrame, SegmentClusterModel | None, dict[str, object]]:
    calibration_count = config.private_state_calibration_confirmed_segments
    if len(target_segments) < calibration_count:
        return pd.DataFrame(), None, {"status": "FAIL", "reason": "fewer_than_6_confirmed_target_segments", "confirmed_segments": len(target_segments)}
    calibration = target_segments.iloc[:calibration_count].copy(); model = fit_segment_clusters(calibration, config.private_state_k_candidates, config, "target_confirmed_segments_only")
    if model is None: return pd.DataFrame(), None, {"status": "FAIL", "reason": "target_private_state_fit_failed", "confirmed_segments": len(target_segments)}
    labels = model.labels(target_segments); rows: list[dict[str, object]] = []
    for position, (_, segment) in enumerate(target_segments.iterrows()):
        label = -1 if position < calibration_count else int(labels[position]); rows.append({"segment_id": int(segment.segment_id), "start_cycle": segment.start_cycle, "end_cycle": segment.end_cycle, "private_state_id": label, "private_state_name": "TARGET_CALIBRATION" if label < 0 else f"TARGET_PRIVATE_{label}", "selected_private_k": model.selected_k, "state_centre_provenance": model.provenance, "source_state_centre_used": False, "cross_experiment_state_alignment": False, "descriptor_row_is_confirmed_segment": True})
    return pd.DataFrame(rows), model, {"status": "PASS", "reason": "target_only_bic_private_states", "confirmed_segments": len(target_segments), "selected_private_k": model.selected_k}

