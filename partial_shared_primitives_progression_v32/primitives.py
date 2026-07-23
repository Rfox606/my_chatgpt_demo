from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import V32Config


def descriptor_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in frame.columns if column.startswith("descriptor_"))


def _scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - location), axis=0)
    return location, np.maximum(mad, 1e-9)


def _bic(values: np.ndarray, labels: np.ndarray, centres: np.ndarray) -> float:
    sse = float(((values - centres[labels]) ** 2).sum())
    n, d, k = len(values), values.shape[1], len(centres)
    return float(n * d * np.log(max(sse / max(n * d, 1), 1e-12)) + (k * d + k) * np.log(max(n, 2)))


@dataclass(frozen=True)
class SegmentClusterModel:
    centres: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    selected_k: int
    columns: tuple[str, ...]
    provenance: str

    def transform(self, descriptors: pd.DataFrame) -> np.ndarray:
        return (descriptors.loc[:, list(self.columns)].to_numpy(float) - self.location) / self.scale

    def distances(self, descriptors: pd.DataFrame) -> np.ndarray:
        values = self.transform(descriptors)
        return np.sqrt(((values[:, None, :] - self.centres[None, :, :]) ** 2).mean(axis=2))


def fit_segment_clusters(
    descriptors: pd.DataFrame, candidates: tuple[int, ...], config: V32Config, provenance: str
) -> SegmentClusterModel | None:
    if descriptors.empty or len(descriptors) < 2:
        return None
    columns = descriptor_columns(descriptors)
    if not columns:
        return None
    raw = descriptors.loc[:, list(columns)].to_numpy(float)
    location, scale = _scale(raw)
    values = (raw - location) / scale
    valid: list[tuple[float, int, KMeans]] = []
    for k in candidates:
        if k > len(values):
            continue
        fitted = KMeans(n_clusters=k, n_init=20, random_state=config.random_seed + k).fit(values)
        counts = np.bincount(fitted.labels_, minlength=k)
        if np.all(counts >= max(1, int(len(values) * config.minimum_cluster_fraction))):
            valid.append((_bic(values, fitted.labels_, fitted.cluster_centers_), k, fitted))
    if not valid:
        fitted = KMeans(n_clusters=min(2, len(values)), n_init=20, random_state=config.random_seed).fit(values)
        selected = len(fitted.cluster_centers_)
    else:
        _, selected, fitted = min(valid, key=lambda item: item[0])
    return SegmentClusterModel(
        fitted.cluster_centers_, location, scale, int(selected), columns, provenance
    )


def primitive_table(
    model: SegmentClusterModel | None, source_segments: pd.DataFrame, dataset: str
) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame(
            columns=["dataset", "primitive_id", "selected_k", "source_segment_rows", "descriptor_provenance"]
        )
    labels = model.distances(source_segments).argmin(axis=1)
    rows: list[dict[str, object]] = []
    for primitive_id, centre in enumerate(model.centres):
        support = int(np.sum(labels == primitive_id))
        rows.append(
            {
                "dataset": dataset,
                "row_type": "segment_dynamic_primitive",
                "primitive_id": int(primitive_id),
                "selected_k": model.selected_k,
                "support_segments": support,
                "source_segment_rows": int(len(source_segments)),
                "descriptor_provenance": "BOCPD_confirmed_segments",
                "window_rows_not_used_as_primitives": True,
                "centre": np.array2string(centre, precision=7, separator=","),
            }
        )
    return pd.DataFrame(rows)


def online_target_states(
    target_segments: pd.DataFrame, source_model: SegmentClusterModel | None, config: V32Config
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Assign shared-like matches or create target-only private states online.

    Source centres are consulted solely to score *similarity*.  They are never
    copied as target state centres or used to fix a target state count.
    """
    if target_segments.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "FAIL", "reason": "no_confirmed_target_segments", "private_state_count": 0
        }
    columns = descriptor_columns(target_segments)
    if not columns:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "FAIL", "reason": "no_target_segment_descriptors", "private_state_count": 0
        }
    raw = target_segments.loc[:, list(columns)].to_numpy(float)
    target_location, target_scale = _scale(raw[:1])
    # A single segment has zero robust spread.  The fallback scale is a fixed
    # descriptor unit; it is target-local and not a source centre transfer.
    target_scale = np.maximum(target_scale, 1.0)
    private_centres: list[np.ndarray] = []
    private_support: list[int] = []
    path_rows: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []
    for arrival, (_, segment) in enumerate(target_segments.sort_values("end_index").iterrows()):
        descriptor = segment.loc[list(columns)].to_numpy(float)
        if source_model is not None:
            source_normalized = (descriptor - source_model.location) / source_model.scale
            source_distances = np.sqrt(((source_model.centres - source_normalized[None, :]) ** 2).mean(axis=1))
            nearest_source = int(source_distances.argmin())
            source_distance = float(source_distances[nearest_source])
            match_quality = float(np.exp(-source_distance))
        else:
            nearest_source = -1
            source_distance = float("inf")
            match_quality = 0.0
        local = (descriptor - target_location) / target_scale
        shared_like = match_quality >= config.shared_match_quality_threshold
        created = False
        if shared_like:
            state_name = f"SHARED_LIKE_{nearest_source}"
            state_type = "SHARED_LIKE"
            private_id: int | None = None
            support = 0
        else:
            if private_centres:
                distances = np.sqrt(((np.asarray(private_centres) - local[None, :]) ** 2).mean(axis=1))
                nearest_private = int(distances.argmin())
                can_reuse = float(distances[nearest_private]) <= config.private_descriptor_distance_threshold
            else:
                nearest_private = -1
                can_reuse = False
            if can_reuse:
                private_id = nearest_private
                private_support[private_id] += 1
                private_centres[private_id] += (local - private_centres[private_id]) / private_support[private_id]
                event = "updated"
            else:
                private_id = len(private_centres)
                private_centres.append(local.copy())
                private_support.append(1)
                created = True
                event = "created"
            support = private_support[private_id]
            state_name = f"TARGET_PRIVATE_{private_id}"
            state_type = "TARGET_PRIVATE"
            log_rows.append(
                {
                    "arrival_segment_order": arrival,
                    "segment_id": int(segment.segment_id),
                    "end_cycle": float(segment.end_cycle),
                    "private_state_id": private_id,
                    "private_state_name": state_name,
                    "event": event,
                    "private_state_support": support,
                    "private_state_count_after_event": len(private_centres),
                    "online_new_state_allowed": True,
                    "source_state_centre_used": False,
                }
            )
        path_rows.append(
            {
                "segment_id": int(segment.segment_id),
                "start_index": int(segment.start_index),
                "end_index": int(segment.end_index),
                "start_cycle": float(segment.start_cycle),
                "end_cycle": float(segment.end_cycle),
                "current_state_id": state_name,
                "current_state_type": state_type,
                "source_primitive_id": nearest_source,
                "source_primitive_match_quality": match_quality,
                "source_primitive_distance": source_distance,
                "private_state_id": private_id,
                "private_state_support": support,
                "new_private_state_created": created,
                "state_uncertainty": float(1.0 - match_quality if shared_like else 1.0 / max(support, 1)),
                "descriptor_row_is_confirmed_segment": True,
                "target_state_count_so_far": len(private_centres),
                "source_state_centre_used": False,
                "cross_experiment_state_alignment": False,
            }
        )
    decision = {
        "status": "PASS" if len(path_rows) >= 2 else "FAIL",
        "reason": "online_target_states_from_confirmed_segments" if len(path_rows) >= 2 else "fewer_than_two_confirmed_target_segments",
        "confirmed_segments": len(path_rows),
        "private_state_count": len(private_centres),
        "private_state_can_grow_online": True,
        "source_k_imposed_on_target": False,
        "source_state_centre_used": False,
    }
    return pd.DataFrame(path_rows), pd.DataFrame(log_rows), decision
