from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PartialSharedPrimitivesConfig
from .data import robust_location_scale


@dataclass(frozen=True)
class SharedRepresentationResult:
    frame: pd.DataFrame
    predictor_hash: str


def _huber_matrix(residual: np.ndarray, delta: float) -> np.ndarray:
    magnitude = np.maximum(np.abs(residual), 1e-12)
    return np.minimum(1.0, delta / magnitude)


def _input_target(values: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    history = values[:index]
    location, scale = robust_location_scale(history)
    previous = (values[index - 1] - location) / scale
    lag = max(0, index - 4)
    recent_delta = (values[index - 1] - values[lag]) / scale
    target_delta = (values[index] - values[index - 1]) / scale
    return np.concatenate((previous, recent_delta)), target_delta


def _representation(coefficient: np.ndarray, x: np.ndarray, dimensions: int) -> np.ndarray:
    if not np.any(np.isfinite(coefficient)) or np.allclose(coefficient, 0.0):
        value = x[:dimensions]
    else:
        left, _, _ = np.linalg.svd(coefficient, full_matrices=False)
        value = x @ left[:, :min(dimensions, left.shape[1])]
    if len(value) < dimensions:
        value = np.pad(value, (0, dimensions - len(value)))
    return np.asarray(value[:dimensions], dtype=float)


def run_shared_causal_representation(frame: pd.DataFrame, config: PartialSharedPrimitivesConfig) -> SharedRepresentationResult:
    """Online pooled ridge predictor. At index j it only sees each experiment's rows < j."""
    grouped = {str(name): group.sort_values(["center_cycle", "window_index"]).reset_index(drop=True) for name, group in frame.groupby("dataset", sort=True)}
    if len(grouped) < 2:
        raise ValueError("v3 requires at least two experiments for the shared representation")
    feature_count = len(config.feature_columns)
    input_count = 2 * feature_count
    normal = config.ridge_alpha * np.eye(input_count)
    cross = np.zeros((input_count, feature_count))
    rows: dict[str, list[dict[str, object]]] = {name: [] for name in grouped}
    # Synchronise independent experiments by *exact* relative observation progress,
    # not raw window number or rounded resampling.  This prevents a short experiment
    # from contributing a post-cutoff target before a long experiment's same-prefix
    # prediction has been emitted.
    events: list[tuple[float, str, int]] = []
    for name, group in grouped.items():
        denominator = max(len(group) - 1, 1)
        events.extend((local_index / denominator, name, local_index) for local_index in range(len(group)))
    events.sort(key=lambda item: (item[0], item[1]))
    update_count = 0
    event_start = 0
    while event_start < len(events):
        phase = events[event_start][0]
        event_end = event_start + 1
        while event_end < len(events) and np.isclose(events[event_end][0], phase, rtol=0.0, atol=1e-15):
            event_end += 1
        coefficient = np.linalg.solve(normal, cross)
        staged: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for _, name, local_index in events[event_start:event_end]:
            group = grouped[name]
            raw = group.iloc[local_index]
            values = group.loc[:, list(config.feature_columns)].to_numpy(float)
            if local_index == 0:
                x = np.zeros(input_count); target = np.zeros(feature_count)
            else:
                x, target = _input_target(values, local_index)
            prediction = x @ coefficient if local_index >= config.causal_context_windows else np.zeros(feature_count)
            residual_vector = target - prediction
            representation = _representation(coefficient, x, config.representation_dimension)
            record: dict[str, object] = {key: raw[key] for key in ("dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle")}
            record.update({
                "forecast_mae": float(np.mean(np.abs(residual_vector))),
                "persistence_mae": float(np.mean(np.abs(target))),
                "forecast_activity": float(np.sqrt(np.mean(target ** 2))),
                "shared_predictor_observed_updates": int(update_count),
                "causal_warmup": int(local_index < config.causal_context_windows),
            })
            record.update({f"shared_z{dimension}": float(representation[dimension]) for dimension in range(config.representation_dimension)})
            rows[name].append(record)
            if local_index >= 1:
                staged.append((x, target, residual_vector))
        # Targets at this time are used only after all same-index predictions have been emitted.
        for x, target, residual in staged:
            weight = _huber_matrix(residual, config.huber_delta).mean()
            normal += weight * np.outer(x, x)
            cross += weight * np.outer(x, target)
            update_count += 1
        event_start = event_end
    result = pd.concat([pd.DataFrame(rows[name]) for name in sorted(rows)], ignore_index=True)
    import hashlib
    digest = hashlib.sha256(); digest.update(np.asarray(normal, dtype=np.float64).tobytes()); digest.update(np.asarray(cross, dtype=np.float64).tobytes())
    return SharedRepresentationResult(result, digest.hexdigest())
