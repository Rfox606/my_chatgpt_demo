from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from .config import PartialSharedPrimitivesConfig
from .data import robust_location_scale


@dataclass(frozen=True)
class PrimitiveDictionary:
    centres: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    selected_k: int
    descriptor_columns: tuple[str, ...]
    calibration_rows: int

    def standardize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.location) / self.scale

    def assignment(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        standardized = self.standardize(values)
        squared = ((standardized[:, None, :] - self.centres[None, :, :]) ** 2).mean(axis=2)
        logits = -squared
        logits -= logits.max(axis=1, keepdims=True)
        posterior = np.exp(logits); posterior /= posterior.sum(axis=1, keepdims=True)
        return posterior.argmax(axis=1), posterior


def primitive_descriptor_columns(config: PartialSharedPrimitivesConfig) -> tuple[str, ...]:
    return tuple([*(f"shared_z{index}" for index in range(config.representation_dimension)), "forecast_mae", "forecast_activity"])


def _bic(values: np.ndarray, labels: np.ndarray, centres: np.ndarray) -> float:
    sse = float(((values - centres[labels]) ** 2).sum())
    n, d, k = len(values), values.shape[1], len(centres)
    return float(n * d * np.log(max(sse / max(n * d, 1), 1e-12)) + (k * d + k) * np.log(max(n, 2)))


def fit_primitive_dictionary(representation: pd.DataFrame, config: PartialSharedPrimitivesConfig) -> tuple[PrimitiveDictionary, pd.DataFrame]:
    columns = primitive_descriptor_columns(config)
    calibration = representation.loc[
        (representation.window_index >= config.causal_context_windows) & (representation.window_index < config.primitive_calibration_windows), list(columns)
    ].to_numpy(float)
    if len(calibration) < max(config.primitive_k_candidates):
        raise ValueError("Primitive calibration period is too short")
    location, scale = robust_location_scale(calibration); values = (calibration - location) / scale
    candidates: list[tuple[float, int, KMeans]] = []
    for k in config.primitive_k_candidates:
        fitted = KMeans(n_clusters=k, n_init=20, random_state=config.random_seed).fit(values)
        counts = np.bincount(fitted.labels_, minlength=k)
        if np.all(counts >= max(1, int(len(values) * config.primitive_min_fraction))):
            candidates.append((_bic(values, fitted.labels_, fitted.cluster_centers_), k, fitted))
    if not candidates:
        fitted = KMeans(n_clusters=2, n_init=20, random_state=config.random_seed).fit(values); selected = 2
    else:
        _, selected, fitted = min(candidates, key=lambda item: item[0])
    dictionary = PrimitiveDictionary(fitted.cluster_centers_, location, scale, int(selected), columns, len(values))
    rows = []
    for primitive in range(dictionary.selected_k):
        rows.append({
            "row_type": "shared_dynamic_primitive", "primitive_id": primitive, "selected_k": dictionary.selected_k,
            "calibration_rows": dictionary.calibration_rows, "descriptor_columns": "|".join(columns),
            "centre": np.array2string(dictionary.centres[primitive], precision=8, separator=","),
            "shared_not_state_centre": True,
        })
    return dictionary, pd.DataFrame(rows)


def attach_primitive_prior(representation: pd.DataFrame, dictionary: PrimitiveDictionary, config: PartialSharedPrimitivesConfig) -> pd.DataFrame:
    result = representation.copy(); values = result.loc[:, list(dictionary.descriptor_columns)].to_numpy(float); labels, posterior = dictionary.assignment(values)
    warmup = result.window_index.to_numpy(int) < config.primitive_calibration_windows
    labels = labels.astype(int); labels[warmup] = -1; posterior[warmup] = 0.0
    result["dynamic_primitive_id"] = labels
    result["dynamic_primitive_distance"] = np.sqrt(np.min(((dictionary.standardize(values)[:, None, :] - dictionary.centres[None, :, :]) ** 2).mean(axis=2), axis=1))
    for primitive in range(dictionary.selected_k):
        result[f"primitive_p{primitive}"] = posterior[:, primitive]
    return result


def bootstrap_primitive_stability(representation: pd.DataFrame, dictionary: PrimitiveDictionary, config: PartialSharedPrimitivesConfig) -> pd.DataFrame:
    calibration = representation.loc[
        (representation.window_index >= config.causal_context_windows) & (representation.window_index < config.primitive_calibration_windows)
    ].copy()
    values = dictionary.standardize(calibration.loc[:, list(dictionary.descriptor_columns)].to_numpy(float))
    reference, _ = dictionary.assignment(calibration.loc[:, list(dictionary.descriptor_columns)].to_numpy(float))
    rng = np.random.default_rng(config.random_seed + 301); rows: list[dict[str, object]] = []; block = max(8, len(values) // 12)
    for replicate in range(config.primitive_bootstrap_replicates):
        pieces: list[np.ndarray] = []
        while sum(len(piece) for piece in pieces) < len(values):
            start = int(rng.integers(0, max(1, len(values) - block + 1))); pieces.append(np.arange(start, min(len(values), start + block)))
        sample = values[np.concatenate(pieces)[:len(values)]]
        fitted = KMeans(n_clusters=dictionary.selected_k, n_init=10, random_state=config.random_seed + replicate + 1).fit(sample)
        labels = ((values[:, None, :] - fitted.cluster_centers_[None, :, :]) ** 2).mean(axis=2).argmin(axis=1)
        for dataset, positions in calibration.groupby("dataset", sort=True).groups.items():
            index = np.asarray(list(positions), dtype=int)
            # group indices are labels from the parent DataFrame; use positional selection explicitly.
            local = calibration.index.get_indexer(index)
            local = local[local >= 0]
            rows.append({"replicate": replicate, "dataset": dataset, "adjusted_rand_index": float(adjusted_rand_score(reference[local], labels[local])), "selected_k": dictionary.selected_k})
    return pd.DataFrame(rows)

