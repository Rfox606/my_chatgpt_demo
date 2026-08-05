from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import PartialSharedPrimitivesConfig
from .data import robust_location_scale


@dataclass
class LocalStateModel:
    dataset: str
    centres: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    selected_k: int
    descriptor_columns: tuple[str, ...]
    support: np.ndarray
    previous_posterior: np.ndarray
    current_state: int = -1
    current_dwell: int = 0

    @property
    def provenance(self) -> str:
        return "local_experiment_only"


def state_descriptor_columns(prior: pd.DataFrame, config: PartialSharedPrimitivesConfig) -> tuple[str, ...]:
    primitive = tuple(column for column in prior.columns if column.startswith("primitive_p"))
    return tuple([*(f"shared_z{index}" for index in range(config.representation_dimension)), "forecast_mae", "forecast_activity", *primitive])


def _bic(values: np.ndarray, labels: np.ndarray, centres: np.ndarray) -> float:
    sse = float(((values - centres[labels]) ** 2).sum()); n, d, k = len(values), values.shape[1], len(centres)
    return float(n * d * np.log(max(sse / max(n * d, 1), 1e-12)) + (k * d + k) * np.log(max(n, 2)))


def fit_local_state_model(prior: pd.DataFrame, dataset: str, config: PartialSharedPrimitivesConfig) -> LocalStateModel:
    columns = state_descriptor_columns(prior, config)
    group = prior.loc[prior.dataset.eq(dataset)].sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    calibration = group.loc[(group.window_index >= config.primitive_calibration_windows) & (group.window_index < config.state_calibration_windows), list(columns)].to_numpy(float)
    if len(calibration) < max(config.state_k_candidates):
        raise ValueError(f"{dataset} has too little local state calibration data")
    location, scale = robust_location_scale(calibration); values = (calibration - location) / scale
    candidates: list[tuple[float, int, KMeans]] = []
    for k in config.state_k_candidates:
        fitted = KMeans(n_clusters=k, n_init=20, random_state=config.random_seed + sum(map(ord, dataset))).fit(values)
        counts = np.bincount(fitted.labels_, minlength=k)
        if np.all(counts >= max(1, int(len(values) * config.state_min_fraction))):
            candidates.append((_bic(values, fitted.labels_, fitted.cluster_centers_), k, fitted))
    if not candidates:
        fitted = KMeans(n_clusters=2, n_init=20, random_state=config.random_seed + sum(map(ord, dataset))).fit(values); selected = 2
    else:
        _, selected, fitted = min(candidates, key=lambda item: item[0])
    return LocalStateModel(dataset, fitted.cluster_centers_.copy(), location, scale, int(selected), columns, np.ones(int(selected)), np.full(int(selected), 1.0 / int(selected)))


def score_local_state_path(prior: pd.DataFrame, model: LocalStateModel, config: PartialSharedPrimitivesConfig) -> pd.DataFrame:
    group = prior.loc[prior.dataset.eq(model.dataset)].sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for _, item in group.iterrows():
        values = item.loc[list(model.descriptor_columns)].to_numpy(float); standardized = (values - model.location) / model.scale
        warmup = int(item.window_index) < config.state_calibration_windows
        if warmup:
            posterior = np.full(model.selected_k, 1.0 / model.selected_k); state = -1; event = "LOCAL_CALIBRATION"
        else:
            squared = ((model.centres - standardized[None, :]) ** 2).mean(axis=1)
            logits = -squared
            if not np.isfinite(logits).any():
                emission = np.full(model.selected_k, 1.0 / model.selected_k)
            else:
                peak = np.max(logits[np.isfinite(logits)])
                emission = np.exp(np.clip(logits - peak, -700.0, 0.0)); emission[~np.isfinite(emission)] = 0.0
                emission /= max(float(emission.sum()), 1e-12)
            posterior = config.state_stickiness * model.previous_posterior + (1.0 - config.state_stickiness) * emission; posterior /= posterior.sum()
            candidate = int(posterior.argmax())
            if model.current_state >= 0 and candidate != model.current_state and model.current_dwell < config.state_min_dwell_windows and posterior[model.current_state] >= .20:
                state = model.current_state; event = "STICKY_HOLD"
            else:
                state = candidate; event = "STATE_RETURN" if model.current_state >= 0 and state != model.current_state else "STATE_STAY"
            model.current_dwell = model.current_dwell + 1 if state == model.current_state else 1
            model.current_state = state; model.previous_posterior = posterior
            model.support[state] += 1.0; learning = 1.0 / min(model.support[state], 100.0); model.centres[state] += learning * (standardized - model.centres[state])
        rows.append({
            "dataset": model.dataset, "window_id": int(item.window_id), "window_index": int(item.window_index), "center_cycle": float(item.center_cycle),
            "local_state_id": int(state), "local_state_name": f"{model.dataset}_{'CALIBRATION' if state < 0 else f'LOCAL_{state}'}",
            "state_posterior": np.array2string(posterior, precision=8, separator=","), "state_uncertainty": float(-np.sum(posterior * np.log(np.maximum(posterior, 1e-12)))),
            "state_event": event, "state_centre_provenance": model.provenance, "selected_local_k": int(model.selected_k), "state_path_monotonic_constraint": False,
        })
    return pd.DataFrame(rows)


def synthetic_state_revisit(config: PartialSharedPrimitivesConfig) -> dict[str, object]:
    model = LocalStateModel("Synthetic", np.asarray([[0.0, 0.0], [4.0, 4.0]]), np.zeros(2), np.ones(2), 2, ("a", "b"), np.ones(2), np.asarray([.5, .5]))
    points = np.asarray([[0., 0.]] * 10 + [[4., 4.]] * 10 + [[0., 0.]] * 10)
    frame = pd.DataFrame({"dataset": "Synthetic", "window_id": np.arange(30), "window_index": np.arange(config.state_calibration_windows, config.state_calibration_windows + 30), "center_cycle": np.arange(30), "a": points[:, 0], "b": points[:, 1]})
    scored = score_local_state_path(frame, model, config); labels = scored.local_state_id.to_numpy(int)
    return {"returns_are_allowed": bool(labels[-1] == labels[0] and len(np.unique(labels)) >= 2), "no_monotonic_state_constraint": bool(not scored.state_path_monotonic_constraint.any())}
