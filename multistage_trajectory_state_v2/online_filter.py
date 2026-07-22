from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MultiStageTrajectoryConfig
from .regime_model import RegimeStructure


def _entropy(probability: np.ndarray) -> float:
    return float(-np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0))) / np.log(max(len(probability), 2)))


@dataclass
class OnlineRegimeFilter:
    structure: RegimeStructure
    config: MultiStageTrajectoryConfig
    adaptive: bool = True

    def __post_init__(self) -> None:
        k, d = self.structure.centres.shape
        self.offset = np.zeros((k, d)); self.log_scale = np.zeros((k, d)); self.support = np.zeros(k); self.prior_weights = np.full(k, 1 / k)
        self.previous = self.prior_weights.copy(); self.current: int | None = None; self.pending: int | None = None; self.pending_count = 0; self.duration = 0
        self.duration_estimate = self.structure.duration_mean.copy(); self.aborted = False; self.events: list[dict[str, object]] = []

    def _emission(self, value: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        scale = np.exp(self.log_scale); variance = self.structure.variances * scale
        distance = np.sqrt(np.mean((value[None, :] - (self.structure.centres + self.offset)) ** 2 / np.maximum(variance, 1e-12), axis=1))
        novelty = float(distance.min() / self.structure.novelty_threshold)
        likelihood = np.exp(-.5 * distance ** 2); likelihood /= max(float(likelihood.sum()), 1e-12)
        return likelihood, distance, novelty

    def _update(self, state: int, value: np.ndarray, posterior: np.ndarray, novelty: float, volatility: float, index: int) -> tuple[int, str]:
        if not self.adaptive: return 0, "SOURCE_FROZEN"
        if novelty > self.config.regime_update_max_novelty: return 0, "HIGH_NOVELTY"
        if posterior[state] < self.config.regime_update_min_posterior: return 0, "LOW_POSTERIOR"
        if volatility > self.config.regime_update_max_volatility: return 0, "HIGH_VOLATILITY"
        eta = self.config.regime_adapter_learning_rate / np.sqrt(self.support[state] + 1.0)
        old_offset = self.offset[state].copy(); residual = value - self.structure.centres[state]
        self.offset[state] = (1 - eta * self.config.regime_adapter_l2) * self.offset[state] + eta * residual
        normalised_square = (value - self.structure.centres[state] - self.offset[state]) ** 2 / np.maximum(self.structure.variances[state], 1e-12)
        self.log_scale[state] = (1 - eta * self.config.regime_adapter_l2) * self.log_scale[state] + eta * np.log(normalised_square + 1e-6)
        self.prior_weights = (1 - eta) * self.prior_weights + eta * posterior; self.prior_weights /= self.prior_weights.sum()
        self.duration_estimate[state] = (1 - eta) * self.duration_estimate[state] + eta * max(self.duration, 1)
        if not np.isfinite(self.offset).all() or not np.isfinite(self.log_scale).all() or np.linalg.norm(self.offset) > 100:
            self.offset[state] = old_offset; self.aborted = True; self.events.append({"window_index": index, "event": "NUMERIC_ABORT", "regime_id": state}); return 0, "NUMERIC_ABORT"
        self.support[state] += 1; return 1, "UPDATED_SOFT_SHRINKAGE"

    def step(self, value: np.ndarray, metadata: dict[str, object], *, frozen_future: bool = False) -> dict[str, object]:
        index = int(metadata["window_index"]); cycle = float(metadata["center_cycle"]); likelihood, distance, novelty = self._emission(value)
        prior = .5 * (self.previous @ self.structure.transition) + .5 * self.prior_weights
        posterior = prior * likelihood; posterior /= max(float(posterior.sum()), 1e-12)
        proposed = int(np.argmax(posterior)); unknown = bool(novelty > 1.0)
        transition_event = "STAY"
        if unknown:
            state = -1; visible_duration = 0; transition_event = "UNKNOWN_NOVEL"
        else:
            if self.current is None:
                self.current = proposed; self.duration = 1; transition_event = "INITIALISE"
            elif proposed == self.current:
                self.duration += 1; self.pending = None; self.pending_count = 0
            else:
                self.pending_count = self.pending_count + 1 if self.pending == proposed else 1; self.pending = proposed
                self.duration += 1
                if self.pending_count >= self.config.regime_min_dwell_windows:
                    self.events.append({"window_index": index, "center_cycle": cycle, "event": "REGIME_TRANSITION", "from_regime": int(self.current), "to_regime": proposed})
                    self.current = proposed; self.duration = 1; self.pending = None; self.pending_count = 0; transition_event = "REGIME_TRANSITION"
            state = int(self.current); visible_duration = self.duration
        volatility = float(metadata.get("volatility_mean_500", 0.0))
        if state < 0:
            updated, update_reason = 0, "UNKNOWN_NOVEL"
        elif frozen_future:
            updated, update_reason = 0, "FROZEN_FUTURE"
        else:
            updated, update_reason = self._update(state, value, posterior, novelty, volatility, index)
        self.previous = posterior
        activity = float(1 / (1 + np.exp(-(abs(float(metadata.get("slope_short_long_gap", 0.0))) + volatility))))
        match = 0.0 if state < 0 else float(posterior[state] * np.exp(-min(novelty, 20.0)))
        within = 0.0 if state < 0 else float(1 - np.exp(-visible_duration / max(self.duration_estimate[state], 1.0)))
        uncertainty = 1.0 if state < 0 else float(.5 * _entropy(posterior) + .5 * min(novelty, 1.0))
        row: dict[str, object] = {
            "dataset": metadata["dataset"], "window_id": metadata["window_id"], "window_index": index, "center_cycle": cycle,
            "regime_probability": json.dumps(posterior.tolist()), "most_likely_regime": "UNKNOWN_NOVEL" if state < 0 else f"REGIME_{state}", "regime_id": state,
            "regime_duration": visible_duration, "within_regime_progress": within, "activity_score": activity, "trajectory_match_score": match,
            "novelty_score": novelty, "state_uncertainty": uncertainty, "emission_nll": float(-np.log(max(posterior[proposed], 1e-12))),
            "feature_reconstruction_error": float(distance[state] if state >= 0 else distance.min()), "transition_event": transition_event,
            "target_update_applied": updated, "target_update_reason": update_reason, "source_structure_frozen_hash": self.structure.source_hash,
        }
        for label, probability in enumerate(posterior): row[f"p_regime_{label}"] = float(probability)
        return row


def run_online_filter(descriptors: pd.DataFrame, structure: RegimeStructure, config: MultiStageTrajectoryConfig, *, adaptive: bool, freeze_after_index: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, OnlineRegimeFilter]:
    runner = OnlineRegimeFilter(structure, config, adaptive=adaptive); rows: list[dict[str, object]] = []
    values = descriptors.loc[:, list(structure.descriptor_columns)].to_numpy(float)
    for position, (_, item) in enumerate(descriptors.iterrows()):
        frozen = freeze_after_index is not None and position >= freeze_after_index
        row = runner.step(values[position], item.to_dict(), frozen_future=frozen)
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(runner.events), runner
