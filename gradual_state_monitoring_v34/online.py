from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from gradual_state_monitoring_v33.data import MAIN_FEATURES

from .config import GradualStateMonitoringV34Config
from .data import validate_online_frame
from .model import GradualStateTCN
from .training import RobustLocationScale, robust_location_scale


@dataclass
class OnlineRun:
    scores: pd.DataFrame
    adaptation: pd.DataFrame


class CausalTargetNormalizer:
    """A robust target normalizer updated only after a safe, already-scored window."""

    def __init__(self, initial: np.ndarray, config: GradualStateMonitoringV34Config) -> None:
        self.config = config
        base = robust_location_scale(initial, config.calibration_windows, config.eps)
        self.median = base.median
        self.scale = base.scale
        self.history: list[np.ndarray] = [np.asarray(row, dtype=float).copy() for row in initial]
        self.update_count = 0

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.median) / self.scale

    def update_after_output(self, value: np.ndarray) -> None:
        self.history.append(np.asarray(value, dtype=float).copy())
        if len(self.history) > self.config.target_normalizer_max_history:
            self.history = self.history[-self.config.target_normalizer_max_history:]
        values = np.asarray(self.history, dtype=float)
        self.median = np.median(values, axis=0)
        mad = np.median(np.abs(values - self.median), axis=0) * 1.4826
        self.scale = np.maximum(mad, self.config.eps)
        self.update_count += 1


class RobustScoreHistory:
    def __init__(self, seed: Iterable[float], config: GradualStateMonitoringV34Config) -> None:
        values = [float(value) for value in seed if np.isfinite(value)]
        self.values = values if values else [0.0]
        self.config = config

    def standardise(self, value: float) -> float:
        baseline = np.asarray(self.values, dtype=float)
        median = float(np.median(baseline))
        scale = max(float(np.median(np.abs(baseline - median)) * 1.4826), self.config.score_minimum_scale)
        return max(0.0, (float(value) - median) / scale)

    def quantile(self, q: float) -> float:
        return float(np.quantile(np.asarray(self.values, dtype=float), q))

    def append(self, value: float) -> None:
        if np.isfinite(value):
            self.values.append(float(value))
            if len(self.values) > self.config.target_score_max_history:
                self.values = self.values[-self.config.target_score_max_history:]


def _history(values: np.ndarray, index: int, config: GradualStateMonitoringV34Config,
             transform: callable) -> np.ndarray:
    """Return a causal 128-window history, left-padded only with an already seen row."""
    start = max(0, index - config.history_windows + 1)
    window = np.asarray(values[start:index + 1], dtype=float)
    if len(window) < config.history_windows:
        pad = np.repeat(window[:1], config.history_windows - len(window), axis=0)
        window = np.vstack([pad, window])
    return transform(window).astype(np.float32)


def _encode(model: GradualStateTCN, raw: np.ndarray, index: int, config: GradualStateMonitoringV34Config,
            transform: callable) -> tuple[np.ndarray, float]:
    history = torch.from_numpy(_history(raw, index, config, transform)).unsqueeze(0)
    with torch.no_grad():
        z, logit, _ = model(history)
    return z.squeeze(0).cpu().numpy().astype(float), float(torch.sigmoid(logit).item())


def _encode_static_sequence(model: GradualStateTCN, raw: np.ndarray, config: GradualStateMonitoringV34Config,
                            transform: callable) -> list[tuple[np.ndarray, float]]:
    """Batch a fixed-model run without changing its causal per-row histories."""
    output: list[tuple[np.ndarray, float]] = []
    for start in range(0, len(raw), 256):
        histories = np.stack([_history(raw, index, config, transform) for index in range(start, min(len(raw), start + 256))])
        with torch.no_grad():
            z, logits, _ = model(torch.from_numpy(histories))
        output.extend((embedding.astype(float), float(probability)) for embedding, probability in zip(
            z.cpu().numpy(), torch.sigmoid(logits).cpu().numpy()))
    return output


def _initial_score_histories(model: GradualStateTCN, raw: np.ndarray, normalizer: CausalTargetNormalizer,
                             config: GradualStateMonitoringV34Config,
                             encoded: list[tuple[np.ndarray, float]] | None = None) -> tuple[list[np.ndarray], list[RobustScoreHistory], RobustScoreHistory]:
    embeddings: list[np.ndarray] = []
    probabilities: list[float] = []
    for index in range(config.calibration_windows):
        z, p = encoded[index] if encoded is not None else _encode(model, raw, index, config, normalizer.transform)
        embeddings.append(z); probabilities.append(p)
    distance_seed: list[list[float]] = [[], [], []]
    raw_seed: list[float] = []
    final_seed: list[float] = []
    for index, z in enumerate(embeddings):
        values = [float(np.linalg.norm(z - embeddings[max(0, index - lag)])) for lag in config.score_lags]
        for bucket, value in zip(distance_seed, values):
            bucket.append(value)
    histories = [RobustScoreHistory(values, config) for values in distance_seed]
    for index in range(len(embeddings)):
        distances = [float(np.linalg.norm(embeddings[index] - embeddings[max(0, index - lag)])) for lag in config.score_lags]
        score = sum(weight * history.standardise(distance) for weight, distance, history in zip(config.score_weights, distances, histories))
        raw_seed.append(score)
        final_seed.append((1.0 - config.source_probability_weight) * score + config.source_probability_weight * probabilities[index])
    return embeddings, histories, RobustScoreHistory(raw_seed, config), RobustScoreHistory(final_seed, config)


def _adapter_update(model: GradualStateTCN, teacher: torch.nn.Module, raw: np.ndarray, index: int,
                    normalizer: CausalTargetNormalizer, optimizer: torch.optim.Optimizer,
                    config: GradualStateMonitoringV34Config) -> tuple[float, float, float]:
    """One causal self-supervised update.  Every referenced index is <= index."""
    current_history = torch.from_numpy(_history(raw, index, config, normalizer.transform)).unsqueeze(0)
    previous_history = torch.from_numpy(_history(raw, max(0, index - 1), config, normalizer.transform)).unsqueeze(0)
    origin = max(0, index - config.source_prediction_horizon)
    origin_history = torch.from_numpy(_history(raw, origin, config, normalizer.transform)).unsqueeze(0)
    model.eval(); model.adapter.train()
    optimizer.zero_grad(set_to_none=True)
    # One batched causal encoder pass for current, adjacent, and h=20 origin.
    # Batching does not mix time: Conv1d has no batch-axis interaction.
    encoded = model.encode(torch.cat([current_history, previous_history, origin_history], dim=0))
    z, previous, origin_z = encoded[:1], encoded[1:2], encoded[2:3]
    predicted = model.prediction_head(origin_z)
    with torch.no_grad():
        # The EMA consistency term is deliberately adapter-level.  It still
        # teaches the student from an EMA target but avoids a second non-causal
        # or future-looking representation computation.
        teacher_adjusted = teacher(current_history)
    student_adjusted = model.adapter(current_history)
    target_delta = torch.from_numpy((normalizer.transform(raw[index]) - normalizer.transform(raw[origin])).astype(np.float32)).unsqueeze(0)
    consistency = torch.nn.functional.mse_loss(z, previous)
    prediction = torch.nn.functional.mse_loss(predicted, target_delta)
    teacher_loss = torch.nn.functional.mse_loss(student_adjusted, teacher_adjusted)
    loss = consistency + prediction + teacher_loss
    loss.backward(); optimizer.step()
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(teacher.parameters(), model.adapter.parameters()):
            teacher_parameter.mul_(config.teacher_ema_decay).add_(student_parameter, alpha=1.0 - config.teacher_ema_decay)
    return float(consistency.item()), float(prediction.item()), float(teacher_loss.item())


def run_target_online(source_model: GradualStateTCN, target: pd.DataFrame, config: GradualStateMonitoringV34Config,
                      *, setting: str, source_normalizer: RobustLocationScale | None = None) -> OnlineRun:
    """Execute a target run without accepting or reading target labels.

    The emitted row is frozen before any adaptation update.  Therefore appending
    future target rows cannot revise a historical score, event flag, or audit row.
    """
    frame = validate_online_frame(target, config, exact_columns=True)
    if len(frame) < config.calibration_windows:
        raise ValueError("Target domain needs 128 label-free calibration windows")
    raw = frame.loc[:, MAIN_FEATURES].to_numpy(float)
    direct = setting == "DirectTransfer"
    online = setting == "OnlineAdaptation"
    if setting not in {"DirectTransfer", "CalibrationOnly", "OnlineAdaptation", "TargetSupervisedUpperBound"}:
        raise ValueError(f"Unknown V3.4 setting: {setting}")
    if direct:
        if source_normalizer is None:
            raise ValueError("DirectTransfer requires its source normalization parameters")
        model = source_model
        transform = source_normalizer.transform
        normalizer: CausalTargetNormalizer | None = None
    else:
        model = source_model.target_copy()
        normalizer = CausalTargetNormalizer(raw[:config.calibration_windows], config)
        transform = normalizer.transform
    model.eval()
    model.freeze_for_target_adaptation()
    teacher = deepcopy(model.adapter).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=config.adapter_learning_rate)

    # Calibration emits no formal alert.  It initializes target-only robust score
    # distributions and the independent adapter state.
    static_encoded = _encode_static_sequence(model, raw, config, transform) if not online else None
    if normalizer is None:
        calibration_embeddings = [static_encoded[index][0] for index in range(config.calibration_windows)]
        distance_histories = [RobustScoreHistory([0.0], config) for _ in config.score_lags]
        raw_history = RobustScoreHistory([0.0], config); final_history = RobustScoreHistory([0.0], config)
    else:
        calibration_embeddings, distance_histories, raw_history, final_history = _initial_score_histories(
            model, raw, normalizer, config, encoded=static_encoded)
    embeddings = list(calibration_embeddings)
    rows: list[dict[str, object]] = []
    adaptation_rows: list[dict[str, object]] = []
    rising: list[float] = []
    high_run = 0
    for index in range(len(frame)):
        if index < config.calibration_windows:
            z = embeddings[index]
            source_probability = static_encoded[index][1] if static_encoded is not None else _encode(model, raw, index, config, transform)[1]
            rows.append(_score_row(frame, index, setting, "CALIBRATION", z, np.nan, np.nan, np.nan, np.nan,
                                   source_probability, np.nan, np.nan, 1, 0, 0, 0.0))
            continue
        z, source_probability = static_encoded[index] if static_encoded is not None else _encode(model, raw, index, config, transform)
        embeddings.append(z)
        distances = [float(np.linalg.norm(z - embeddings[index - lag])) for lag in config.score_lags]
        components = [history.standardise(value) for history, value in zip(distance_histories, distances)]
        transition_score = float(sum(weight * component for weight, component in zip(config.score_weights, components)))
        final_score = float((1.0 - config.source_probability_weight) * transition_score +
                            config.source_probability_weight * source_probability)
        transition_threshold = raw_history.quantile(config.score_baseline_quantile)
        candidate_threshold = max(config.candidate_min_score, final_history.quantile(config.score_baseline_quantile))
        rising.append(transition_score)
        if len(rising) > 3:
            rising.pop(0)
        strictly_rising = len(rising) == 3 and rising[0] < rising[1] < rising[2]
        high_run = high_run + 1 if final_score >= candidate_threshold else 0
        candidate_active = high_run >= config.candidate_min_windows
        frozen = bool(transition_score > transition_threshold or strictly_rising or candidate_active)
        updated = False; consistency = prediction = teacher_loss = np.nan
        if online and not frozen and index % config.adaptation_update_interval == 0:
            assert normalizer is not None
            consistency, prediction, teacher_loss = _adapter_update(model, teacher, raw, index, normalizer, optimizer, config)
            normalizer.update_after_output(raw[index])
            updated = True
        # Baselines update only in calm periods, after the score is committed.
        if not frozen:
            for history, value in zip(distance_histories, distances):
                history.append(value)
            raw_history.append(transition_score); final_history.append(final_score)
        rows.append(_score_row(frame, index, setting, "ONLINE", z, distances[0], distances[1], distances[2],
                               transition_score, source_probability, final_score, candidate_threshold,
                               int(frozen), int(updated), int(candidate_active), transition_threshold))
        adaptation_rows.append({
            "dataset": str(frame.dataset.iloc[index]), "setting": setting, "window_index": int(frame.window_index.iloc[index]),
            "center_cycle": float(frame.center_cycle.iloc[index]), "adapter_updated": int(updated), "adapter_frozen": int(frozen),
            "update_max_input_index": int(index), "current_index": int(index),
            "consistency_loss": consistency, "prediction_loss": prediction, "teacher_loss": teacher_loss,
            "normalizer_updates": int(normalizer.update_count) if normalizer is not None else 0,
        })
    return OnlineRun(pd.DataFrame(rows), pd.DataFrame(adaptation_rows))


def _score_row(frame: pd.DataFrame, index: int, setting: str, phase: str, z: np.ndarray,
               short: float, medium: float, long: float, transition: float, probability: float,
               final: float, threshold: float, frozen: int, updated: int, candidate: int,
               transition_threshold: float) -> dict[str, object]:
    return {
        "dataset": str(frame.dataset.iloc[index]), "setting": setting, "window_index": int(frame.window_index.iloc[index]),
        "center_cycle": float(frame.center_cycle.iloc[index]), "phase": phase,
        "embedding_0": float(z[0]), "embedding_15": float(z[-1]),
        "short_score": short, "medium_score": medium, "long_score": long,
        "transition_score": transition, "source_transition_probability": probability,
        "final_transition_score": final, "candidate_threshold": threshold,
        "transition_threshold": transition_threshold, "adapter_frozen": frozen,
        "adapter_updated": updated, "candidate_active": candidate,
    }


def extract_candidate_events(scores: pd.DataFrame, config: GradualStateMonitoringV34Config) -> pd.DataFrame:
    """Keep the single highest point in each merged high-score region."""
    online = scores[(scores.phase == "ONLINE") & scores.final_transition_score.notna()].reset_index(drop=True)
    rows: list[dict[str, object]] = []
    if online.empty:
        return pd.DataFrame(columns=["event_id", "start_cycle", "peak_cycle", "end_cycle", "peak_score"])
    high = online.final_transition_score.to_numpy(float) >= online.candidate_threshold.to_numpy(float)
    regions: list[tuple[int, int]] = []
    start: int | None = None; last: int | None = None
    for position in np.flatnonzero(high):
        position = int(position)
        if start is None:
            start = last = position
        elif position - int(last) <= config.candidate_merge_gap_windows + 1:
            last = position
        else:
            regions.append((int(start), int(last))); start = last = position
    if start is not None:
        regions.append((int(start), int(last)))
    prior_peak = -10**12
    event_id = 0
    for begin, end in regions:
        if end - begin + 1 < config.candidate_min_windows:
            continue
        region = online.iloc[begin:end + 1]
        peak_position = int(region.final_transition_score.astype(float).values.argmax())
        peak = region.iloc[peak_position]
        peak_index = int(peak.window_index)
        if peak_index - prior_peak < config.candidate_cooldown_windows:
            continue
        prior_peak = peak_index; event_id += 1
        rows.append({
            "event_id": event_id, "dataset": str(peak.dataset), "setting": str(peak.setting),
            "start_cycle": float(region.center_cycle.iloc[0]), "peak_cycle": float(peak.center_cycle),
            "end_cycle": float(region.center_cycle.iloc[-1]), "peak_score": float(peak.final_transition_score),
            "short_score": float(peak.short_score), "medium_score": float(peak.medium_score),
            "long_score": float(peak.long_score), "source_probability": float(peak.source_transition_probability),
            "adapter_frozen_fraction": float(region.adapter_frozen.mean()),
            "event_status": "unmatched_candidate_event",
        })
    return pd.DataFrame(rows)
