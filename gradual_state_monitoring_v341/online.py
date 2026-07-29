from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from gradual_state_monitoring_v33.data import MAIN_FEATURES
from gradual_state_monitoring_v34.data import validate_online_frame
from gradual_state_monitoring_v34.model import GradualStateTCN
from gradual_state_monitoring_v34.online import CausalTargetNormalizer, RobustScoreHistory, _adapter_update, _history
from gradual_state_monitoring_v34.training import RobustLocationScale

from .config import GradualStateMonitoringV341Config


@dataclass(frozen=True)
class SourceScoreBaseline:
    distance_histories: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    transition_history: tuple[float, ...]
    final_history: tuple[float, ...]
    candidate_threshold: float


@dataclass
class OnlineRun:
    scores: pd.DataFrame
    adaptation: pd.DataFrame


def _encode_many(model: GradualStateTCN, raw: np.ndarray, indices: list[int], config: GradualStateMonitoringV341Config,
                 transform: callable) -> tuple[np.ndarray, np.ndarray]:
    histories = np.stack([_history(raw, index, config, transform) for index in indices])
    with torch.no_grad():
        z, logits, _ = model(torch.from_numpy(histories))
    return z.cpu().numpy().astype(float), torch.sigmoid(logits).cpu().numpy().astype(float)


def _score(z: np.ndarray, probabilities: np.ndarray, histories: list[RobustScoreHistory],
           config: GradualStateMonitoringV341Config) -> tuple[list[float], float, float]:
    distances = [float(np.linalg.norm(z[0] - z[position])) for position in (1, 2, 3)]
    components = [history.standardise(value) for history, value in zip(histories, distances)]
    transition = float(sum(weight * value for weight, value in zip(config.score_weights, components)))
    final = float((1.0 - config.source_probability_weight) * transition + config.source_probability_weight * probabilities[0])
    return distances, transition, final


def _make_histories(seed: list[list[float]], config: GradualStateMonitoringV341Config) -> list[RobustScoreHistory]:
    return [RobustScoreHistory(values, config) for values in seed]


def build_source_score_baseline(model: GradualStateTCN, source: pd.DataFrame, normalizer: RobustLocationScale,
                                config: GradualStateMonitoringV341Config) -> SourceScoreBaseline:
    """Build source-only score histories after source training; no target data appears here."""
    raw = source.loc[:, MAIN_FEATURES].to_numpy(float)
    seed = [[], [], []]; transitions: list[float] = []; finals: list[float] = []
    provisional = _make_histories([[0.0], [0.0], [0.0]], config)
    for index in range(config.calibration_windows, len(raw)):
        refs = [index, index - 16, index - 64, index - 128]
        z, p = _encode_many(model, raw, refs, config, normalizer.transform)
        distances, transition, final = _score(z, p, provisional, config)
        for bucket, value in zip(seed, distances): bucket.append(value)
        transitions.append(transition); finals.append(final)
        for history, value in zip(provisional, distances): history.append(value)
    histories = _make_histories([values or [0.0] for values in seed], config)
    transition_history = tuple(transitions or [0.0]); final_history = tuple(finals or [0.0])
    candidate = max(config.candidate_min_score, float(np.quantile(final_history, config.score_baseline_quantile)))
    return SourceScoreBaseline(tuple(tuple(history.values) for history in histories), transition_history, final_history, candidate)


def _initial_target_histories(model: GradualStateTCN, raw: np.ndarray, normalizer: CausalTargetNormalizer,
                              config: GradualStateMonitoringV341Config) -> tuple[list[RobustScoreHistory], RobustScoreHistory, RobustScoreHistory, float]:
    seed = [[], [], []]
    for index in range(128):
        refs = [index, max(0, index - 16), max(0, index - 64), max(0, index - 128)]
        z, _ = _encode_many(model, raw, refs, config, normalizer.transform)
        for bucket, value in zip(seed, [float(np.linalg.norm(z[0] - z[position])) for position in (1, 2, 3)]): bucket.append(value)
    histories = _make_histories(seed, config)
    transition = RobustScoreHistory([0.0], config); final = RobustScoreHistory([0.0], config)
    for index in range(128):
        refs = [index, max(0, index - 16), max(0, index - 64), max(0, index - 128)]
        z, p = _encode_many(model, raw, refs, config, normalizer.transform)
        _, value, score = _score(z, p, histories, config); transition.append(value); final.append(score)
    return histories, transition, final, max(config.candidate_min_score, final.quantile(config.score_baseline_quantile))


def _persistent_active(high: list[bool], minimum: int) -> bool:
    return len(high) >= minimum and all(high[-minimum:])


def _single_branch(source_model: GradualStateTCN, target: pd.DataFrame, config: GradualStateMonitoringV341Config, *,
                   setting: str, source_normalizer: RobustLocationScale, source_baseline: SourceScoreBaseline | None,
                   adaptive: bool) -> OnlineRun:
    frame = validate_online_frame(target, config, exact_columns=True); raw = frame.loc[:, MAIN_FEATURES].to_numpy(float)
    direct = setting == "DirectTransfer"
    model = source_model if direct else source_model.target_copy()
    normalizer = None if direct else CausalTargetNormalizer(raw[:config.calibration_windows], config)
    transform = source_normalizer.transform if direct else normalizer.transform
    model.eval(); model.freeze_for_target_adaptation(); teacher = deepcopy(model.adapter).eval()
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=config.adapter_learning_rate)
    if direct:
        assert source_baseline is not None
        histories = _make_histories([list(values) for values in source_baseline.distance_histories], config)
        transition_history = RobustScoreHistory(source_baseline.transition_history, config)
        final_history = RobustScoreHistory(source_baseline.final_history, config)
        candidate_threshold = source_baseline.candidate_threshold
    else:
        assert normalizer is not None
        histories, transition_history, final_history, candidate_threshold = _initial_target_histories(model, raw, normalizer, config)
    rows: list[dict[str, object]] = []; audit: list[dict[str, object]] = []; high: list[bool] = []
    for index in range(len(frame)):
        if index < config.calibration_windows:
            rows.append({"dataset": str(frame.dataset.iloc[index]), "setting": setting, "window_index": int(frame.window_index.iloc[index]), "center_cycle": float(frame.center_cycle.iloc[index]), "phase": "CALIBRATION", "transition_score": np.nan, "final_transition_score": np.nan, "candidate_threshold": candidate_threshold, "candidate_active": 0, "adapter_updated": 0, "adapter_frozen": 1})
            continue
        refs = [index, index - 16, index - 64, index - 128]
        z, p = _encode_many(model, raw, refs, config, transform)  # all four use current adapter/normalizer
        distances, transition, final = _score(z, p, histories, config)
        high.append(final >= candidate_threshold)
        active = _persistent_active(high, config.candidate_min_windows)
        frozen = bool(transition > transition_history.quantile(config.score_baseline_quantile) or active)
        updated = False
        if adaptive and not frozen and index % config.adaptation_update_interval == 0:
            assert normalizer is not None
            _adapter_update(model, teacher, raw, index, normalizer, optimizer, config); normalizer.update_after_output(raw[index]); updated = True
        if not frozen and not direct:
            for history, value in zip(histories, distances): history.append(value)
            transition_history.append(transition); final_history.append(final)
        rows.append({"dataset": str(frame.dataset.iloc[index]), "setting": setting, "window_index": int(frame.window_index.iloc[index]), "center_cycle": float(frame.center_cycle.iloc[index]), "phase": "ONLINE", "short_score": distances[0], "medium_score": distances[1], "long_score": distances[2], "transition_score": transition, "source_transition_probability": float(p[0]), "final_transition_score": final, "candidate_threshold": candidate_threshold, "candidate_active": int(active), "adapter_updated": int(updated), "adapter_frozen": int(frozen)})
        audit.append({"dataset": str(frame.dataset.iloc[index]), "setting": setting, "window_index": int(frame.window_index.iloc[index]), "current_index": index, "reference_indices": "|".join(map(str, refs[1:])), "references_reencoded_current_model": 1, "adapter_updated": int(updated), "adapter_frozen": int(frozen)})
    return OnlineRun(pd.DataFrame(rows), pd.DataFrame(audit))


def run_target_online(source_model: GradualStateTCN, target: pd.DataFrame, config: GradualStateMonitoringV341Config, *,
                      setting: str, source_normalizer: RobustLocationScale, source_baseline: SourceScoreBaseline | None = None) -> OnlineRun:
    if setting in {"DirectTransfer", "CalibrationOnly", "OnlineAdaptation", "TargetSupervisedReference"}:
        return _single_branch(source_model, target, config, setting=setting, source_normalizer=source_normalizer, source_baseline=source_baseline, adaptive=setting == "OnlineAdaptation")
    if setting != "DualBranchAdaptation": raise ValueError(setting)
    frozen = _single_branch(source_model, target, config, setting="DualBranchFrozen", source_normalizer=source_normalizer, source_baseline=None, adaptive=False)
    adaptive = _single_branch(source_model, target, config, setting="DualBranchAdaptive", source_normalizer=source_normalizer, source_baseline=None, adaptive=True)
    scores = frozen.scores.copy(); other = adaptive.scores
    scores["setting"] = setting; scores["score_frozen"] = scores.final_transition_score; scores["score_adaptive"] = other.final_transition_score.to_numpy()
    scores["final_transition_score"] = scores.score_frozen + 0.3 * np.maximum(0.0, scores.score_adaptive - scores.score_frozen)
    scores["candidate_active"] = (scores.final_transition_score >= scores.candidate_threshold).rolling(config.candidate_min_windows, min_periods=config.candidate_min_windows).sum().fillna(0).ge(config.candidate_min_windows).astype(int)
    audit = pd.concat([frozen.adaptation.assign(branch="frozen"), adaptive.adaptation.assign(branch="adaptive")], ignore_index=True)
    return OnlineRun(scores, audit)


def extract_candidate_events(scores: pd.DataFrame, config: GradualStateMonitoringV341Config) -> pd.DataFrame:
    online = scores[(scores.phase == "ONLINE") & scores.final_transition_score.notna()].reset_index(drop=True)
    high = online.final_transition_score.to_numpy(float) >= online.candidate_threshold.to_numpy(float)
    raw_regions: list[tuple[int, int]] = []; start = None
    for pos, value in enumerate(high):
        if value and start is None: start = pos
        if start is not None and (not value or pos == len(high) - 1):
            end = pos if value and pos == len(high) - 1 else pos - 1
            if end - start + 1 >= config.candidate_min_windows: raw_regions.append((start, end))
            start = None
    merged: list[tuple[int, int]] = []
    for region in raw_regions:
        if merged and region[0] - merged[-1][1] - 1 <= config.candidate_merge_gap_windows: merged[-1] = (merged[-1][0], region[1])
        else: merged.append(region)
    rows = []; previous_peak = -10**9
    for event_id, (begin, end) in enumerate(merged, start=1):
        region = online.iloc[begin:end + 1]; peak = region.iloc[int(region.final_transition_score.to_numpy(float).argmax())]
        if int(peak.window_index) - previous_peak < config.candidate_cooldown_windows: continue
        previous_peak = int(peak.window_index)
        rows.append({"event_id": event_id, "dataset": str(peak.dataset), "setting": str(peak.setting), "start_cycle": float(region.center_cycle.iloc[0]), "peak_cycle": float(peak.center_cycle), "end_cycle": float(region.center_cycle.iloc[-1]), "peak_score": float(peak.final_transition_score), "adapter_frozen_fraction": float(region.adapter_frozen.mean()), "event_status": "unmatched_candidate_event"})
    columns = ["event_id", "dataset", "setting", "start_cycle", "peak_cycle", "end_cycle", "peak_score", "adapter_frozen_fraction", "event_status"]
    return pd.DataFrame(rows, columns=columns)
