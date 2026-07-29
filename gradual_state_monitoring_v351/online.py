from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch

from gradual_state_monitoring_v33.data import MAIN_FEATURES
from gradual_state_monitoring_v34.data import validate_online_frame
from gradual_state_monitoring_v34.model import GradualStateTCN
from gradual_state_monitoring_v34.online import CausalTargetNormalizer, RobustScoreHistory, _adapter_update, _history
from gradual_state_monitoring_v35.online import extract_candidate_events as extract_v35_events
from gradual_state_monitoring_v35.online import run_target_online as run_v35_online

from .config import GradualStateMonitoringV351Config


@dataclass
class OnlineRun:
    scores: pd.DataFrame
    adaptation: pd.DataFrame
    events: pd.DataFrame
    event_audit: pd.DataFrame


@dataclass(frozen=True)
class ScoreCalibration:
    fast: RobustScoreHistory
    medium: RobustScoreHistory
    long: RobustScoreHistory
    residual_median: float
    residual_mad: float
    fast_warning: float
    fast_alarm: float
    slow_warning: float
    slow_alarm: float
    residual_warning: float
    residual_alarm: float
    rows: tuple[dict[str, float], ...]
    residual_z_history: tuple[float, ...]


@dataclass
class PendingEvent:
    start: int
    end: int
    high_indices: list[int]


def _encode_many(model: GradualStateTCN, raw: np.ndarray, indices: list[int], config: GradualStateMonitoringV351Config,
                 transform: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    histories = np.stack([_history(raw, index, config, transform) for index in indices])
    with torch.no_grad():
        z, logits, delta = model(torch.from_numpy(histories))
    return z.cpu().numpy().astype(float), torch.sigmoid(logits).cpu().numpy().astype(float), delta.cpu().numpy().astype(float)


def _refs(index: int, config: GradualStateMonitoringV351Config) -> list[int]:
    return [index, index - 16, index - 20, index - 64, index - 128]


def _measure(model: GradualStateTCN, raw: np.ndarray, index: int, config: GradualStateMonitoringV351Config,
             transform: Callable[[np.ndarray], np.ndarray]) -> dict[str, float | list[int]]:
    refs = _refs(index, config)
    if min(refs) < 0:
        raise ValueError("V3.5.1 score calibration requires complete 16/20/64/128 lags")
    z, probability, predicted = _encode_many(model, raw, refs, config, transform)
    observed = transform(raw[index]) - transform(raw[index - 20])
    return {
        "fast_distance": float(np.linalg.norm(z[0] - z[1])),
        "medium_distance": float(np.linalg.norm(z[0] - z[3])),
        "long_distance": float(np.linalg.norm(z[0] - z[4])),
        "source_transition_probability": float(probability[0]),
        "prediction_residual": float(np.linalg.norm(observed - predicted[2])),
        "refs": refs,
    }


def _score_calibration(model: GradualStateTCN, raw: np.ndarray, normalizer: CausalTargetNormalizer,
                       config: GradualStateMonitoringV351Config) -> ScoreCalibration:
    measured = [_measure(model, raw, index, config, normalizer.transform)
                for index in range(config.score_calibration_start, config.score_calibration_end)]
    fast = RobustScoreHistory([float(row["fast_distance"]) for row in measured], config)
    medium = RobustScoreHistory([float(row["medium_distance"]) for row in measured], config)
    long = RobustScoreHistory([float(row["long_distance"]) for row in measured], config)
    residual = np.asarray([float(row["prediction_residual"]) for row in measured], dtype=float)
    median = float(np.median(residual)); mad = max(float(np.median(np.abs(residual - median)) * 1.4826), config.score_minimum_scale)
    values: list[dict[str, float]] = []; residual_z: list[float] = []
    for row in measured:
        fast_score = fast.standardise(float(row["fast_distance"]))
        slow_score = (config.slow_weights[0] * medium.standardise(float(row["medium_distance"])) +
                      config.slow_weights[1] * long.standardise(float(row["long_distance"])))
        slow_final = .5 * slow_score + .5 * float(row["source_transition_probability"])
        z = max(0.0, (float(row["prediction_residual"]) - median) / mad)
        residual_z.append(z)
        values.append({"fast_score": fast_score, "slow_score": slow_score, "slow_final_score": slow_final,
                       "source_transition_probability": float(row["source_transition_probability"]),
                       "prediction_residual": float(row["prediction_residual"]), "residual_z": z,
                       "residual_persistent": float(np.median(residual_z[-config.residual_persistence_windows:]))})
    def threshold(column: str, quantile: float) -> float:
        return float(np.quantile([row[column] for row in values], quantile))
    return ScoreCalibration(
        fast=fast, medium=medium, long=long, residual_median=median, residual_mad=mad,
        fast_warning=threshold("fast_score", config.warning_quantile), fast_alarm=threshold("fast_score", config.alarm_quantile),
        slow_warning=threshold("slow_final_score", config.warning_quantile), slow_alarm=threshold("slow_final_score", config.alarm_quantile),
        residual_warning=threshold("residual_persistent", config.warning_quantile), residual_alarm=threshold("residual_persistent", config.alarm_quantile),
        rows=tuple(values), residual_z_history=tuple(residual_z))


def _row(frame: pd.DataFrame, index: int, setting: str, phase: str, value: dict[str, float], calibration: ScoreCalibration,
         slow_high: int, warning: int, pending: int, frozen: int, updated: int) -> dict[str, object]:
    return {
        "dataset": str(frame.dataset.iloc[index]), "setting": setting, "window_index": int(frame.window_index.iloc[index]),
        "center_cycle": float(frame.center_cycle.iloc[index]), "phase": phase,
        "fast_score": value.get("fast_score", np.nan), "slow_score": value.get("slow_score", np.nan),
        "slow_final_score": value.get("slow_final_score", np.nan),
        "source_transition_probability": value.get("source_transition_probability", np.nan),
        "prediction_residual": value.get("prediction_residual", np.nan), "residual_z": value.get("residual_z", np.nan),
        "residual_persistent": value.get("residual_persistent", np.nan),
        "fast_warning_threshold": calibration.fast_warning, "fast_alarm_threshold": calibration.fast_alarm,
        "slow_warning_threshold": calibration.slow_warning, "slow_alarm_threshold": calibration.slow_alarm,
        "residual_warning_threshold": calibration.residual_warning, "residual_alarm_threshold": calibration.residual_alarm,
        "slow_high": int(slow_high), "adaptation_warning": int(warning), "pending_event": int(pending),
        "adapter_frozen": int(frozen), "adapter_updated": int(updated),
        "final_transition_score": value.get("slow_final_score", np.nan),
    }


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["event_id", "dataset", "setting", "start_cycle", "peak_cycle", "end_cycle", "emit_cycle",
                                 "peak_score", "event_rank_score", "residual_event_peak", "residual_high_count",
                                 "residual_high_fraction", "slow_excess_area", "event_duration", "peak_slow_high",
                                 "adapter_frozen_fraction", "event_status"])


def _event_evidence(event: PendingEvent, rows_by_index: dict[int, dict[str, object]], current_index: int,
                    config: GradualStateMonitoringV351Config) -> dict[str, object]:
    support = [rows_by_index[index] for index in range(max(0, event.start - 5), event.end + 6) if index in rows_by_index]
    high_rows = [rows_by_index[index] for index in event.high_indices if index in rows_by_index]
    peak = max(high_rows, key=lambda row: float(row["slow_final_score"]))
    residuals = np.asarray([float(row["residual_persistent"]) for row in support], dtype=float)
    slow = np.asarray([float(row["slow_final_score"]) for row in high_rows], dtype=float)
    alarm = float(peak["slow_alarm_threshold"]); residual_alarm = float(peak["residual_alarm_threshold"])
    residual_peak = float(np.max(residuals)) if len(residuals) else 0.0
    high_count = int(np.sum(residuals >= residual_alarm)); fraction = float(high_count / len(residuals)) if len(residuals) else 0.0
    slow_excess = float(np.sum(np.maximum(0.0, slow / max(alarm, config.score_minimum_scale) - 1.0)))
    slow_peak_excess = max(0.0, float(peak["slow_final_score"]) / max(alarm, config.score_minimum_scale) - 1.0)
    residual_excess = max(0.0, residual_peak / max(residual_alarm, config.score_minimum_scale) - 1.0)
    return {
        "event_id": 0, "dataset": str(peak["dataset"]), "setting": str(peak["setting"]),
        "start_cycle": float(rows_by_index[event.start]["center_cycle"]), "peak_cycle": float(peak["center_cycle"]),
        "end_cycle": float(rows_by_index[event.end]["center_cycle"]), "emit_cycle": float(rows_by_index[current_index]["center_cycle"]),
        "peak_score": float(peak["slow_final_score"]), "event_rank_score": slow_peak_excess + .5 * residual_excess,
        "residual_event_peak": residual_peak, "residual_high_count": high_count, "residual_high_fraction": fraction,
        "slow_excess_area": slow_excess, "event_duration": int(event.end - event.start + 1), "peak_slow_high": int(peak["slow_high"]),
        "adapter_frozen_fraction": float(np.mean([float(row["adapter_frozen"]) for row in support])) if support else 1.0,
    }


def run_target_online(source_model: GradualStateTCN, target: pd.DataFrame, config: GradualStateMonitoringV351Config,
                      *, setting: str) -> OnlineRun:
    if setting not in set(config.v351_settings).difference({"V35_HardAND_Current"}):
        raise ValueError(setting)
    frame = validate_online_frame(target, config, exact_columns=True)
    if len(frame) < config.score_calibration_end:
        raise ValueError("V3.5.1 target sequence needs at least 256 windows for two-phase calibration")
    raw = frame.loc[:, MAIN_FEATURES].to_numpy(float)
    model = source_model.target_copy(); model.eval(); model.freeze_for_target_adaptation()
    normalizer = CausalTargetNormalizer(raw[:config.normalizer_calibration_windows], config)
    teacher = deepcopy(model.adapter).eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=config.adapter_learning_rate)
    calibration = _score_calibration(model, raw, normalizer, config)
    warning_freeze = "WarningFreeze" in setting
    rerank = setting.startswith("V351_ResidualRerank")
    update_interval = 5 if setting.endswith("Update5") else config.adaptation_update_interval
    rows: list[dict[str, object]] = []; audit: list[dict[str, object]] = []; events: list[dict[str, object]] = []; event_audit: list[dict[str, object]] = []
    rows_by_index: dict[int, dict[str, object]] = {}
    for index in range(config.normalizer_calibration_windows):
        row = _row(frame, index, setting, "NORMALIZER_CALIBRATION", {}, calibration, 0, 0, 0, 1, 0); rows.append(row); rows_by_index[index] = row
        audit.append({"dataset": str(frame.dataset.iloc[index]), "setting": setting, "window_index": int(frame.window_index.iloc[index]), "phase": "NORMALIZER_CALIBRATION", "reference_indices": "", "complete_lags": 0, "score_before_update": 1, "adapter_frozen": 1, "adapter_updated": 0, "normalizer_updates": 0})
    for offset, value in enumerate(calibration.rows, start=config.score_calibration_start):
        row = _row(frame, offset, setting, "SCORE_CALIBRATION", value, calibration, 0, 0, 0, 1, 0); rows.append(row); rows_by_index[offset] = row
        audit.append({"dataset": str(frame.dataset.iloc[offset]), "setting": setting, "window_index": int(frame.window_index.iloc[offset]), "phase": "SCORE_CALIBRATION", "reference_indices": "|".join(map(str, _refs(offset, config))), "complete_lags": 1, "score_before_update": 1, "adapter_frozen": 1, "adapter_updated": 0, "normalizer_updates": 0})
    residual_history = list(calibration.residual_z_history); high_start: int | None = None; high_indices: list[int] = []; pending: list[PendingEvent] = []
    safe_run = 0; prior_peak = -10**9
    for index in range(config.score_calibration_end, len(frame)):
        measured = _measure(model, raw, index, config, normalizer.transform)
        residual_z = max(0.0, (float(measured["prediction_residual"]) - calibration.residual_median) / calibration.residual_mad)
        residual_history.append(residual_z)
        value = {"fast_score": calibration.fast.standardise(float(measured["fast_distance"])),
                 "slow_score": config.slow_weights[0] * calibration.medium.standardise(float(measured["medium_distance"])) + config.slow_weights[1] * calibration.long.standardise(float(measured["long_distance"])),
                 "source_transition_probability": float(measured["source_transition_probability"]), "prediction_residual": float(measured["prediction_residual"]),
                 "residual_z": residual_z, "residual_persistent": float(np.median(residual_history[-config.residual_persistence_windows:]))}
        value["slow_final_score"] = .5 * value["slow_score"] + .5 * value["source_transition_probability"]
        slow_high = bool(value["slow_final_score"] >= calibration.slow_alarm)
        warning = bool(value["fast_score"] >= calibration.fast_warning or value["slow_final_score"] >= calibration.slow_warning or value["residual_persistent"] >= calibration.residual_warning)
        if slow_high:
            if high_start is None: high_start = index; high_indices = []
            high_indices.append(index)
        elif high_start is not None:
            if len(high_indices) >= config.candidate_min_windows:
                candidate = PendingEvent(high_start, high_indices[-1], list(high_indices))
                if pending and candidate.start - pending[-1].end - 1 <= config.candidate_merge_gap_windows:
                    pending[-1].end = candidate.end; pending[-1].high_indices.extend(candidate.high_indices)
                else:
                    pending.append(candidate)
            high_start = None; high_indices = []
        pending_now = bool(high_start is not None or pending)
        warning = warning or (warning_freeze and pending_now)
        safe = not warning if warning_freeze else bool(value["fast_score"] < calibration.fast_alarm and value["slow_final_score"] < calibration.slow_alarm and value["residual_persistent"] < calibration.residual_alarm)
        safe_run = safe_run + 1 if safe else 0
        required_safe = config.warning_safe_windows if warning_freeze else config.adaptation_safe_windows
        frozen = safe_run < required_safe
        row = _row(frame, index, setting, "ONLINE", value, calibration, int(slow_high), int(warning), int(pending_now), int(frozen), 0)
        rows.append(row); rows_by_index[index] = row
        updated = False
        if not frozen and index % update_interval == 0:
            _adapter_update(model, teacher, raw, index, normalizer, optimizer, config); normalizer.update_after_output(raw[index]); updated = True; row["adapter_updated"] = 1
        audit.append({"dataset": str(frame.dataset.iloc[index]), "setting": setting, "window_index": int(frame.window_index.iloc[index]), "phase": "ONLINE", "reference_indices": "|".join(map(str, measured["refs"])), "complete_lags": 1, "score_before_update": 1, "adapter_frozen": int(frozen), "adapter_updated": int(updated), "normalizer_updates": int(normalizer.update_count), "safe_run": int(safe_run), "pending_event": int(pending_now), "residual_origin_index": int(measured["refs"][2]), "maximum_online_input_index": int(index)})
        # Waiting 20+2 windows protects causal merge semantics; end+5 support is therefore already observed.
        completed: list[PendingEvent] = []
        for candidate in pending:
            active_can_merge = high_start is not None and high_start - candidate.end - 1 <= config.candidate_merge_gap_windows
            if index >= candidate.end + config.candidate_merge_gap_windows + config.candidate_min_windows - 1 and not active_can_merge:
                evidence = _event_evidence(candidate, rows_by_index, index, config); confirmed = rerank or evidence["residual_high_count"] >= config.event_residual_min_high_count
                cooldown = int(candidate.high_indices[np.argmax([float(rows_by_index[item]["slow_final_score"]) for item in candidate.high_indices])]) - prior_peak < config.candidate_cooldown_windows
                status = "CONFIRMED" if confirmed and not cooldown else ("COOLDOWN_SUPPRESSED" if cooldown else "RESIDUAL_REJECTED")
                event_audit.append({**evidence, "event_start_window": candidate.start, "event_end_window": candidate.end,
                                    "event_emit_window": index, "event_status": status})
                if status == "CONFIRMED":
                    evidence["event_id"] = len(events) + 1; evidence["event_status"] = "unmatched_candidate_event"
                    if not rerank: evidence["event_rank_score"] = evidence["peak_score"]
                    events.append(evidence); prior_peak = candidate.high_indices[np.argmax([float(rows_by_index[item]["slow_final_score"]) for item in candidate.high_indices])]
                completed.append(candidate)
        pending = [candidate for candidate in pending if candidate not in completed]
    return OnlineRun(pd.DataFrame(rows), pd.DataFrame(audit), pd.DataFrame(events, columns=_empty_events().columns), pd.DataFrame(event_audit))


def run_v35_hard_and(source_model: GradualStateTCN, target: pd.DataFrame, config: GradualStateMonitoringV351Config) -> OnlineRun:
    """Unmodified V3.5 SlowResidual OnlineAdaptation control, mapped into the V3.5.1 schema."""
    old = run_v35_online(source_model, target, config, setting="V35_SlowResidual_OnlineAdaptation")
    scores = old.scores.copy(); scores["setting"] = "V35_HardAND_Current"; scores["phase"] = scores.phase.replace({"CALIBRATION": "V35_CALIBRATION"})
    scores["fast_warning_threshold"] = scores.fast_threshold; scores["fast_alarm_threshold"] = scores.fast_threshold
    scores["slow_warning_threshold"] = scores.slow_threshold; scores["slow_alarm_threshold"] = scores.slow_threshold
    scores["residual_warning_threshold"] = scores.residual_threshold; scores["residual_alarm_threshold"] = scores.residual_threshold
    scores["slow_high"] = (scores.slow_final_score >= scores.slow_alarm_threshold).fillna(False).astype(int)
    scores["adaptation_warning"] = scores.adapter_frozen; scores["pending_event"] = scores.candidate_active
    audit = old.adaptation.copy(); audit["setting"] = "V35_HardAND_Current"; audit["phase"] = "ONLINE"; audit["complete_lags"] = 1; audit["score_before_update"] = 1
    events = extract_v35_events(old.scores, config)
    if len(events):
        events["setting"] = "V35_HardAND_Current"; events["emit_cycle"] = events.end_cycle; events["event_rank_score"] = events.peak_score
        events["residual_event_peak"] = np.nan; events["residual_high_count"] = np.nan; events["residual_high_fraction"] = np.nan; events["slow_excess_area"] = np.nan; events["event_duration"] = np.nan; events["peak_slow_high"] = 1
    return OnlineRun(scores, audit, events if len(events) else _empty_events(), pd.DataFrame())
