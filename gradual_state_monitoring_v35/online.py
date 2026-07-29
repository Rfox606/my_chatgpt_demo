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
from gradual_state_monitoring_v34.training import RobustLocationScale
from gradual_state_monitoring_v341.online import extract_candidate_events as extract_v341_events
from gradual_state_monitoring_v341.online import run_target_online as run_v341_online

from .config import GradualStateMonitoringV35Config


@dataclass
class OnlineRun:
    scores: pd.DataFrame
    adaptation: pd.DataFrame


@dataclass(frozen=True)
class ScoreCalibration:
    fast_history: RobustScoreHistory
    medium_history: RobustScoreHistory
    long_history: RobustScoreHistory
    residual_median: float
    residual_mad: float
    fast_threshold: float
    slow_threshold: float
    residual_threshold: float
    calibration_rows: tuple[dict[str, float], ...]
    residual_z_history: tuple[float, ...]


def _encode_many(model: GradualStateTCN, raw: np.ndarray, indices: list[int],
                 config: GradualStateMonitoringV35Config, transform: Callable[[np.ndarray], np.ndarray]
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All reference histories are re-encoded by the model and normalizer active at this time."""
    histories = np.stack([_history(raw, index, config, transform) for index in indices])
    with torch.no_grad():
        embeddings, logits, predicted_delta = model(torch.from_numpy(histories))
    return (embeddings.cpu().numpy().astype(float), torch.sigmoid(logits).cpu().numpy().astype(float),
            predicted_delta.cpu().numpy().astype(float))


def _references(index: int, config: GradualStateMonitoringV35Config) -> list[int]:
    return [index, max(0, index - config.fast_lag), max(0, index - config.prediction_lag),
            max(0, index - config.slow_lags[0]), max(0, index - config.slow_lags[1])]


def _measurement(model: GradualStateTCN, raw: np.ndarray, index: int, config: GradualStateMonitoringV35Config,
                 transform: Callable[[np.ndarray], np.ndarray]) -> dict[str, float | list[int]]:
    refs = _references(index, config)
    z, probability, predicted_delta = _encode_many(model, raw, refs, config, transform)
    fast_distance = float(np.linalg.norm(z[0] - z[1]))
    medium_distance = float(np.linalg.norm(z[0] - z[3]))
    long_distance = float(np.linalg.norm(z[0] - z[4]))
    observed_delta = transform(raw[index]) - transform(raw[refs[2]])
    residual = float(np.linalg.norm(observed_delta - predicted_delta[2]))
    return {
        "fast_distance": fast_distance,
        "medium_distance": medium_distance,
        "long_distance": long_distance,
        "source_transition_probability": float(probability[0]),
        "prediction_residual": residual,
        "references": refs,
    }


def _initial_calibration(model: GradualStateTCN, raw: np.ndarray, normalizer: CausalTargetNormalizer,
                         config: GradualStateMonitoringV35Config) -> ScoreCalibration:
    measured = [_measurement(model, raw, index, config, normalizer.transform) for index in range(config.calibration_windows)]
    fast = RobustScoreHistory([float(value["fast_distance"]) for value in measured], config)
    medium = RobustScoreHistory([float(value["medium_distance"]) for value in measured], config)
    long = RobustScoreHistory([float(value["long_distance"]) for value in measured], config)
    residuals = np.asarray([float(value["prediction_residual"]) for value in measured], dtype=float)
    residual_median = float(np.median(residuals))
    residual_mad = max(float(np.median(np.abs(residuals - residual_median)) * 1.4826), config.score_minimum_scale)
    rows: list[dict[str, float]] = []
    residual_z_values: list[float] = []
    for value in measured:
        fast_score = fast.standardise(float(value["fast_distance"]))
        slow_score = (config.slow_weights[0] * medium.standardise(float(value["medium_distance"])) +
                      config.slow_weights[1] * long.standardise(float(value["long_distance"])))
        slow_final = 0.5 * slow_score + 0.5 * float(value["source_transition_probability"])
        residual_z = max(0.0, (float(value["prediction_residual"]) - residual_median) / residual_mad)
        residual_z_values.append(residual_z)
        persistent = float(np.median(residual_z_values[-config.residual_persistence_windows:]))
        rows.append({
            "fast_score": fast_score, "slow_score": slow_score, "slow_final_score": slow_final,
            "prediction_residual": float(value["prediction_residual"]), "residual_z": residual_z,
            "residual_persistent": persistent,
            "source_transition_probability": float(value["source_transition_probability"]),
        })
    return ScoreCalibration(
        fast_history=fast, medium_history=medium, long_history=long,
        residual_median=residual_median, residual_mad=residual_mad,
        fast_threshold=float(np.quantile([row["fast_score"] for row in rows], config.score_baseline_quantile)),
        slow_threshold=float(np.quantile([row["slow_final_score"] for row in rows], config.score_baseline_quantile)),
        residual_threshold=float(np.quantile([row["residual_persistent"] for row in rows], config.score_baseline_quantile)),
        calibration_rows=tuple(rows), residual_z_history=tuple(residual_z_values),
    )


def _row(frame: pd.DataFrame, index: int, setting: str, phase: str, value: dict[str, float],
         calibration: ScoreCalibration, candidate_high: int, candidate_active: int, frozen: int, updated: int) -> dict[str, object]:
    return {
        "dataset": str(frame.dataset.iloc[index]), "setting": setting, "window_index": int(frame.window_index.iloc[index]),
        "center_cycle": float(frame.center_cycle.iloc[index]), "phase": phase,
        "fast_score": value["fast_score"], "slow_score": value["slow_score"],
        "source_transition_probability": value["source_transition_probability"],
        "slow_final_score": value["slow_final_score"], "prediction_residual": value["prediction_residual"],
        "residual_z": value["residual_z"], "residual_persistent": value["residual_persistent"],
        "fast_threshold": calibration.fast_threshold, "slow_threshold": calibration.slow_threshold,
        "residual_threshold": calibration.residual_threshold, "candidate_high": int(candidate_high),
        "candidate_active": int(candidate_active), "adapter_frozen": int(frozen), "adapter_updated": int(updated),
        # The established evaluator ranks a formal score under this column.  The residual is a gate, not a score term.
        "final_transition_score": value["slow_final_score"],
        "candidate_threshold": calibration.slow_threshold,
    }


def run_target_online(source_model: GradualStateTCN, target: pd.DataFrame, config: GradualStateMonitoringV35Config,
                      *, setting: str) -> OnlineRun:
    """Run one V3.5 ablation without accepting target labels or experiment-end information."""
    if setting not in {
        "V35_SlowOnly_CalibrationOnly", "V35_SlowOnly_OnlineAdaptation",
        "V35_SlowResidual_CalibrationOnly", "V35_SlowResidual_OnlineAdaptation",
    }:
        raise ValueError(f"Unknown V3.5 setting: {setting}")
    frame = validate_online_frame(target, config, exact_columns=True)
    if len(frame) < config.calibration_windows:
        raise ValueError("Target domain needs 128 label-free calibration windows")
    raw = frame.loc[:, MAIN_FEATURES].to_numpy(float)
    online_adaptation = setting.endswith("OnlineAdaptation")
    residual_gate = "SlowResidual" in setting
    model = source_model.target_copy(); model.eval(); model.freeze_for_target_adaptation()
    normalizer = CausalTargetNormalizer(raw[:config.calibration_windows], config)
    teacher = deepcopy(model.adapter).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=config.adapter_learning_rate)
    calibration = _initial_calibration(model, raw, normalizer, config)
    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for index, value in enumerate(calibration.calibration_rows):
        rows.append(_row(frame, index, setting, "CALIBRATION", value, calibration, 0, 0, 1, 0))
        audit.append({
            "dataset": str(frame.dataset.iloc[index]), "setting": setting, "window_index": int(frame.window_index.iloc[index]),
            "phase": "CALIBRATION", "reference_indices": "|".join(map(str, _references(index, config))),
            "references_reencoded_current_model": 1, "score_computed_before_update": 1,
            "residual_origin_index": int(max(0, index - config.prediction_lag)), "maximum_online_input_index": int(index),
            "adapter_frozen": 1, "adapter_updated": 0, "normalizer_updates": 0,
        })
    residual_history = list(calibration.residual_z_history)
    high_run = 0; safe_run = 0
    for index in range(config.calibration_windows, len(frame)):
        measured = _measurement(model, raw, index, config, normalizer.transform)
        residual_z = max(0.0, (float(measured["prediction_residual"]) - calibration.residual_median) / calibration.residual_mad)
        residual_history.append(residual_z)
        value = {
            "fast_score": calibration.fast_history.standardise(float(measured["fast_distance"])),
            "slow_score": (config.slow_weights[0] * calibration.medium_history.standardise(float(measured["medium_distance"])) +
                           config.slow_weights[1] * calibration.long_history.standardise(float(measured["long_distance"]))),
            "source_transition_probability": float(measured["source_transition_probability"]),
            "prediction_residual": float(measured["prediction_residual"]), "residual_z": residual_z,
            "residual_persistent": float(np.median(residual_history[-config.residual_persistence_windows:])),
        }
        value["slow_final_score"] = 0.5 * value["slow_score"] + 0.5 * value["source_transition_probability"]
        slow_high = value["slow_final_score"] >= calibration.slow_threshold
        residual_high = value["residual_persistent"] >= calibration.residual_threshold
        candidate_high = bool(slow_high and (residual_high if residual_gate else True))
        high_run = high_run + 1 if candidate_high else 0
        candidate_active = high_run >= config.candidate_min_windows
        safe = bool(value["fast_score"] < calibration.fast_threshold and not slow_high and not residual_high)
        safe_run = safe_run + 1 if safe else 0
        frozen = True if not online_adaptation else safe_run < config.adaptation_safe_windows
        updated = False
        # Commit the current score and gate decision before an adaptation call can change either model state.
        rows.append(_row(frame, index, setting, "ONLINE", value, calibration, int(candidate_high), int(candidate_active), int(frozen), 0))
        # This is deliberately after every value above has been committed to local output state.
        if online_adaptation and not frozen and index % config.adaptation_update_interval == 0:
            _adapter_update(model, teacher, raw, index, normalizer, optimizer, config)
            normalizer.update_after_output(raw[index]); updated = True
        if updated:
            rows[-1]["adapter_updated"] = 1
        audit.append({
            "dataset": str(frame.dataset.iloc[index]), "setting": setting, "window_index": int(frame.window_index.iloc[index]),
            "phase": "ONLINE", "reference_indices": "|".join(map(str, measured["references"])),
            "references_reencoded_current_model": 1, "score_computed_before_update": 1,
            "residual_origin_index": int(measured["references"][2]), "maximum_online_input_index": int(index),
            "adapter_frozen": int(frozen), "adapter_updated": int(updated),
            "normalizer_updates": int(normalizer.update_count), "safe_gate_consecutive_windows": int(safe_run),
        })
    return OnlineRun(pd.DataFrame(rows), pd.DataFrame(audit))


def extract_candidate_events(scores: pd.DataFrame, config: GradualStateMonitoringV35Config) -> pd.DataFrame:
    """V3.4.1 event protocol applied to the already-gated V3.5 candidate-high sequence."""
    online = scores[(scores.phase == "ONLINE") & scores.slow_final_score.notna()].reset_index(drop=True)
    regions: list[tuple[int, int]] = []; start: int | None = None
    high = online.candidate_high.to_numpy(bool)
    for position, value in enumerate(high):
        if value and start is None:
            start = position
        if start is not None and (not value or position == len(high) - 1):
            end = position if value and position == len(high) - 1 else position - 1
            if end - start + 1 >= config.candidate_min_windows:
                regions.append((start, end))
            start = None
    merged: list[tuple[int, int]] = []
    for region in regions:
        if merged and region[0] - merged[-1][1] - 1 <= config.candidate_merge_gap_windows:
            merged[-1] = (merged[-1][0], region[1])
        else:
            merged.append(region)
    rows: list[dict[str, object]] = []; prior_peak = -10**9; event_id = 0
    for begin, end in merged:
        region = online.iloc[begin:end + 1]
        peak = region.iloc[int(region.slow_final_score.to_numpy(float).argmax())]
        if int(peak.window_index) - prior_peak < config.candidate_cooldown_windows:
            continue
        prior_peak = int(peak.window_index); event_id += 1
        rows.append({
            "event_id": event_id, "dataset": str(peak.dataset), "setting": str(peak.setting),
            "start_cycle": float(region.center_cycle.iloc[0]), "peak_cycle": float(peak.center_cycle),
            "end_cycle": float(region.center_cycle.iloc[-1]), "peak_score": float(peak.slow_final_score),
            "fast_score": float(peak.fast_score), "slow_score": float(peak.slow_score),
            "residual_persistent": float(peak.residual_persistent),
            "adapter_frozen_fraction": float(region.adapter_frozen.mean()), "event_status": "unmatched_candidate_event",
        })
    columns = ["event_id", "dataset", "setting", "start_cycle", "peak_cycle", "end_cycle", "peak_score",
               "fast_score", "slow_score", "residual_persistent", "adapter_frozen_fraction", "event_status"]
    return pd.DataFrame(rows, columns=columns)


def run_v341_baseline(source_model: GradualStateTCN, target: pd.DataFrame, config: GradualStateMonitoringV35Config,
                      *, setting: str, source_normalizer: RobustLocationScale) -> OnlineRun:
    """Execute the unmodified V3.4.1 online implementation for the two declared controls."""
    old_setting = setting.removeprefix("V341_")
    baseline = run_v341_online(source_model, target, config, setting=old_setting, source_normalizer=source_normalizer)
    scores = baseline.scores.copy(); scores["setting"] = setting
    scores["fast_score"] = scores.get("short_score", np.nan)
    scores["slow_score"] = 0.4667 * scores.get("medium_score", np.nan) + 0.5333 * scores.get("long_score", np.nan)
    scores["slow_final_score"] = scores.get("final_transition_score", np.nan)
    scores["prediction_residual"] = np.nan; scores["residual_z"] = np.nan; scores["residual_persistent"] = np.nan
    scores["fast_threshold"] = np.nan; scores["slow_threshold"] = scores.get("candidate_threshold", np.nan); scores["residual_threshold"] = np.nan
    scores["candidate_high"] = (scores.get("final_transition_score", pd.Series(np.nan, index=scores.index)) >= scores.get("candidate_threshold", pd.Series(np.nan, index=scores.index))).fillna(False).astype(int)
    audit = baseline.adaptation.copy(); audit["setting"] = setting
    if len(audit):
        audit["phase"] = "ONLINE"; audit["references_reencoded_current_model"] = 1
        audit["score_computed_before_update"] = 1; audit["residual_origin_index"] = np.nan
        audit["maximum_online_input_index"] = audit.current_index
    return OnlineRun(scores, audit)


def extract_v341_baseline_events(scores: pd.DataFrame, config: GradualStateMonitoringV35Config) -> pd.DataFrame:
    original = scores.copy(); setting = str(original.setting.iloc[0]) if len(original) else "V341"
    original["setting"] = setting.removeprefix("V341_")
    events = extract_v341_events(original, config)
    if len(events):
        events["setting"] = setting
    return events
