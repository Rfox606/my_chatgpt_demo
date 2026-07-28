from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, confusion_matrix, f1_score, normalized_mutual_info_score

from .config import GradualStateMonitoringV332Config


@dataclass(frozen=True)
class StageMapping:
    dataset: str
    stage: int
    effective_start: float
    effective_end: float
    actual_start: float
    actual_end: float
    note: str


def load_existing_stage_mapping(config: GradualStateMonitoringV332Config) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read the repository's already versioned Stage mapping; never infer a boundary."""
    path = Path(config.cycle_mapping_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    required = {"dataset", "stage", "effective_start", "effective_end", "actual_start", "actual_end", "note"}
    if not segments or any(not required.issubset(segment) for segment in segments):
        raise ValueError("Existing cycle mapping lacks an explicit Stage boundary definition")
    frame = pd.DataFrame(segments).loc[:, sorted(required)].copy()
    frame = frame.astype({"stage": int, "effective_start": float, "effective_end": float,
                          "actual_start": float, "actual_end": float})
    frame = frame.sort_values(["dataset", "stage"], kind="stable").reset_index(drop=True)
    provenance: dict[str, object] = {
        "status": "EXISTING_REPOSITORY_LABEL_DEFINITION",
        "source_path": str(path),
        "source_sha256": _sha256(path),
        "label_definition": "Stage and its boundaries are the existing mapping segments; no V3.3.2 boundary was inferred or tuned.",
        "online_use": "PROHIBITED",
        "permitted_use": ["posthoc_evaluation", "posthoc_figures"],
        "exp2_actual_cycle_prefix": payload.get("exp2_nan_prefix_actual_cycles", []),
    }
    return frame, provenance


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def effective_to_actual(cycles: Iterable[float], dataset: str, segments: pd.DataFrame) -> np.ndarray:
    """Use the original piecewise mapping without changing any segment endpoint."""
    values = np.asarray(list(cycles), dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    local = segments[segments.dataset == dataset].sort_values("stage", kind="stable")
    for _, row in local.iterrows():
        # Endpoints intentionally belong to the lower-numbered segment.
        mask = (values >= float(row.effective_start)) & (values <= float(row.effective_end)) & ~np.isfinite(output)
        denominator = float(row.effective_end - row.effective_start)
        output[mask] = float(row.actual_start) + (values[mask] - float(row.effective_start)) * float(row.actual_end - row.actual_start) / denominator
    return output


def attach_posthoc_stage(online_output: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Attach Stage only after label-free online outputs have been fully generated."""
    required = {"dataset", "cycle"}
    if not required.issubset(online_output.columns):
        raise ValueError(f"Post-hoc output lacks {sorted(required)}")
    result = online_output.copy()
    result["actual_cycle"] = np.nan
    result["stage"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    for dataset, index in result.groupby("dataset", sort=False).groups.items():
        positions = np.asarray(list(index), dtype=int)
        effective = result.loc[positions, "cycle"].to_numpy(float)
        actual = effective_to_actual(effective, str(dataset), segments)
        result.loc[positions, "actual_cycle"] = actual
        local = segments[segments.dataset == dataset]
        stage_values = np.full(len(positions), np.nan)
        for _, row in local.iterrows():
            # Adjacent source segments share an endpoint.  Match effective_to_actual:
            # retain the lower-numbered/source-earlier segment rather than overwrite it.
            mask = ((effective >= float(row.effective_start)) & (effective <= float(row.effective_end))
                    & ~np.isfinite(stage_values))
            stage_values[mask] = int(row.stage)
        result.loc[positions, "stage"] = pd.array(stage_values, dtype="Int64")
    return result


def stage_boundaries(segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, local in segments.groupby("dataset", sort=True):
        ordered = local.sort_values("stage", kind="stable").reset_index(drop=True)
        for index in range(1, len(ordered)):
            before = ordered.iloc[index - 1]
            after = ordered.iloc[index]
            rows.append({
                "dataset": dataset,
                "from_stage": int(before.stage), "to_stage": int(after.stage),
                "boundary_effective_cycle": float(after.effective_start),
                "boundary_actual_cycle": float(after.actual_start),
                "source": "existing_cycle_mapping_config",
            })
    return pd.DataFrame(rows)


def stable_region_mask(actual_cycles: np.ndarray, boundaries_actual: np.ndarray, buffer_cycles: float) -> np.ndarray:
    if len(boundaries_actual) == 0:
        return np.isfinite(actual_cycles)
    distance = np.min(np.abs(actual_cycles[:, None] - boundaries_actual[None, :]), axis=1)
    return np.isfinite(actual_cycles) & (distance > buffer_cycles)


def _state_changes(values: np.ndarray) -> np.ndarray:
    return np.r_[False, values[1:] != values[:-1]] if len(values) else np.empty(0, dtype=bool)


def stage_stability_metrics(posthoc: pd.DataFrame, boundaries: pd.DataFrame, config: GradualStateMonitoringV332Config,
                            *, model: str, setting: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, stage), group in posthoc.groupby(["dataset", "stage"], sort=True, dropna=True):
        ordered = group.sort_values("input_index", kind="stable")
        actual = ordered.actual_cycle.to_numpy(float)
        boundary_values = boundaries.loc[boundaries.dataset == dataset, "boundary_actual_cycle"].to_numpy(float)
        inside = stable_region_mask(actual, boundary_values, config.stage_buffer_actual_cycles)
        stable = ordered.online_state.to_numpy(str) == "STABLE"
        transition = ordered.online_state.to_numpy(str) == "TRANSITION"
        labels = ordered.local_state_id.to_numpy(int) if "local_state_id" in ordered else ordered.online_state.to_numpy(str)
        selected = labels[inside]
        state_values = ordered.online_state.to_numpy(str)
        switch = _state_changes(state_values)
        switch_count = int(np.sum(switch & inside))
        count = int(np.sum(inside))
        dominant = _dominant(selected)
        purity = float(np.mean(selected == dominant)) if count else np.nan
        rows.append({
            "dataset": dataset, "stage": int(stage), "model": model, "setting": setting,
            "stage_windows": int(len(ordered)), "stable_evaluation_windows": count,
            "stage_stable_rate": float(np.mean(stable[inside])) if count else np.nan,
            "stage_transition_rate": float(np.mean(transition[inside])) if count else np.nan,
            "dominant_detected_state": dominant, "stage_purity": purity,
            "detected_state_count": int(len(np.unique(selected))) if count else 0,
            "state_switch_count": switch_count,
            "false_switches_per_1000_windows": float(1000.0 * switch_count / count) if count else np.nan,
            "buffer_actual_cycles": config.stage_buffer_actual_cycles,
        })
    return pd.DataFrame(rows)


def _dominant(values: np.ndarray) -> str | int | float:
    if len(values) == 0:
        return np.nan
    unique, counts = np.unique(values, return_counts=True)
    return unique[int(np.argmax(counts))].item() if hasattr(unique[int(np.argmax(counts))], "item") else unique[int(np.argmax(counts))]


def stable_to_transition_alarms(posthoc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in posthoc.groupby("dataset", sort=True):
        ordered = group.sort_values("input_index", kind="stable").reset_index(drop=True)
        states = ordered.online_state.to_numpy(str)
        hit = (states[:-1] == "STABLE") & (states[1:] == "TRANSITION")
        for position in np.flatnonzero(hit) + 1:
            rows.append({"dataset": dataset, "input_index": int(ordered.input_index.iloc[position]),
                         "alarm_cycle": float(ordered.cycle.iloc[position]),
                         "alarm_actual_cycle": float(ordered.actual_cycle.iloc[position]),
                         "alarm_type": "STABLE_TO_TRANSITION"})
    return pd.DataFrame(rows)


def boundary_detection(posthoc: pd.DataFrame, boundaries: pd.DataFrame, tolerance: float,
                       *, model: str, setting: str) -> tuple[dict[str, object], pd.DataFrame]:
    """One first alarm per true boundary; later in-range alarms are duplicates."""
    alarms = stable_to_transition_alarms(posthoc)
    logs: list[dict[str, object]] = []
    matched_alarm_indices: set[int] = set()
    duplicate_indices: set[int] = set()
    for dataset, true_rows in boundaries.groupby("dataset", sort=True):
        available = alarms[alarms.dataset == dataset]
        for _, boundary in true_rows.sort_values("boundary_actual_cycle", kind="stable").iterrows():
            distance = np.abs(available.alarm_actual_cycle.to_numpy(float) - float(boundary.boundary_actual_cycle)) if len(available) else np.empty(0)
            candidates = available.index[np.flatnonzero(distance <= tolerance)] if len(available) else pd.Index([])
            candidates = [int(index) for index in candidates if int(index) not in matched_alarm_indices]
            if candidates:
                candidates.sort(key=lambda index: (float(alarms.loc[index, "alarm_actual_cycle"]), index))
                first = candidates[0]
                matched_alarm_indices.add(first)
                logs.append(_match_row(dataset, boundary, tolerance, alarms.loc[first], "MATCHED"))
                for duplicate in candidates[1:]:
                    matched_alarm_indices.add(duplicate)
                    duplicate_indices.add(duplicate)
                    logs.append(_match_row(dataset, boundary, tolerance, alarms.loc[duplicate], "DUPLICATE"))
            else:
                logs.append(_match_row(dataset, boundary, tolerance, None, "MISSED"))
    false_indices = [index for index in alarms.index if int(index) not in matched_alarm_indices]
    for index in false_indices:
        alarm = alarms.loc[index]
        logs.append({"dataset": alarm.dataset, "model": model, "setting": setting, "tolerance_actual_cycles": tolerance,
                     "from_stage": np.nan, "to_stage": np.nan, "boundary_effective_cycle": np.nan,
                     "boundary_actual_cycle": np.nan, "alarm_cycle": float(alarm.alarm_cycle),
                     "alarm_actual_cycle": float(alarm.alarm_actual_cycle), "detection_delay_cycles": np.nan,
                     "match_status": "FALSE_ALARM"})
    log = pd.DataFrame(logs)
    if len(log):
        log.loc[:, "model"] = model
        log.loc[:, "setting"] = setting
    true_count = int(len(boundaries))
    matched = int((log.match_status == "MATCHED").sum()) if len(log) else 0
    false = int((log.match_status == "FALSE_ALARM").sum()) if len(log) else 0
    duplicate = int((log.match_status == "DUPLICATE").sum()) if len(log) else 0
    precision = float(matched / (matched + false)) if matched + false else 0.0
    recall = float(matched / true_count) if true_count else np.nan
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    delays = log.loc[log.match_status == "MATCHED", "detection_delay_cycles"].to_numpy(float) if len(log) else np.empty(0)
    metric = {"model": model, "setting": setting, "tolerance_actual_cycles": tolerance,
              "boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1,
              "mean_detection_delay": float(np.mean(delays)) if len(delays) else np.nan,
              "median_detection_delay": float(np.median(delays)) if len(delays) else np.nan,
              "missed_boundary_count": int(true_count - matched), "false_alarm_count": false,
              "duplicate_alarm_count": duplicate, "matched_boundary_count": matched,
              "true_boundary_count": true_count}
    return metric, log


def _match_row(dataset: str, boundary: pd.Series, tolerance: float, alarm: pd.Series | None, status: str) -> dict[str, object]:
    return {"dataset": dataset, "tolerance_actual_cycles": tolerance,
            "from_stage": int(boundary.from_stage), "to_stage": int(boundary.to_stage),
            "boundary_effective_cycle": float(boundary.boundary_effective_cycle),
            "boundary_actual_cycle": float(boundary.boundary_actual_cycle),
            "alarm_cycle": float(alarm.alarm_cycle) if alarm is not None else np.nan,
            "alarm_actual_cycle": float(alarm.alarm_actual_cycle) if alarm is not None else np.nan,
            "detection_delay_cycles": float(alarm.alarm_actual_cycle - boundary.boundary_actual_cycle) if alarm is not None else np.nan,
            "match_status": status}


def ordered_state_mapping(posthoc: pd.DataFrame, *, model: str, setting: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Best monotone (first-appearance) state-to-Stage mapping, never Hungarian matching."""
    rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    summaries: list[dict[str, float]] = []
    for dataset, group in posthoc.dropna(subset=["stage"]).groupby("dataset", sort=True):
        ordered = group.sort_values("input_index", kind="stable")
        state_order = list(dict.fromkeys(ordered.local_state_id.astype(int).tolist()))
        observed_stage = ordered.stage.astype(int).to_numpy()
        observed_state = ordered.local_state_id.astype(int).to_numpy()
        stage_values = sorted(np.unique(observed_stage).tolist())
        mapping = _best_monotone_mapping(state_order, stage_values, observed_state, observed_stage)
        mapped = np.asarray([mapping[int(value)] for value in observed_state], dtype=int)
        for sequence, state_id in enumerate(state_order, start=1):
            rows.append({"dataset": dataset, "model": model, "setting": setting, "first_appearance_order": sequence,
                         "detected_state": state_id, "mapped_stage": mapping[state_id], "mapping_constraint": "nondecreasing_first_appearance"})
        matrix = confusion_matrix(observed_stage, mapped, labels=stage_values)
        for true_index, true_stage in enumerate(stage_values):
            for pred_index, predicted_stage in enumerate(stage_values):
                confusion_rows.append({"dataset": dataset, "model": model, "setting": setting,
                                       "true_stage": true_stage, "mapped_detected_stage": predicted_stage,
                                       "count": int(matrix[true_index, pred_index])})
        summaries.append({"dataset": dataset, "model": model, "setting": setting,
                          "ordered_mapping_accuracy": float(np.mean(mapped == observed_stage)),
                          "macro_f1": float(f1_score(observed_stage, mapped, labels=stage_values, average="macro", zero_division=0)),
                          "ARI": float(adjusted_rand_score(observed_stage, observed_state)),
                          "NMI": float(normalized_mutual_info_score(observed_stage, observed_state))})
    return pd.DataFrame(rows), pd.DataFrame(confusion_rows), pd.DataFrame(summaries).to_dict("records")


def _best_monotone_mapping(state_order: list[int], stages: list[int], states: np.ndarray, truth: np.ndarray) -> dict[int, int]:
    if not state_order or not stages:
        return {}
    counts = {(state, stage): int(np.sum((states == state) & (truth == stage))) for state in state_order for stage in stages}
    best_score = -1
    best: tuple[int, ...] | None = None
    for candidate in combinations_with_replacement(stages, len(state_order)):
        score = sum(counts[(state, stage)] for state, stage in zip(state_order, candidate))
        if score > best_score:
            best_score, best = score, candidate
    assert best is not None
    return dict(zip(state_order, best))
