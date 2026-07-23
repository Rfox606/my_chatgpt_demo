from __future__ import annotations

import numpy as np
import pandas as pd

from .bocpd import bocpd_confirmed_segments, causal_activity_energy, segment_descriptors
from .config import V32Config
from .continuous import continuous_process, delayed_entry_convergence, initial_progression_prior, synthetic_ood_uncertainty
from .forecasting import fit_source_frozen, gate_a_summary, prefix_freeze_metrics, run_target_transfer
from .primitives import fit_segment_clusters, online_target_states, primitive_table


def _target_errors(records: pd.DataFrame, count: int) -> np.ndarray:
    error = np.zeros(count, dtype=float)
    if records.empty:
        return error
    chosen = records.loc[(records.model == "Negative_transfer_Gated_Mixture") & (records.horizon == 1)]
    for index, group in chosen.groupby("observed_index"):
        if 0 <= int(index) < count:
            error[int(index)] = float(group.absolute_error.mean())
    return error


def _segment_target(
    target: pd.DataFrame, records: pd.DataFrame, config: V32Config
) -> tuple[pd.DataFrame, list[tuple[int, int]], pd.DataFrame, np.ndarray]:
    activity = causal_activity_energy(target, config.features)
    errors = _target_errors(records, len(target))
    # Forecast residuals are causal at their observed time.  Where no 1-step
    # forecast is yet due, activity is the only available causal evidence.
    energy = errors + 0.10 * activity
    bocpd, bounds = bocpd_confirmed_segments(energy, target.center_cycle.to_numpy(float), config)
    descriptors = segment_descriptors(target, errors, activity, bounds, config.features)
    return bocpd, bounds, descriptors, errors


def _delayed_path(
    source, source_segments: pd.DataFrame, source_model, target: pd.DataFrame, entry: float, config: V32Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active = target.loc[target.center_cycle.ge(entry)].reset_index(drop=True)
    records, _ = run_target_transfer(source, target, config, entry)
    bocpd, _, descriptors, _ = _segment_target(active, records, config)
    states, _, _ = online_target_states(descriptors, source_model, config)
    prior = initial_progression_prior(source_segments, descriptors, source_model)
    process = continuous_process(active, records, bocpd, prior, states, config, entry)
    return process, states, bocpd, descriptors


def run_direction(source: pd.DataFrame, target: pd.DataFrame, config: V32Config, include_delayed: bool = True) -> dict[str, object]:
    source = source.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    target = target.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    frozen = fit_source_frozen(source, config)
    records, weight_log = run_target_transfer(frozen, target, config)
    prefix_metrics = prefix_freeze_metrics(frozen, target, config)
    gate_a = gate_a_summary(prefix_metrics, weight_log, config)

    source_activity = causal_activity_energy(source, config.features)
    source_bocpd, source_bounds = bocpd_confirmed_segments(source_activity, source.center_cycle.to_numpy(float), config)
    source_segments = segment_descriptors(source, source_activity, source_activity, source_bounds, config.features)
    target_bocpd, target_bounds, target_segments, _ = _segment_target(target, records, config)
    source_model = fit_segment_clusters(source_segments, config.primitive_k_candidates, config, "source_confirmed_segments_only")
    source_primitives = primitive_table(source_model, source_segments, str(source.dataset.iloc[0]))
    state_path, private_log, private_decision = online_target_states(target_segments, source_model, config)
    multiple_confirmed = len(source_segments) >= 2 and len(target_segments) >= 2
    gate_b_pass = bool(multiple_confirmed and source_model is not None and private_decision["status"] == "PASS")
    gate_b = {
        "status": "PASS" if gate_b_pass else "FAIL",
        "reason": "confirmed_multiple_segments_and_online_target_states" if gate_b_pass else "detector_or_data_did_not_confirm_multiple_segments",
        "source_confirmed_segments": int(len(source_segments)),
        "target_confirmed_segments": int(len(target_segments)),
        "source_primitive_count": int(source_model.selected_k) if source_model else 0,
        **private_decision,
        "segment_descriptor_rows": int(len(source_segments) + len(target_segments)),
        "single_window_primitives_used": False,
    }

    prior = initial_progression_prior(source_segments, target_segments, source_model)
    continuous = continuous_process(target, records, target_bocpd, prior, state_path, config)
    synthetic = synthetic_ood_uncertainty(config)
    if continuous.empty:
        coverage = False
        platform_low = False
        spike_guard = False
    else:
        coverage = bool(np.isfinite(continuous.select_dtypes(include=[np.number]).to_numpy(float)).all())
        low = continuous.loc[continuous.activity_score.le(continuous.activity_score.quantile(0.25)), "progression_increment"]
        high = continuous.loc[continuous.activity_score.ge(continuous.activity_score.quantile(0.75)), "progression_increment"]
        platform_low = bool(float(low.mean()) <= float(high.mean()) + 1e-12) if len(low) and len(high) else False
        # No isolated activity point can emit trend/anomaly evidence, so a
        # single row's contribution is bounded by a confirmed transition only.
        spike_guard = bool(continuous.progression_increment.max() <= 1.0 + 1e-12)
    delayed = pd.DataFrame()
    delayed_paths: dict[float, tuple[pd.DataFrame, pd.DataFrame]] = {}
    if include_delayed:
        for entry in config.entries(str(target.dataset.iloc[0])):
            process, states, _, _ = _delayed_path(frozen, source_segments, source_model, target, entry, config)
            delayed_paths[entry] = (process, states)
        delayed = delayed_entry_convergence(delayed_paths, config)
    delayed_common_ok = bool(
        not delayed.empty
        and (delayed.common_arrived_windows >= config.delayed_common_arrived_windows).all()
        and np.isfinite(delayed.select_dtypes(include=[np.number]).to_numpy(float)).all()
    )
    delayed_increment_converges = False
    delayed_score_converges = False
    delayed_uncertainty_declines = False
    if delayed_common_ok:
        comparisons: list[tuple[bool, bool, bool]] = []
        for _, group in delayed.groupby("entry_cycle", sort=True):
            ordered = group.sort_values("common_window_rank")
            block = max(1, len(ordered) // 4)
            early = ordered.iloc[:block]
            late = ordered.iloc[-block:]
            comparisons.append(
                (
                    float(late.progression_increment_abs_difference.mean()) <= float(early.progression_increment_abs_difference.mean()) + 1e-12,
                    float(late.progression_score_abs_difference.mean()) <= float(early.progression_score_abs_difference.mean()) + 1e-12,
                    float(late.uncertainty.mean()) <= float(early.uncertainty.mean()) + 1e-12,
                )
            )
        delayed_increment_converges = all(item[0] for item in comparisons)
        delayed_score_converges = all(item[1] for item in comparisons)
        delayed_uncertainty_declines = all(item[2] for item in comparisons)
    delayed_ok = bool(delayed_common_ok and delayed_increment_converges and delayed_score_converges and delayed_uncertainty_declines)
    gate_c_pass = bool(
        coverage
        and platform_low
        and spike_guard
        and delayed_ok
        and synthetic["status"] == "PASS"
        and continuous.state_id_input_count.eq(0).all()
    )
    gate_c = {
        "status": "PASS" if gate_c_pass else "FAIL",
        "reason": "continuous_evidence_and_delayed_entry_checks_pass" if gate_c_pass else "coverage_platform_spike_or_delayed_entry_check_failed",
        "continuous_coverage": coverage,
        "platform_increment_not_higher_than_change_increment": platform_low,
        "short_spike_increment_guard": spike_guard,
        "delayed_entry_ok": delayed_ok,
        "delayed_common_window_ok": delayed_common_ok,
        "delayed_increment_converges": delayed_increment_converges,
        "delayed_score_converges": delayed_score_converges,
        "delayed_uncertainty_declines": delayed_uncertainty_declines,
        "state_id_input_count": int(continuous.state_id_input_count.sum()) if not continuous.empty else -1,
        "synthetic_ood": synthetic,
    }
    return {
        "gate_a": gate_a,
        "gate_b": gate_b,
        "gate_c": gate_c,
        "source_model": frozen,
        "forecast_records": records,
        "weight_log": weight_log,
        "prefix_metrics": prefix_metrics,
        "source_bocpd": source_bocpd,
        "target_bocpd": target_bocpd,
        "source_segments": source_segments,
        "target_segments": target_segments,
        "source_primitives": source_primitives,
        "target_state_path": state_path,
        "private_state_log": private_log,
        "continuous": continuous,
        "delayed": delayed,
        "initial_prior": prior,
    }
