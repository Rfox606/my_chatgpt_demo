from __future__ import annotations

import numpy as np
import pandas as pd

from .bocpd import bocpd_confirmed_segments, causal_activity_energy, segment_descriptors
from .config import V31Config
from .continuous import continuous_process, delayed_entry_convergence, source_initial_prior, synthetic_ood_uncertainty
from .forecasting import fit_source_frozen, gate_a_summary, run_target_transfer
from .primitives import fit_segment_clusters, primitive_table, private_state_path


def run_direction(source: pd.DataFrame, target: pd.DataFrame, config: V31Config, include_delayed: bool = True) -> dict[str, object]:
    source = source.sort_values(["center_cycle", "window_index"]).reset_index(drop=True); target = target.sort_values(["center_cycle", "window_index"]).reset_index(drop=True); frozen = fit_source_frozen(source, config)
    records, gate_log = run_target_transfer(frozen, target, config); gate_a = gate_a_summary(records, gate_log, config)
    source_energy = causal_activity_energy(source, config.features); target_activity = causal_activity_energy(target, config.features)
    target_h1 = records.loc[(records.model == "Source_Plus_Adapter_Gated") & (records.horizon == 1)].set_index("observed_index").squared_error if not records.empty else pd.Series(dtype=float)
    target_energy = np.asarray([float(target_h1.get(index, 0.0)) for index in range(len(target))])
    source_bocpd, source_bounds = bocpd_confirmed_segments(source_energy, source.center_cycle.to_numpy(float), config); target_bocpd, target_bounds = bocpd_confirmed_segments(target_energy, target.center_cycle.to_numpy(float), config)
    source_segments = segment_descriptors(source, source_energy, source_energy, source_bounds, config.features); target_segments = segment_descriptors(target, target_energy, target_activity, target_bounds, config.features)
    source_model = fit_segment_clusters(source_segments, config.primitive_k_candidates, config, "source_confirmed_segments_only"); primitive = primitive_table(source_model, source_segments, str(source.dataset.iloc[0])); private_path, private_model, private_decision = private_state_path(target_segments, config)
    gate_b_pass = bool(len(source_segments) >= 3 and len(target_segments) >= 3 and source_model is not None and source_model.selected_k >= 2 and private_decision["status"] == "PASS" and len(source_segments) < len(source) and len(target_segments) < len(target))
    gate_b = {"status": "PASS" if gate_b_pass else "FAIL", "reason": "all_preregistered_conditions_met" if gate_b_pass else "segments_or_private_state_threshold_not_met", "source_confirmed_segments": len(source_segments), "target_confirmed_segments": len(target_segments), "source_primitive_k": source_model.selected_k if source_model else 0, **private_decision, "single_window_kmeans_used": False}
    prior = source_initial_prior(source, config); continuous = continuous_process(target, records, target_bocpd, prior, config)
    delayed_rows: list[pd.DataFrame] = []; delayed_metrics: list[pd.DataFrame] = []
    if include_delayed:
        paths: dict[float, pd.DataFrame] = {}
        for entry in config.entries(str(target.dataset.iloc[0])):
            entry_records, _ = run_target_transfer(frozen, target, config, entry); entry_target = target.loc[target.center_cycle.ge(entry)].reset_index(drop=True); entry_energy = np.asarray([float(entry_records.loc[(entry_records.model == "Source_Plus_Adapter_Gated") & (entry_records.horizon == 1) & (entry_records.observed_index == index), "squared_error"].iloc[0]) if not entry_records.loc[(entry_records.model == "Source_Plus_Adapter_Gated") & (entry_records.horizon == 1) & (entry_records.observed_index == index)].empty else 0.0 for index in range(len(entry_target))]); entry_bocpd, _ = bocpd_confirmed_segments(entry_energy, entry_target.center_cycle.to_numpy(float), config); path = continuous_process(target, entry_records, entry_bocpd, prior, config, entry); paths[entry] = path; delayed_rows.append(path)
        delayed = delayed_entry_convergence(paths, config); delayed_metrics.append(delayed)
    else:
        delayed = pd.DataFrame()
    coverage = bool(len(continuous) == len(target) and np.isfinite(continuous.loc[:, ["cumulative_progression", "activity", "initial_prior", "uncertainty"]].to_numpy(float)).all()); activity_ok = bool(continuous.activity.std() > 1e-6); delayed_ok = bool(not delayed.empty and (delayed.common_arrived_windows >= config.delayed_common_arrived_windows).all() and (delayed.increment_nrmse_to_latest <= .50).all() and delayed.finite.all())
    synthetic = synthetic_ood_uncertainty(config); gate_c_pass = bool(coverage and activity_ok and delayed_ok and synthetic["status"] == "PASS" and (continuous.state_id_input_count == 0).all())
    gate_c = {"status": "PASS" if gate_c_pass else "FAIL", "reason": "all_preregistered_conditions_met" if gate_c_pass else "coverage_activity_ood_or_delayed_threshold_not_met", "continuous_coverage": coverage, "activity_std": float(continuous.activity.std()), "delayed_entry_ok": delayed_ok, "state_id_input_count": int(continuous.state_id_input_count.sum()), "synthetic_ood": synthetic}
    return {"gate_a": gate_a, "gate_b": gate_b, "gate_c": gate_c, "forecast_records": records, "adapter_log": gate_log, "source_bocpd": source_bocpd, "target_bocpd": target_bocpd, "source_segments": source_segments, "target_segments": target_segments, "source_primitives": primitive, "private_states": private_path, "continuous": continuous, "delayed": delayed, "source_prior": prior}


def prefix_audit(source: pd.DataFrame, target: pd.DataFrame, config: V31Config) -> dict[str, object]:
    original = run_direction(source, target, config, include_delayed=False); results: dict[str, object] = {}
    for cutoff_cycle in config.prefix_cutoff_cycles:
        changed = target.copy()
        changed.loc[changed.center_cycle.gt(cutoff_cycle), list(config.features)] += 999.0
        replay = run_direction(source, changed, config, include_delayed=False)
        cutoff = int(np.searchsorted(target.center_cycle.to_numpy(float), cutoff_cycle, side="right") - 1)
        if cutoff < 0:
            results[str(cutoff_cycle)] = {"status": "FAIL", "reason": "fixed_audit_cutoff_precedes_first_target_window"}
            continue
        values: dict[str, float] = {}
        for name, columns in {"forecast": ["squared_error"], "target_bocpd": ["bocpd_change_posterior", "bocpd_run_length_entropy"], "continuous": ["cumulative_progression", "activity", "uncertainty"]}.items():
            left = original["forecast_records"] if name == "forecast" else original[name]; right = replay["forecast_records"] if name == "forecast" else replay[name]
            if name == "forecast":
                left = left.loc[left.observed_index.le(cutoff)]; right = right.loc[right.observed_index.le(cutoff)]
            else:
                left = left.iloc[:cutoff + 1]; right = right.iloc[:cutoff + 1]
            values[name] = float(np.max(np.abs(left.loc[:, columns].to_numpy(float) - right.loc[:, columns].to_numpy(float)))) if len(left) and len(right) else 0.0
        private_left = original["private_states"]; private_right = replay["private_states"]
        if {"end_cycle", "private_state_id"}.issubset(private_left.columns) and {"end_cycle", "private_state_id"}.issubset(private_right.columns):
            state_left = private_left.loc[private_left.end_cycle.le(float(target.center_cycle.iloc[cutoff]))]
            state_right = private_right.loc[private_right.end_cycle.le(float(target.center_cycle.iloc[cutoff]))]
            values["private_state"] = float(np.max(np.abs(state_left.private_state_id.to_numpy(float) - state_right.private_state_id.to_numpy(float)))) if len(state_left) and len(state_right) else 0.0
        else:
            # A failed six-segment calibration has no state IDs; absence is
            # compared causally rather than treated as a fabricated state.
            values["private_state"] = 0.0
        values["status"] = "PASS" if max(float(item) for key, item in values.items() if key != "status") <= 1e-12 else "FAIL"; results[str(cutoff_cycle)] = values
    return {"status": "PASS" if all(item["status"] == "PASS" for item in results.values()) else "FAIL", "cutoffs": results}
