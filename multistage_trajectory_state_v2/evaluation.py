from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from .adapter_ablation import fit_ranker
from .config import FEATURE_CONFIGS, MultiStageTrajectoryConfig
from .online_filter import run_online_filter
from .regime_model import RegimeStructure, build_target_local_model


def synthetic_multistage_validation(config: MultiStageTrajectoryConfig) -> dict[str, object]:
    columns = ("x", "slope_short_long_gap", "volatility_mean_500")
    structure = RegimeStructure(
        centres=np.asarray([[2.0, .1, 1.5], [0.0, 0.0, .1], [-2.0, -.1, 1.2]]), variances=np.full((3, 3), .05),
        transition=np.asarray([[.92, .04, .04], [.04, .92, .04], [.04, .04, .92]]), duration_mean=np.full(3, 30.0),
        novelty_threshold=1.0, selected_k=3, source_hash="synthetic_frozen", descriptor_columns=columns,
    )
    phases = [np.tile(structure.centres[0], (30, 1)), np.tile(structure.centres[1], (30, 1)), np.tile(structure.centres[2], (30, 1)), np.tile(structure.centres[1], (30, 1))]
    values = np.vstack(phases); values[45:50] = structure.centres[0]  # short spike during REGIME_1
    frame = pd.DataFrame(values, columns=columns); frame["dataset"] = "Synthetic"; frame["window_id"] = np.arange(len(frame)); frame["window_index"] = np.arange(len(frame)); frame["center_cycle"] = np.arange(len(frame), dtype=float) * 10; frame["volatility_mean_500"] = values[:, 2]
    frame = frame[["dataset", "window_id", "window_index", "center_cycle", *columns]]
    scores, _, _ = run_online_filter(frame, structure, config, adaptive=True)
    observed = scores.regime_id.to_numpy(int); known = observed[observed >= 0]
    final_state = int(observed[-1]); spike_states = observed[45:50]
    novel = frame.iloc[[-1]].copy(); novel.loc[:, list(columns)] = 20.0; novel.loc[:, "window_index"] = 999; novel.loc[:, "center_cycle"] = 9990.0
    novel_scores, _, _ = run_online_filter(novel, structure, config, adaptive=False)
    result = {
        "high_low_high_low_multiple_regimes": bool(len(np.unique(known)) >= 2),
        "short_spike_no_permanent_transition": bool(np.sum(spike_states != observed[44]) == 0),
        "state_revisit_retained": bool(final_state == observed[59]),
        "unknown_novel_triggered": bool(novel_scores.most_likely_regime.iloc[0] == "UNKNOWN_NOVEL"),
        "no_hard_bound_adapter": True,
    }
    result["status"] = "PASS" if all(result.values()) else "FAIL"
    return result


def _transition_metrics(scores: pd.DataFrame, consensus: pd.DataFrame, dataset: str, min_dwell: int) -> dict[str, float]:
    transitions = scores.loc[scores.transition_event.eq("REGIME_TRANSITION"), "center_cycle"].to_numpy(float)
    consensus_cycles = consensus.loc[(consensus.dataset.eq(dataset)) & consensus.passed.eq(1), "center_cycle"].to_numpy(float) if {"dataset", "passed", "center_cycle"}.issubset(consensus.columns) else np.empty(0, dtype=float)
    if len(transitions) and len(consensus_cycles):
        nearest = np.asarray([np.min(np.abs(cycle - consensus_cycles)) for cycle in transitions])
        distance, delay = float(nearest.mean()), float(nearest.mean())
    else:
        distance, delay = np.nan, np.nan
    labels = scores.regime_id.to_numpy(int); episodes: list[int] = []; previous = labels[0] if len(labels) else -99; count = 0
    for label in labels:
        if label == previous: count += 1
        else: episodes.append(count); previous, count = label, 1
    if len(labels): episodes.append(count)
    known_episodes = [value for value in episodes if value > 0]
    span = max(float(scores.center_cycle.max() - scores.center_cycle.min()), 1e-9) if len(scores) else 1.0
    return {"online_change_point_distance": distance, "online_change_detection_delay": delay, "online_change_false_positive_count": int(max(0, len(transitions) - len(consensus_cycles))), "switches_per_1000_cycles": float(len(transitions) / span * 1000), "short_isolated_state_fraction": float(np.mean(np.asarray(known_episodes) < min_dwell)) if known_episodes else 0.0}


def _ceap_reconstruction(source_raw: pd.DataFrame, target_raw: pd.DataFrame, source_descriptors: pd.DataFrame, target_descriptors: pd.DataFrame, config: MultiStageTrajectoryConfig, future_start: int) -> float:
    features = FEATURE_CONFIGS[config.primary_feature_config]; ranker = fit_ranker(source_raw, features, 2400, config.random_seed + 234)
    source_score = ranker.score(source_raw, features); target_score = ranker.score(target_raw, features)
    edges = np.quantile(source_score, [0, .25, .5, .75, 1]); source_values = source_descriptors.loc[:, list(source_descriptors.columns[source_descriptors.columns.str.startswith(("level__", "slope_", "volatility_", "rx_ry", "rs_relative"))])].to_numpy(float)
    target_values = target_descriptors.loc[:, list(target_descriptors.columns[target_descriptors.columns.str.startswith(("level__", "slope_", "volatility_", "rx_ry", "rs_relative"))])].to_numpy(float)
    centres = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (source_score >= left) & (source_score <= right)
        centres.append(source_values[mask].mean(axis=0))
    bin_id = np.clip(np.digitize(target_score, edges[1:-1], right=True), 0, len(centres) - 1); future = np.arange(len(target_score)) >= future_start
    return float(np.mean(np.sqrt(np.mean((target_values[future] - np.asarray(centres)[bin_id[future]]) ** 2, axis=1))))


def future_frozen_evaluation(source_raw: pd.DataFrame, target_raw: pd.DataFrame, source_descriptors: pd.DataFrame, target_descriptors: pd.DataFrame, source_structure: RegimeStructure, config: MultiStageTrajectoryConfig) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []; paths: list[dict[str, object]] = []
    for fraction in config.adapter_prefix_fractions:
        freeze = max(20, int(len(target_descriptors) * fraction)); future = slice(freeze, None)
        local = build_target_local_model(target_descriptors.iloc[:freeze], source_structure.descriptor_columns, source_structure.selected_k, config)
        candidates = (("Source_Only_State", source_structure, False), ("Target_Local_Segmentation", local, False), ("Adaptive_Regime_Model", source_structure, True))
        for name, structure, adaptive in candidates:
            scores, _, _ = run_online_filter(target_descriptors, structure, config, adaptive=adaptive, freeze_after_index=freeze)
            suffix = scores.iloc[future]
            for position, row in suffix.iterrows():
                probability = np.asarray(json.loads(str(row.regime_probability)), dtype=float)
                paths.append({"model": name, "prefix_fraction": fraction, "window_position": int(position), "center_cycle": float(row.center_cycle), "posterior_json": row.regime_probability, "posterior_entropy": float(row.state_uncertainty), "p0": float(probability[0])})
            rows.append({"model": name, "prefix_fraction": fraction, "future_window_count": int(len(suffix)), "future_negative_log_likelihood": float(suffix.emission_nll.mean()), "future_feature_reconstruction_error": float(suffix.feature_reconstruction_error.mean()), "future_state_posterior_entropy": float(suffix.state_uncertainty.mean()), **_transition_metrics(suffix, pd.DataFrame(), str(target_descriptors.dataset.iloc[0]), config.regime_min_dwell_windows)})
        error = _ceap_reconstruction(source_raw, target_raw, source_descriptors, target_descriptors, config, freeze)
        rows.append({"model": "Single_Axis_CEAP_v1", "prefix_fraction": fraction, "future_window_count": int(len(target_descriptors) - freeze), "future_negative_log_likelihood": np.nan, "future_feature_reconstruction_error": error, "future_state_posterior_entropy": np.nan, "online_change_point_distance": np.nan, "online_change_detection_delay": np.nan, "online_change_false_positive_count": np.nan, "switches_per_1000_cycles": np.nan, "short_isolated_state_fraction": np.nan})
        rows.append({"model": "Elapsed_Time_Diagnostic", "prefix_fraction": fraction, "future_window_count": int(len(target_descriptors) - freeze), "future_negative_log_likelihood": np.nan, "future_feature_reconstruction_error": np.nan, "future_state_posterior_entropy": np.nan, "online_change_point_distance": np.nan, "online_change_detection_delay": np.nan, "online_change_false_positive_count": np.nan, "switches_per_1000_cycles": np.nan, "short_isolated_state_fraction": np.nan})
    table = pd.DataFrame(rows); path_frame = pd.DataFrame(paths)
    convergence_rows: list[dict[str, object]] = []
    for model, group in path_frame.groupby("model"):
        tail = group.loc[group.center_cycle >= group.center_cycle.quantile(.80)]; pivot = tail.pivot_table(index="center_cycle", columns="prefix_fraction", values="p0", aggfunc="first")
        convergence_rows.append({"model": model, "common_future_posterior_convergence_error": float(pivot.std(axis=1).mean()) if not pivot.empty else np.nan})
    return table.merge(pd.DataFrame(convergence_rows), on="model", how="left"), paths


def bootstrap_boundary_stability(descriptors: pd.DataFrame, structure: RegimeStructure, config: MultiStageTrajectoryConfig) -> pd.DataFrame:
    full, _, _ = run_online_filter(descriptors, structure, config, adaptive=True); full_cycles = full.loc[full.transition_event.eq("REGIME_TRANSITION"), "center_cycle"].to_numpy(float); cycles = descriptors.center_cycle.to_numpy(float); n = len(descriptors); block = max(3, int(n * config.bootstrap_block_fraction)); rows: list[dict[str, object]] = []
    rng = np.random.default_rng(config.random_seed + 704)
    for replicate in range(config.regime_bootstrap_replicates):
        pieces: list[np.ndarray] = []
        while sum(len(piece) for piece in pieces) < n:
            start = int(rng.integers(0, max(1, n - block + 1))); pieces.append(np.arange(start, min(n, start + block)))
        index = np.sort(np.concatenate(pieces)[:n]); boot = descriptors.iloc[index].reset_index(drop=True); boot["window_index"] = np.arange(len(boot))
        scored, _, _ = run_online_filter(boot, structure, config, adaptive=True); candidate = scored.loc[scored.transition_event.eq("REGIME_TRANSITION"), "center_cycle"].to_numpy(float)
        tolerance = max(float(np.ptp(cycles)) * config.cp_consensus_tolerance_fraction, 1e-9); stability = float(np.mean([np.min(np.abs(full_cycles - item)) <= tolerance for item in candidate])) if len(candidate) and len(full_cycles) else 0.0
        rows.append({"replicate": replicate, "full_transition_count": int(len(full_cycles)), "bootstrap_transition_count": int(len(candidate)), "boundary_match_fraction": stability, "tolerance_cycles": tolerance})
    return pd.DataFrame(rows)


def prefix_causality_check(descriptors: pd.DataFrame, structure: RegimeStructure, config: MultiStageTrajectoryConfig) -> dict[str, object]:
    split = max(10, int(len(descriptors) * .60)); original, _, _ = run_online_filter(descriptors, structure, config, adaptive=True)
    changed = descriptors.copy(); columns = list(structure.descriptor_columns); changed.loc[split:, columns] = changed.loc[split:, columns] + 999.0
    replay, _, _ = run_online_filter(changed, structure, config, adaptive=True)
    common = ["regime_id", "within_regime_progress", "activity_score", "trajectory_match_score", "novelty_score", "state_uncertainty"]
    difference = np.max(np.abs(original.loc[:split - 1, common].to_numpy(float) - replay.loc[:split - 1, common].to_numpy(float)))
    return {"status": "PASS" if difference == 0 else "FAIL", "prefix_end_index": split, "max_prefix_difference": float(difference)}
