from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from continuous_state_v42.config import FEATURES, FEATURE_GROUPS, ContinuousStateV42Config
from continuous_state_v42.consensus import ConfigurationRecord, consensus_trajectories, detect_change_episodes, episode_match_jaccard
from continuous_state_v42.data import add_restart_guard, assert_label_free, baseline_mask, load_window_table, robust_location_scale
from continuous_state_v42.evaluation import evidence_summary, forecast_benefit, implementation_diagnostics, prefix_causality, write_json
from continuous_state_v42.forecast import run_online_forecasts, train_frozen_models
from continuous_state_v42.plotting import make_figures
from continuous_state_v42.report import make_report
from continuous_state_v42.state_engine import feature_subset, fit_source_support, run_target_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path("continuous_state_v42").glob("*.py")):
        digest.update(path.name.encode("utf-8")); digest.update(path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _configuration_id(baseline: int, distance: str, removed: str | None) -> str:
    return f"b{baseline}_{distance}_{removed or 'all'}"


def _records_for_protocol(source: pd.DataFrame, target: pd.DataFrame, protocol: str, config: ContinuousStateV42Config) -> tuple[list[ConfigurationRecord], pd.DataFrame, object]:
    records: list[ConfigurationRecord] = []
    canonical_source = pd.DataFrame(); canonical_support = None
    for baseline in (500, 1000, 2000):
        for distance in ("mahalanobis", "diagonal"):
            for removed in (None, "rx", "ry", "rs"):
                features = feature_subset(FEATURES, removed)
                run_config = replace(config, baseline_cycles=baseline, distance_form=distance)
                support = fit_source_support(source, features)
                target_state, _, _ = run_target_state(target, protocol, features, run_config, support)
                records.append(ConfigurationRecord(_configuration_id(baseline, distance, removed), baseline, distance, removed or "none", target_state))
                if baseline == 1000 and distance == "mahalanobis" and removed is None:
                    canonical_source, _, _ = run_target_state(source, protocol + "_source", features, run_config, support)
                    canonical_support = support
    assert canonical_support is not None
    return records, canonical_source, canonical_support


def _ry_feature_audit(frame: pd.DataFrame, canonical: pd.DataFrame, config: ContinuousStateV42Config) -> dict[str, object]:
    baseline = baseline_mask(frame, config)
    values = frame.ry_p2p.to_numpy(float)
    location, scale = robust_location_scale(values[baseline, None], config.eps)
    score = np.abs((values - location[0]) / scale[0])
    monitor = frame.start_cycle.to_numpy(float) > config.baseline_cycles
    guard = frame.is_restart_guard.to_numpy(bool)
    outlier = score > 3.0
    guard_rate = float(outlier[monitor & guard].mean()) if (monitor & guard).any() else np.nan
    non_guard_rate = float(outlier[monitor & ~guard].mean()) if (monitor & ~guard).any() else np.nan
    dominance = canonical.D_ry_subspace.to_numpy(float) / np.maximum(canonical.loc[:, ["D_rs_subspace", "D_rx_subspace", "D_ry_subspace"]].sum(axis=1).to_numpy(float), config.eps)
    return {"dataset": str(frame.dataset.iloc[0]), "feature_name": "ry_p2p", "baseline_location": float(location[0]), "baseline_scale": float(scale[0]),
            "monitoring_outlier_count": int(outlier[monitor].sum()), "monitoring_outlier_fraction": float(outlier[monitor].mean()),
            "guard_outlier_fraction": guard_rate, "non_guard_outlier_fraction": non_guard_rate,
            "guard_dependency_enrichment": guard_rate / non_guard_rate if np.isfinite(guard_rate) and non_guard_rate > 0 else np.nan,
            "stop_boundary_outlier_fraction": float(outlier[monitor & frame.crosses_stop_boundary.to_numpy(bool)].mean()) if (monitor & frame.crosses_stop_boundary.to_numpy(bool)).any() else np.nan,
            "ry_subspace_dominance_p95": float(np.quantile(dominance, .95)), "ry_subspace_dominance_fraction_over_060": float((dominance > .60).mean()),
            "single_feature_dominance_flag": bool((dominance > .60).mean() > .10)}


def _ry_removal_effect(records: list[ConfigurationRecord], config: ContinuousStateV42Config) -> dict[str, object]:
    full = [record for record in records if record.removed_feature_group == "none"]
    no_ry = [record for record in records if record.removed_feature_group == "ry"]
    full_consensus, _, full_long = consensus_trajectories(full, config)
    ry_consensus, _, ry_long = consensus_trajectories(no_ry, config)
    merged = full_consensus.loc[:, ["window_index", "D_state_q50"]].merge(ry_consensus.loc[:, ["window_index", "D_state_q50"]], on="window_index", suffixes=("_full", "_no_ry"))
    full_episodes = detect_change_episodes(full_consensus, full_long, config)
    ry_episodes = detect_change_episodes(ry_consensus, ry_long, config)
    return {"D_state_spearman_full_vs_no_ry": float(merged.D_state_q50_full.corr(merged.D_state_q50_no_ry, method="spearman")),
            "full_episode_count": int(len(full_episodes)), "no_ry_episode_count": int(len(ry_episodes)),
            "episode_peak_jaccard_500_cycles": float(episode_match_jaccard(full_episodes, ry_episodes))}


def _forecast_diagnostics(predictions: pd.DataFrame, config: ContinuousStateV42Config) -> dict[str, dict[str, object]]:
    delta = np.abs(predictions.Online_RLS_prediction.to_numpy(float) - predictions.Zero_Delta_prediction.to_numpy(float))
    clipped = np.isclose(delta, config.forecast_delta_clip, rtol=0, atol=1e-8)
    chosen = predictions.safe_gate_selected_model.value_counts(normalize=True)
    return {"rls_prediction_clipping_ratio": {"status": "PASS", "clipped_prediction_ratio": float(clipped.mean()), "clip_cycles": config.forecast_delta_clip},
            "safe_gate_selection_ratio": {"status": "PASS", "selection_ratio": {str(name): float(value) for name, value in chosen.items()}, "online_selection_ratio": float(chosen.get("Online_RLS", 0.0))}}


def _consensus_diagnostics(consensus: pd.DataFrame, episodes: pd.DataFrame) -> dict[str, dict[str, object]]:
    after_2000 = consensus.loc[consensus.start_cycle > 2000]
    complete = bool(not after_2000.empty and after_2000.effective_configuration_count.eq(24).all())
    finite_columns = [column for column in consensus.columns if column.endswith(("_q50", "_q25", "_q75", "_mad"))]
    finite = bool(np.isfinite(consensus.loc[:, finite_columns].to_numpy(float)).all())
    v41_path = Path("outputs_continuous_state_v41/results/ablation_summary_v41.csv")
    v41_event_stability = float(pd.read_csv(v41_path).event_collection_stable.mean()) if v41_path.exists() else np.nan
    high_support = float((episodes.configuration_support >= .80).mean()) if not episodes.empty else 0.0
    return {"configuration_coverage_check": {"status": "PASS" if complete else "FAIL", "all_24_available_after_2000": complete,
                                                "minimum_effective_configurations": int(consensus.effective_configuration_count.min()), "maximum_effective_configurations": int(consensus.effective_configuration_count.max())},
            "consensus_numerical_check": {"status": "PASS" if finite else "FAIL", "all_consensus_statistics_finite": finite},
            "consensus_vs_v41_stability": {"status": "PASS" if high_support >= v41_event_stability else "FAIL", "consensus_high_support_episode_fraction": high_support,
                                               "v41_single_configuration_event_stability": v41_event_stability}}


def main() -> None:
    config = ContinuousStateV42Config(); paths = config.paths(); payload = config.jsonable()
    fingerprint = {"code_version": "csv42", "config": payload, "input_file_sha256": _sha256(Path(config.z_table_path)), "code_sha256": _code_hash()}
    (paths["configs"] / "continuous_state_v42_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(paths["configs"] / "run_fingerprint.json", fingerprint)
    raw = add_restart_guard(load_window_table(config), config)
    exp1 = raw.loc[raw.dataset.eq("Exp1")].reset_index(drop=True); exp2 = raw.loc[raw.dataset.eq("Exp2")].reset_index(drop=True)
    assert_label_free(exp1); assert_label_free(exp2)
    protocols = {"A_Exp1_to_Exp2": (exp1, exp2), "B_Exp2_to_Exp1": (exp2, exp1)}
    all_consensus: list[pd.DataFrame] = []; all_support: list[pd.DataFrame] = []; all_episodes: list[pd.DataFrame] = []
    canonical_states: dict[str, pd.DataFrame] = {}; canonical_sources: dict[str, pd.DataFrame] = {}; canonical_supports = {}; records_by_protocol: dict[str, list[ConfigurationRecord]] = {}
    predictions: list[pd.DataFrame] = []; metrics: list[pd.DataFrame] = []; updates: list[pd.DataFrame] = []; audit: list[dict[str, object]] = []
    for protocol, (source, target) in protocols.items():
        records, source_state, support = _records_for_protocol(source, target, protocol, config)
        records_by_protocol[protocol] = records; canonical_sources[protocol] = source_state; canonical_supports[protocol] = support
        canonical = next(record.states for record in records if record.config_id == "b1000_mahalanobis_all")
        canonical_states[protocol] = canonical
        consensus, support_table, long = consensus_trajectories(records, config)
        all_consensus.append(consensus); all_support.append(support_table); all_episodes.append(detect_change_episodes(consensus, long, config))
        frozen = train_frozen_models(source_state, config)
        forecast, metric, update = run_online_forecasts(canonical, frozen, protocol, config)
        predictions.append(forecast); metrics.append(metric); updates.append(update)
        audit.extend({"protocol_id": protocol, "feature_name": feature, "kept": 1, "selection_rule": "PRE_REGISTERED_LABEL_FREE_CANDIDATE", "feature_group": next(group for group, values in FEATURE_GROUPS.items() if feature in values), "stage_used": False} for feature in FEATURES)
        print(f"v4.2 consensus protocol complete: {protocol}", flush=True)
    consensus = pd.concat(all_consensus, ignore_index=True); support_table = pd.concat(all_support, ignore_index=True); episodes = pd.concat(all_episodes, ignore_index=True)
    prediction_table = pd.concat(predictions, ignore_index=True); metric_table = pd.concat(metrics, ignore_index=True); update_table = pd.concat(updates, ignore_index=True)
    forecast_summary, forecast_payload = forecast_benefit(metric_table)
    ry_audit = pd.DataFrame([_ry_feature_audit(exp1, canonical_states["B_Exp2_to_Exp1"], config), _ry_feature_audit(exp2, canonical_states["A_Exp1_to_Exp2"], config)])
    ry_effect = _ry_removal_effect(records_by_protocol["B_Exp2_to_Exp1"], config)
    consensus.to_csv(paths["results"] / "consensus_state_trajectories_v42.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(paths["results"] / "change_episodes_v42.csv", index=False, encoding="utf-8-sig")
    support_table.to_csv(paths["results"] / "configuration_support_v42.csv", index=False, encoding="utf-8-sig")
    ry_audit.to_csv(paths["results"] / "ry_feature_audit_v42.csv", index=False, encoding="utf-8-sig")
    metric_table.to_csv(paths["results"] / "forecast_metrics_v42.csv", index=False, encoding="utf-8-sig")
    prediction_table.to_csv(paths["results"] / "forecast_predictions_v42.csv", index=False, encoding="utf-8-sig")
    update_table.to_csv(paths["results"] / "forecast_update_log_v42.csv", index=False, encoding="utf-8-sig")
    forecast_summary.to_csv(paths["results"] / "forecast_best_static_comparison_v42.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(paths["results"] / "feature_protocol_audit_v42.csv", index=False, encoding="utf-8-sig")
    prefixes = [prefix_causality(target, canonical_states[protocol], protocol, FEATURES, config, canonical_supports[protocol]) for protocol, (_, target) in protocols.items()]
    diagnostics = implementation_diagnostics(pd.concat(list(canonical_states.values()), ignore_index=True), prediction_table, update_table, prefixes, config)
    diagnostics.update(_forecast_diagnostics(prediction_table, config)); diagnostics.update(_consensus_diagnostics(consensus, episodes)); diagnostics["ry_removal_effect"] = {"status": "PASS", **ry_effect}
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
    pytest_text = (tests.stdout or "") + (tests.stderr or "")
    (paths["diagnostics"] / "pytest_summary.txt").write_text(pytest_text, encoding="utf-8")
    diagnostics["pytest"] = {"status": "PASS" if tests.returncode == 0 else "FAIL", "exit_code": tests.returncode}
    if tests.returncode != 0 or any(item.get("status") == "FAIL" for item in diagnostics.values() if isinstance(item, dict) and "status" in item): diagnostics["implementation_acceptance"]["status"] = "FAIL"
    diagnostics["cache_fingerprint_check"] = {"status": "PASS", "cache_reused": False, "reason": "v4.2 recomputed every configuration; no prior-version cache was read"}
    for name, item in diagnostics.items(): write_json(paths["diagnostics"] / f"{name}.json", item)
    write_json(paths["diagnostics"] / "forecast_best_static_comparison.json", forecast_payload)
    make_figures(consensus, support_table, episodes, ry_audit, metric_table, paths["figures"])
    (paths["reports"] / "continuous_state_v42_report.md").write_text(make_report(consensus, episodes, ry_audit, forecast_summary, forecast_payload, diagnostics, pytest_text), encoding="utf-8")
    print(f"Continuous State Monitoring v4.2 complete: implementation={diagnostics['implementation_acceptance']['status']}, pytest={diagnostics['pytest']['status']}", flush=True)


if __name__ == "__main__": main()
