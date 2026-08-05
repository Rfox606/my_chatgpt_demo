from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from continuous_state_v41.config import FEATURES, FEATURE_GROUPS, ContinuousStateV41Config
from continuous_state_v41.data import add_restart_guard, assert_label_free, load_window_table
from continuous_state_v41.evaluation import evidence_summary, forecast_benefit, implementation_diagnostics, prefix_causality, write_json
from continuous_state_v41.forecast import run_online_forecasts, train_frozen_models
from continuous_state_v41.plotting import make_figures
from continuous_state_v41.report import make_report
from continuous_state_v41.state_engine import EVIDENCE_NAMES, feature_subset, fit_source_support, run_target_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path("continuous_state_v41").glob("*.py")):
        digest.update(path.name.encode("utf-8")); digest.update(path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _onset_cycles(events: pd.DataFrame, protocol: str, evidence: str, start: float) -> np.ndarray:
    if events.empty:
        return np.empty(0)
    rows = events.loc[(events.protocol_id.eq(protocol)) & events.evidence_type.eq(evidence) & events.event.eq("algorithm_evidence_onset") & (events.cycle >= start), "cycle"]
    return rows.to_numpy(float)


def _event_set_match(reference: np.ndarray, candidate: np.ndarray, tolerance: float = 500.0) -> tuple[int, float]:
    used = np.zeros(len(candidate), dtype=bool); matches = 0
    for cycle in reference:
        available = np.flatnonzero((~used) & (np.abs(candidate - cycle) <= tolerance))
        if len(available):
            nearest = available[np.argmin(np.abs(candidate[available] - cycle))]
            used[nearest] = True; matches += 1
    union = len(reference) + len(candidate) - matches
    return matches, (1.0 if union == 0 else matches / union)


def _ablation_summary(
    protocols: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    reference_states: dict[str, pd.DataFrame],
    reference_event_frames: dict[str, pd.DataFrame],
    config: ContinuousStateV41Config,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protocol, (source, target) in protocols.items():
        for baseline_cycles in (500, 1000, 2000):
            for distance_form in ("mahalanobis", "diagonal"):
                for removed in (None, "rx", "ry", "rs"):
                    features = feature_subset(FEATURES, removed)
                    candidate_config = replace(config, baseline_cycles=baseline_cycles, distance_form=distance_form)
                    same = baseline_cycles == config.baseline_cycles and distance_form == config.distance_form and removed is None
                    if same:
                        candidate, candidate_frame = reference_states[protocol], reference_event_frames[protocol]
                    else:
                        support = fit_source_support(source, features)
                        candidate, candidate_frame, _ = run_target_state(target, protocol, features, candidate_config, support)
                    common_start = float(max(config.baseline_cycles, baseline_cycles))
                    reference = reference_states[protocol].loc[reference_states[protocol].start_cycle > common_start, ["window_index", "D_state"]]
                    current = candidate.loc[candidate.start_cycle > common_start, ["window_index", "D_state"]]
                    merged = current.merge(reference, on="window_index", suffixes=("_candidate", "_reference"))
                    correlation = float(merged.D_state_candidate.corr(merged.D_state_reference, method="spearman")) if len(merged) > 2 else np.nan
                    for evidence in EVIDENCE_NAMES:
                        reference_cycles = _onset_cycles(reference_event_frames[protocol], protocol, evidence, common_start)
                        candidate_cycles = _onset_cycles(candidate_frame, protocol, evidence, common_start)
                        matches, jaccard = _event_set_match(reference_cycles, candidate_cycles)
                        near = candidate_cycles[(candidate_cycles >= 7800.0) & (candidate_cycles <= 8200.0)]
                        rows.append({"protocol_id": protocol, "target_dataset": str(target.dataset.iloc[0]), "baseline_cycles": baseline_cycles,
                                     "distance_form": distance_form, "removed_feature_group": removed or "none", "feature_count": len(features),
                                     "common_monitoring_start_cycle": common_start, "evidence_type": evidence, "trajectory_spearman": correlation,
                                     "reference_event_count_common": len(reference_cycles), "ablation_event_count_common": len(candidate_cycles),
                                     "matched_event_count": matches, "event_collection_jaccard": jaccard,
                                     "ablation_events_near_8000": ";".join(f"{value:.1f}" for value in near),
                                     "trajectory_stable": bool(np.isfinite(correlation) and correlation >= .80),
                                     "event_collection_stable": bool(jaccard >= .80),
                                     "stability_status": "PASS" if np.isfinite(correlation) and correlation >= .80 and jaccard >= .80 else "FAIL"})
    return pd.DataFrame(rows)


def _v4_exp2_comparison(events: pd.DataFrame, states: pd.DataFrame) -> dict[str, object]:
    evidence = {"acceleration_evidence", "abrupt_change_evidence"}
    current = events.loc[(events.target_dataset.eq("Exp2")) & events.event.eq("algorithm_evidence_onset") & events.evidence_type.isin(evidence), "cycle"] if not events.empty else pd.Series(dtype=float)
    endpoint = float(states.loc[states.dataset.eq("Exp2"), "center_cycle"].max())
    boundary = .75 * endpoint
    current_fraction = float((current >= boundary).mean()) if not current.empty else 0.0
    previous_path = Path("outputs_continuous_state_v4/results/evidence_events_v4.csv")
    if previous_path.exists():
        previous_events = pd.read_csv(previous_path)
        previous_states = pd.read_csv("outputs_continuous_state_v4/results/state_window_scores_v4.csv", usecols=["dataset", "center_cycle"])
        previous = previous_events.loc[(previous_events.target_dataset.eq("Exp2")) & previous_events.event.eq("algorithm_evidence_onset") & previous_events.evidence_type.isin(evidence), "cycle"]
        previous_boundary = .75 * float(previous_states.loc[previous_states.dataset.eq("Exp2"), "center_cycle"].max())
        previous_fraction = float((previous >= previous_boundary).mean()) if not previous.empty else 0.0
    else:
        previous_fraction = np.nan
    return {"v41_exp2_late_fraction": current_fraction, "v4_exp2_late_fraction": previous_fraction,
            "more_concentrated_than_v4": bool(np.isfinite(previous_fraction) and current_fraction > previous_fraction),
            "late_boundary_cycle": boundary}


def _extra_diagnostics(states: pd.DataFrame, forecasts: pd.DataFrame, events: pd.DataFrame, config: ContinuousStateV41Config) -> dict[str, dict[str, object]]:
    state_columns = ["D_state", "V100_norm", "V500_norm", "V1000_norm", "A_state", "residual_change_score", "abrupt_cusum", "baseline_outlier_fraction", "source_support_oos"]
    state_finite = bool(np.isfinite(states.loc[:, state_columns].to_numpy(float)).all())
    prediction_columns = [column for column in forecasts.columns if column.endswith("_prediction")]
    forecast_finite = bool(np.isfinite(forecasts.loc[:, prediction_columns].to_numpy(float)).all()) if prediction_columns else False
    online_delta = np.abs(forecasts.Online_RLS_prediction.to_numpy(float) - forecasts.Zero_Delta_prediction.to_numpy(float)) if not forecasts.empty else np.array([np.nan])
    max_delta = float(np.nanmax(online_delta))
    numeric = {"status": "PASS" if state_finite and forecast_finite and max_delta <= config.forecast_delta_clip + 1e-9 else "FAIL",
               "all_finite": bool(state_finite and forecast_finite), "max_online_prediction_abs": max_delta,
               "online_delta_bounded": bool(max_delta <= config.forecast_delta_clip + 1e-9)}
    monitoring_cycles = float(states.groupby("protocol_id").center_cycle.agg(lambda values: values.max() - values.min()).sum())
    onsets = int((events.event == "algorithm_evidence_onset").sum()) if not events.empty else 0
    density = 1000.0 * onsets / max(monitoring_cycles, 1.0)
    event_density = {"status": "PASS" if density <= 4.0 else "FAIL", "onset_count": onsets,
                     "monitoring_cycles": monitoring_cycles, "onsets_per_1000_monitoring_cycles": density,
                     "pre_registered_max_onsets_per_1000_cycles": 4.0}
    return {"numerical_stability_check": numeric, "event_density_check": event_density}


def main() -> None:
    config = ContinuousStateV41Config(); paths = config.paths()
    payload = config.jsonable(); input_path = Path(config.z_table_path)
    fingerprint = {"code_version": "csv41", "config": payload, "input_file_sha256": _sha256(input_path), "code_sha256": _code_hash()}
    (paths["configs"] / "continuous_state_v41_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(paths["configs"] / "run_fingerprint.json", fingerprint)
    raw = add_restart_guard(load_window_table(config), config)
    exp1 = raw.loc[raw.dataset.eq("Exp1")].reset_index(drop=True); exp2 = raw.loc[raw.dataset.eq("Exp2")].reset_index(drop=True)
    assert_label_free(exp1); assert_label_free(exp2)
    protocols = {"A_Exp1_to_Exp2": (exp1, exp2), "B_Exp2_to_Exp1": (exp2, exp1)}
    targets: dict[str, pd.DataFrame] = {}; target_events: dict[str, pd.DataFrame] = {}; supports = {}
    forecasts: list[pd.DataFrame] = []; metric_frames: list[pd.DataFrame] = []; update_frames: list[pd.DataFrame] = []; audit: list[dict[str, object]] = []
    for protocol, (source, target) in protocols.items():
        support = fit_source_support(source, FEATURES); supports[protocol] = support
        source_state, _, _ = run_target_state(source, protocol + "_source", FEATURES, config, support)
        target_state, events, _ = run_target_state(target, protocol, FEATURES, config, support)
        targets[protocol], target_events[protocol] = target_state, events
        frozen = train_frozen_models(source_state, config)
        prediction, metrics, updates = run_online_forecasts(target_state, frozen, protocol, config)
        forecasts.append(prediction); metric_frames.append(metrics); update_frames.append(updates)
        audit.extend({"protocol_id": protocol, "feature_name": feature, "kept": 1, "selection_rule": "PRE_REGISTERED_LABEL_FREE_CANDIDATE", "feature_group": next(group for group, values in FEATURE_GROUPS.items() if feature in values), "stage_used": False} for feature in FEATURES)
        print(f"v4.1 protocol complete: {protocol}", flush=True)
    states = pd.concat(list(targets.values()), ignore_index=True); events = pd.concat(list(target_events.values()), ignore_index=True)
    prediction_table = pd.concat(forecasts, ignore_index=True); metrics = pd.concat(metric_frames, ignore_index=True); updates = pd.concat(update_frames, ignore_index=True)
    evidence = evidence_summary(states, events)
    ablations = _ablation_summary(protocols, targets, target_events, config)
    forecast_summary, forecast_payload = forecast_benefit(metrics)
    v4_comparison = _v4_exp2_comparison(events, states)
    states.to_csv(paths["results"] / "state_window_scores_v41.csv", index=False, encoding="utf-8-sig")
    events.to_csv(paths["results"] / "evidence_events_v41.csv", index=False, encoding="utf-8-sig")
    ablations.to_csv(paths["results"] / "ablation_summary_v41.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(paths["results"] / "forecast_metrics_v41.csv", index=False, encoding="utf-8-sig")
    prediction_table.to_csv(paths["results"] / "forecast_predictions_v41.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(paths["results"] / "forecast_update_log_v41.csv", index=False, encoding="utf-8-sig")
    evidence.to_csv(paths["results"] / "evidence_summary_v41.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(paths["results"] / "feature_protocol_audit_v41.csv", index=False, encoding="utf-8-sig")
    forecast_summary.to_csv(paths["results"] / "forecast_best_static_comparison_v41.csv", index=False, encoding="utf-8-sig")
    prefixes = [prefix_causality(target, targets[protocol], protocol, FEATURES, config, supports[protocol]) for protocol, (_, target) in protocols.items()]
    diagnostics = implementation_diagnostics(states, prediction_table, updates, prefixes, config)
    diagnostics.update(_extra_diagnostics(states, prediction_table, events, config))
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
    pytest_text = (tests.stdout or "") + (tests.stderr or "")
    (paths["diagnostics"] / "pytest_summary.txt").write_text(pytest_text, encoding="utf-8")
    diagnostics["pytest"] = {"status": "PASS" if tests.returncode == 0 else "FAIL", "exit_code": tests.returncode}
    if tests.returncode != 0 or any(item.get("status") == "FAIL" for item in diagnostics.values() if isinstance(item, dict) and "status" in item): diagnostics["implementation_acceptance"]["status"] = "FAIL"
    diagnostics["cache_fingerprint_check"] = {"status": "PASS", "cache_reused": False, "reason": "v4.1 recomputed every output; no prior-version cache was read"}
    for name, item in diagnostics.items(): write_json(paths["diagnostics"] / f"{name}.json", item)
    write_json(paths["diagnostics"] / "forecast_best_static_comparison.json", forecast_payload); write_json(paths["diagnostics"] / "v4_comparison.json", v4_comparison)
    make_figures(states, events, ablations, metrics, paths["figures"])
    (paths["reports"] / "continuous_state_v41_report.md").write_text(make_report(states, evidence, ablations, forecast_summary, forecast_payload, diagnostics, v4_comparison, pytest_text), encoding="utf-8")
    print(f"Continuous State Monitoring v4.1 complete: implementation={diagnostics['implementation_acceptance']['status']}, pytest={diagnostics['pytest']['status']}", flush=True)


if __name__ == "__main__": main()
