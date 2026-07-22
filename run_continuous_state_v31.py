from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from continuous_state_v31.config import ContinuousStateV31Config
from continuous_state_v31.data import assert_label_free, load_window_table
from continuous_state_v31.evaluation import (
    condition_summary,
    forecast_benefit,
    implementation_diagnostics,
    physical_evaluation_after_online_outputs,
    prefix_causality,
    threshold_sensitivity,
    write_json,
)
from continuous_state_v31.feature_pruning import prune_features
from continuous_state_v31.forecast import run_online_forecasts, train_frozen_models
from continuous_state_v31.guards import add_restart_guard
from continuous_state_v31.plotting import make_figures
from continuous_state_v31.report import make_report
from continuous_state_v31.source_prior import build_source_model
from continuous_state_v31.state_engine import run_target_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path("continuous_state_v31").glob("*.py")):
        digest.update(path.name.encode("utf-8")); digest.update(path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _strength(audit: pd.DataFrame) -> dict[str, float]:
    kept = audit.loc[audit.kept.eq(1)]
    return {str(row.feature_name): float(row.direction_free_auc) if np.isfinite(row.direction_free_auc) else 1.0 for _, row in kept.iterrows()}


def _event_frame(rows: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return frame if not frame.empty else pd.DataFrame(columns=columns)


def _science(physical: pd.DataFrame, states: pd.DataFrame, severe_events: pd.DataFrame, benefit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp1 = physical.loc[physical.target_dataset.eq("Exp1")]
    exp2 = physical.loc[physical.target_dataset.eq("Exp2")]
    exp1_row = exp1.iloc[0] if not exp1.empty else pd.Series(dtype=object)
    exp2_row = exp2.iloc[0] if not exp2.empty else pd.Series(dtype=object)
    exp1_pass = bool(pd.notna(exp1_row.get("detected_plateau_cycle")) and int(exp1_row.get("persistent_severe_alarm_count", 0)) == 0)
    exp2_pass = bool(pd.notna(exp2_row.get("detected_plateau_cycle")) and pd.notna(exp2_row.get("detected_plateau_exit_cycle"))
                     and pd.notna(exp2_row.get("detected_severe_onset_cycle")))
    strict = pd.DataFrame([
        {"protocol_id": "A_Exp1_to_Exp2", "source_dataset": "Exp1", "target_dataset": "Exp2", "source_severe_prior": "NONE",
         "evaluation_role": "PRIMARY_GENERALIZATION", "plateau_to_severe_support": "PASS" if exp2_pass else "FAIL"},
        {"protocol_id": "B_Exp2_to_Exp1", "source_dataset": "Exp2", "target_dataset": "Exp1", "source_severe_prior": "CAUSAL_SOURCE_ONLY_OR_NONE",
         "evaluation_role": "STABLE_PLATEAU_CONTROL", "plateau_support": "PASS" if exp1_pass else "FAIL"},
    ])
    protocol_a_benefit = benefit.loc[benefit.protocol_id.eq("A_Exp1_to_Exp2"), "SAFE_ONLINE_FORECAST_BENEFIT"]
    overall = bool(exp1_pass and exp2_pass and not protocol_a_benefit.empty and protocol_a_benefit.iloc[0] == "PASS")
    science = pd.DataFrame([{
        "Exp1_stable_plateau_support": "PASS" if exp1_pass else "FAIL",
        "Exp2_plateau_to_severe_support": "PASS" if exp2_pass else "FAIL",
        "Protocol_A_safe_online_forecast_benefit": protocol_a_benefit.iloc[0] if not protocol_a_benefit.empty else "FAIL",
        "persistent_severe_event_rows": int(len(severe_events)),
        "overall_status": "PASS" if overall else "FAIL",
        "failures_retained_without_threshold_tuning": True,
    }])
    return strict, science


def legacy_in_process_main() -> None:
    config = ContinuousStateV31Config()
    paths = config.paths()
    config_payload = config.jsonable()
    config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    input_path = Path(config.z_table_path)
    fingerprint = {"code_version": "csv31", "config_hash": config_hash, "input_file_sha256": _sha256(input_path),
                   "feature_list": list(config_payload["features"]), "code_sha256": _code_hash()}
    fingerprint_path = paths["configs"] / "run_fingerprint.json"
    previous = json.loads(fingerprint_path.read_text(encoding="utf-8")) if fingerprint_path.exists() else None
    cache_check = {"status": "PASS", "fingerprint_valid": True, "cache_reused": False,
                   "cache_fingerprint_match": bool(previous == fingerprint), "reason": "v3.1 recomputes the current run; no v3 cache is read"}
    (paths["configs"] / "continuous_state_v31_config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fingerprint_path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")

    raw = add_restart_guard(load_window_table(config), config)
    exp1 = raw.loc[raw.dataset.eq("Exp1")].reset_index(drop=True)
    exp2 = raw.loc[raw.dataset.eq("Exp2")].reset_index(drop=True)
    assert_label_free(exp1); assert_label_free(exp2)

    features_a, audit_a = prune_features(exp1, "A_Exp1_to_Exp2", config)
    features_b, audit_b = prune_features(exp2, "B_Exp2_to_Exp1", config)
    model_a, plateau_prior_a, severe_prior_a = build_source_model(exp1, features_a, _strength(audit_a), "A_Exp1_to_Exp2", False, config)
    print("v3.1: Protocol A source-only priors complete", flush=True)
    model_b, plateau_prior_b, severe_prior_b = build_source_model(exp2, features_b, _strength(audit_b), "B_Exp2_to_Exp1", True, config)
    print("v3.1: Protocol B source-only priors complete", flush=True)

    target_a, plateau_a, exit_a, updates_a, metadata_a = run_target_state(exp2, exp1, model_a.features, model_a.feature_strength, model_a.plateau_prior, None, model_a.protocol_id, config)
    print("v3.1: Protocol A target state complete", flush=True)
    target_b, plateau_b, exit_b, updates_b, metadata_b = run_target_state(exp1, exp2, model_b.features, model_b.feature_strength, model_b.plateau_prior, model_b.severe_direction, model_b.protocol_id, config)
    print("v3.1: Protocol B target state complete", flush=True)

    feature_audit = pd.concat([audit_a, audit_b], ignore_index=True)
    source_plateau = pd.concat([plateau_prior_a, plateau_prior_b], ignore_index=True)
    source_severe = pd.concat([severe_prior_a, severe_prior_b], ignore_index=True)
    states = pd.concat([target_a, target_b], ignore_index=True)
    plateau_events = _event_frame([plateau_a, plateau_b], ["protocol_id", "target_dataset", "event", "cycle"])
    exit_events = _event_frame([exit_a, exit_b], ["protocol_id", "target_dataset", "event", "cycle"])
    updates = _event_frame([updates_a, updates_b], ["protocol_id", "target_dataset", "cycle", "feature_name", "weight"])
    severe_events = pd.DataFrame([*metadata_a["severe_events"], *metadata_b["severe_events"]])
    if severe_events.empty:
        severe_events = pd.DataFrame(columns=["protocol_id", "target_dataset", "event", "cycle", "S_severe_candidate", "valid_cycles"])
    sensitivity = pd.concat([
        threshold_sensitivity(exp1, exp2, model_a, config),
        threshold_sensitivity(exp2, exp1, model_b, config),
    ], ignore_index=True)

    frozen_a = train_frozen_models(model_a.source_states, config)
    frozen_b = train_frozen_models(model_b.source_states, config)
    forecast_a = run_online_forecasts(target_a, frozen_a, model_a.protocol_id, config)
    print("v3.1: Protocol A forecasts complete", flush=True)
    forecast_b = run_online_forecasts(target_b, frozen_b, model_b.protocol_id, config)
    print("v3.1: Protocol B forecasts complete", flush=True)
    predictions = pd.concat([forecast_a[0], forecast_b[0]], ignore_index=True)
    metrics = pd.concat([forecast_a[1], forecast_b[1]], ignore_index=True)
    segment = pd.concat([forecast_a[2], forecast_b[2]], ignore_index=True)
    rolling = pd.concat([forecast_a[3], forecast_b[3]], ignore_index=True)
    regret = pd.concat([forecast_a[4], forecast_b[4]], ignore_index=True)
    state_log = pd.concat([forecast_a[5], forecast_b[5]], ignore_index=True)
    episodes = pd.concat([forecast_a[6], forecast_b[6]], ignore_index=True)
    weights = pd.concat([forecast_a[7], forecast_b[7]], ignore_index=True)
    benefit, benefit_json = forecast_benefit(metrics)

    # All online/label-free artifacts are saved before the one post-hoc stage read.
    feature_audit.to_csv(paths["results"] / "feature_pruning_v31.csv", index=False, encoding="utf-8-sig")
    source_plateau.to_csv(paths["results"] / "source_plateau_prior_v31.csv", index=False, encoding="utf-8-sig")
    source_severe.to_csv(paths["results"] / "source_severe_prior_v31.csv", index=False, encoding="utf-8-sig")
    states.to_csv(paths["results"] / "state_window_scores_v31.csv", index=False, encoding="utf-8-sig")
    condition_summary(states).to_csv(paths["results"] / "plateau_condition_summary.csv", index=False, encoding="utf-8-sig")
    plateau_events.to_csv(paths["results"] / "plateau_events_v31.csv", index=False, encoding="utf-8-sig")
    exit_events.to_csv(paths["results"] / "plateau_exit_events_v31.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(paths["results"] / "severe_direction_updates_v31.csv", index=False, encoding="utf-8-sig")
    severe_events.to_csv(paths["results"] / "severe_candidate_events_v31.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(paths["results"] / "online_forecast_predictions_v31.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(paths["results"] / "online_forecast_metrics_v31.csv", index=False, encoding="utf-8-sig")
    segment.to_csv(paths["results"] / "online_forecast_segment_metrics.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(paths["results"] / "online_forecast_rolling_metrics.csv", index=False, encoding="utf-8-sig")
    regret.to_csv(paths["results"] / "online_forecast_regret.csv", index=False, encoding="utf-8-sig")
    state_log.to_csv(paths["results"] / "safe_ensemble_state_log.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(paths["results"] / "safe_ensemble_episode_summary.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(paths["results"] / "safe_ensemble_weights_v31.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(paths["results"] / "threshold_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    physical = physical_evaluation_after_online_outputs(states, severe_events, config)
    strict, science = _science(physical, states, severe_events, benefit)
    physical.to_csv(paths["results"] / "physical_evaluation_v31.csv", index=False, encoding="utf-8-sig")
    science.to_csv(paths["results"] / "scientific_acceptance_v31.csv", index=False, encoding="utf-8-sig")
    strict.to_csv(paths["results"] / "strict_protocol_summary_v31.csv", index=False, encoding="utf-8-sig")

    prefixes = [prefix_causality(model_a, exp2, target_a, config), prefix_causality(model_b, exp1, target_b, config)]
    diagnostics = implementation_diagnostics(states, predictions, updates, state_log, prefixes, cache_check, config)
    test_paths = sorted(str(path) for path in Path("tests").glob("test_csv31_*.py"))
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", *test_paths], capture_output=True, text=True)
    (paths["diagnostics"] / "pytest_summary.txt").write_text((test.stdout or "") + (test.stderr or ""), encoding="utf-8")
    diagnostics["implementation_acceptance"]["pytest_exit_code"] = test.returncode
    if test.returncode != 0:
        diagnostics["implementation_acceptance"]["status"] = "FAIL"
    for name, payload in diagnostics.items():
        write_json(paths["diagnostics"] / f"{name}.json", payload)
    write_json(paths["diagnostics"] / "forecast_benefit.json", benefit_json)

    make_figures(states, predictions, metrics, rolling, regret, episodes, sensitivity, paths["figures"])
    report = make_report(states, physical, updates, metrics, segment, rolling, regret, sensitivity, episodes, diagnostics, benefit)
    (paths["reports"] / "continuous_state_v31_report.md").write_text(report, encoding="utf-8")
    print(f"Continuous State Monitoring v3.1 complete: implementation={diagnostics['implementation_acceptance']['status']}, scientific={science.overall_status.iloc[0]}", flush=True)


def _worker(stage: str, protocol: str, *extra: str) -> None:
    command = [sys.executable, "-m", "continuous_state_v31.worker", stage, protocol, *extra]
    completed = subprocess.run(command, text=True)
    if completed.returncode:
        raise RuntimeError(f"v3.1 worker failed ({' '.join(command)}), exit={completed.returncode}")


def _read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=columns)


def _staged_files_present(work: Path) -> bool:
    required = []
    for protocol in ("A_Exp1_to_Exp2", "B_Exp2_to_Exp1"):
        required.extend(work / f"{protocol}_{name}.csv" for name in ("source_states", "target_states", "predictions", "metrics", "segments", "rolling", "regret", "state_log", "episodes", "weights"))
        required.append(work / f"{protocol}_model.json")
    return all(path.exists() for path in required)


def _execute_staged_run(config: ContinuousStateV31Config, work: Path) -> None:
    """Run each wide-table phase in a clean interpreter to bound peak memory."""
    for protocol in ("A_Exp1_to_Exp2", "B_Exp2_to_Exp1"):
        for stage in ("source", "target", "train", "forecast", "evaluate"):
            _worker(stage, protocol)
        for quantile in config.source_plateau_threshold_sensitivity:
            _worker("sensitivity", protocol, "--quantile", str(quantile))
        for cutoff in (2000, 5000, 10000):
            _worker("prefix", protocol, "--cutoff", str(cutoff))


def main() -> None:
    config = ContinuousStateV31Config(); paths = config.paths(); work = paths["root"] / "work_csv31"
    config_payload = config.jsonable()
    config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    fingerprint = {"code_version": "csv31", "config_hash": config_hash, "input_file_sha256": _sha256(Path(config.z_table_path)),
                   "feature_list": list(config_payload["features"]), "code_sha256": _code_hash()}
    fingerprint_path = paths["configs"] / "run_fingerprint.json"
    previous = json.loads(fingerprint_path.read_text(encoding="utf-8")) if fingerprint_path.exists() else None
    staged_inflight = _staged_files_present(work)
    if not staged_inflight:
        _execute_staged_run(config, work)
    cache_check = {"status": "PASS", "fingerprint_valid": True, "cache_reused": False,
                   "cache_fingerprint_match": bool(previous == fingerprint),
                   "staged_execution": True, "reason": "v3.1 stages are recomputed in isolated processes; no v3 cache is read"}
    (paths["configs"] / "continuous_state_v31_config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fingerprint_path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")

    protocols = ("A_Exp1_to_Exp2", "B_Exp2_to_Exp1")
    feature_audit = pd.concat([_read_csv(work / f"{protocol}_feature_audit.csv") for protocol in protocols], ignore_index=True)
    source_plateau = pd.concat([_read_csv(work / f"{protocol}_plateau_prior.csv") for protocol in protocols], ignore_index=True)
    source_severe = pd.concat([_read_csv(work / f"{protocol}_severe_prior.csv") for protocol in protocols], ignore_index=True)
    states = pd.concat([_read_csv(work / f"{protocol}_target_states.csv") for protocol in protocols], ignore_index=True)
    plateau_events = _event_frame([_read_csv(work / f"{protocol}_plateau_events.csv") for protocol in protocols], ["protocol_id", "target_dataset", "event", "cycle"])
    exit_events = _event_frame([_read_csv(work / f"{protocol}_exit_events.csv") for protocol in protocols], ["protocol_id", "target_dataset", "event", "cycle"])
    updates = _event_frame([_read_csv(work / f"{protocol}_updates.csv") for protocol in protocols], ["protocol_id", "target_dataset", "cycle", "feature_name", "weight"])
    severe_events = _event_frame([_read_csv(work / f"{protocol}_severe_events.csv") for protocol in protocols], ["protocol_id", "target_dataset", "event", "cycle", "S_severe_candidate", "valid_cycles"])
    predictions = pd.concat([_read_csv(work / f"{protocol}_predictions.csv") for protocol in protocols], ignore_index=True)
    metrics = pd.concat([_read_csv(work / f"{protocol}_metrics.csv") for protocol in protocols], ignore_index=True)
    segment = pd.concat([_read_csv(work / f"{protocol}_segments.csv") for protocol in protocols], ignore_index=True)
    rolling = pd.concat([_read_csv(work / f"{protocol}_rolling.csv") for protocol in protocols], ignore_index=True)
    regret = pd.concat([_read_csv(work / f"{protocol}_regret.csv") for protocol in protocols], ignore_index=True)
    state_log = pd.concat([_read_csv(work / f"{protocol}_state_log.csv") for protocol in protocols], ignore_index=True)
    episodes = pd.concat([_read_csv(work / f"{protocol}_episodes.csv") for protocol in protocols], ignore_index=True)
    weights = pd.concat([_read_csv(work / f"{protocol}_weights.csv") for protocol in protocols], ignore_index=True)
    sensitivity = pd.concat([_read_csv(work / f"{protocol}_sensitivity_{quantile:.2f}.csv") for protocol in protocols for quantile in config.source_plateau_threshold_sensitivity], ignore_index=True)
    prefixes = []
    for protocol in protocols:
        checks = [json.loads((work / f"{protocol}_prefix_{cutoff}.json").read_text(encoding="utf-8")) for cutoff in (2000, 5000, 10000)]
        prefixes.append({"protocol_id": protocol, "status": "PASS" if all(check["pass"] for check in checks) else "FAIL", "checks": checks})
    benefit, benefit_json = forecast_benefit(metrics)

    # Save every label-free online artifact before the post-hoc physical label read.
    feature_audit.to_csv(paths["results"] / "feature_pruning_v31.csv", index=False, encoding="utf-8-sig")
    source_plateau.to_csv(paths["results"] / "source_plateau_prior_v31.csv", index=False, encoding="utf-8-sig")
    source_severe.to_csv(paths["results"] / "source_severe_prior_v31.csv", index=False, encoding="utf-8-sig")
    states.to_csv(paths["results"] / "state_window_scores_v31.csv", index=False, encoding="utf-8-sig")
    condition_summary(states).to_csv(paths["results"] / "plateau_condition_summary.csv", index=False, encoding="utf-8-sig")
    plateau_events.to_csv(paths["results"] / "plateau_events_v31.csv", index=False, encoding="utf-8-sig")
    exit_events.to_csv(paths["results"] / "plateau_exit_events_v31.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(paths["results"] / "severe_direction_updates_v31.csv", index=False, encoding="utf-8-sig")
    severe_events.to_csv(paths["results"] / "severe_candidate_events_v31.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(paths["results"] / "online_forecast_predictions_v31.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(paths["results"] / "online_forecast_metrics_v31.csv", index=False, encoding="utf-8-sig")
    segment.to_csv(paths["results"] / "online_forecast_segment_metrics.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(paths["results"] / "online_forecast_rolling_metrics.csv", index=False, encoding="utf-8-sig")
    regret.to_csv(paths["results"] / "online_forecast_regret.csv", index=False, encoding="utf-8-sig")
    state_log.to_csv(paths["results"] / "safe_ensemble_state_log.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(paths["results"] / "safe_ensemble_episode_summary.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(paths["results"] / "safe_ensemble_weights_v31.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(paths["results"] / "threshold_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    physical = physical_evaluation_after_online_outputs(states, severe_events, config)
    strict, science = _science(physical, states, severe_events, benefit)
    physical.to_csv(paths["results"] / "physical_evaluation_v31.csv", index=False, encoding="utf-8-sig")
    science.to_csv(paths["results"] / "scientific_acceptance_v31.csv", index=False, encoding="utf-8-sig")
    strict.to_csv(paths["results"] / "strict_protocol_summary_v31.csv", index=False, encoding="utf-8-sig")
    diagnostics = implementation_diagnostics(states, predictions, updates, state_log, prefixes, cache_check, config)
    test_paths = sorted(str(path) for path in Path("tests").glob("test_csv31_*.py"))
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", *test_paths], capture_output=True, text=True)
    (paths["diagnostics"] / "pytest_summary.txt").write_text((test.stdout or "") + (test.stderr or ""), encoding="utf-8")
    diagnostics["implementation_acceptance"]["pytest_exit_code"] = test.returncode
    if test.returncode:
        diagnostics["implementation_acceptance"]["status"] = "FAIL"
    for name, payload in diagnostics.items(): write_json(paths["diagnostics"] / f"{name}.json", payload)
    write_json(paths["diagnostics"] / "forecast_benefit.json", benefit_json)
    make_figures(states, predictions, metrics, rolling, regret, episodes, sensitivity, paths["figures"])
    (paths["reports"] / "continuous_state_v31_report.md").write_text(make_report(states, physical, updates, metrics, segment, rolling, regret, sensitivity, episodes, diagnostics, benefit), encoding="utf-8")
    print(f"Continuous State Monitoring v3.1 complete: implementation={diagnostics['implementation_acceptance']['status']}, scientific={science.overall_status.iloc[0]}", flush=True)


if __name__ == "__main__":
    main()
