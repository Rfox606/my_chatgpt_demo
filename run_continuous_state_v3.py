from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from continuous_state_v3.config import ContinuousStateV3Config
from continuous_state_v3.data import FORBIDDEN_COLUMNS, assert_label_free, load_window_table
from continuous_state_v3.evaluation import (ablation_summary, forecast_benefit, implementation_diagnostics,
                                            physical_evaluation_after_online_outputs, prefix_causality, write_json)
from continuous_state_v3.feature_pruning import prune_features
from continuous_state_v3.forecast import run_online_forecasts, train_frozen_models
from continuous_state_v3.guards import add_restart_guard
from continuous_state_v3.plotting import make_figures
from continuous_state_v3.report import make_report
from continuous_state_v3.source_prior import build_source_model
from continuous_state_v3.state_engine import run_target_state


def _strength(audit: pd.DataFrame) -> dict[str, float]:
    kept = audit.loc[audit.kept.eq(1)]
    return {str(row.feature_name): float(row.direction_free_auc) if np.isfinite(row.direction_free_auc) else 1. for _, row in kept.iterrows()}


def _severe_events(metadata: dict[str, object]) -> pd.DataFrame:
    rows = metadata.get("severe_events", [])
    return pd.DataFrame(rows, columns=["protocol_id", "target_dataset", "event", "cycle", "S_severe_candidate"])


def _read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=columns)


def _scientific_acceptance(states: pd.DataFrame, physical: pd.DataFrame, benefit: pd.DataFrame) -> dict[str, object]:
    exp1 = physical.loc[physical.target_dataset.eq("Exp1")].iloc[0]
    exp2 = physical.loc[physical.target_dataset.eq("Exp2")].iloc[0]
    exp1_pass = bool(pd.notna(exp1.detected_plateau_cycle) and exp1.persistent_severe_alarm_count == 0)
    exp2_state = states.loc[states.target_dataset.eq("Exp2")]
    exit_cycle = exp2.detected_plateau_exit_cycle
    severe_after = exp2_state.loc[exp2_state.center_cycle >= exit_cycle, "S_severe_candidate"].median() if pd.notna(exit_cycle) else np.nan
    severe_platform = exp2_state.loc[exp2_state.center_cycle < exit_cycle, "S_severe_candidate"].median() if pd.notna(exit_cycle) else np.nan
    exp2_pass = bool(pd.notna(exp2.detected_plateau_cycle) and pd.notna(exit_cycle) and pd.notna(severe_after) and pd.notna(severe_platform) and severe_after > severe_platform)
    return {"Exp1_plateau_support": "PASS" if exp1_pass else "FAIL", "Exp2_plateau_to_severe_support": "PASS" if exp2_pass else "FAIL", "SAFE_ONLINE_FORECAST_BENEFIT": benefit.to_dict(orient="records"), "status": "PASS" if exp1_pass and exp2_pass and bool((benefit.SAFE_ONLINE_FORECAST_BENEFIT == "PASS").any()) else "FAIL"}


def main() -> None:
    config = ContinuousStateV3Config(); paths = config.paths()
    (paths["configs"] / "continuous_state_v3_config.json").write_text(json.dumps(config.jsonable(), ensure_ascii=False, indent=2), encoding="utf-8")
    raw = add_restart_guard(load_window_table(config), config)
    exp1 = raw.loc[raw.dataset.eq("Exp1")].reset_index(drop=True); exp2 = raw.loc[raw.dataset.eq("Exp2")].reset_index(drop=True)
    assert_label_free(exp1); assert_label_free(exp2)
    features_a, audit_a = prune_features(exp1, "A_Exp1_to_Exp2", config)
    features_b, audit_b = prune_features(exp2, "B_Exp2_to_Exp1", config)
    model_a, plateau_prior_a, severe_prior_a = build_source_model(exp1, features_a, _strength(audit_a), "A_Exp1_to_Exp2", False, config)
    print("v3: Protocol A source prior complete", flush=True)
    model_b, plateau_prior_b, severe_prior_b = build_source_model(exp2, features_b, _strength(audit_b), "B_Exp2_to_Exp1", True, config)
    print("v3: Protocol B source prior complete", flush=True)
    cache_dir = paths["root"] / ".cache"; cache_dir.mkdir(exist_ok=True)
    state_cache = paths["results"] / "state_window_scores_v3.csv"
    if state_cache.exists():
        states = _read_csv(state_cache)
        target_a = states.loc[states.protocol_id.eq(model_a.protocol_id)].copy(); target_b = states.loc[states.protocol_id.eq(model_b.protocol_id)].copy()
        plateau_events = _read_csv(paths["results"] / "plateau_events.csv")
        exit_events = _read_csv(paths["results"] / "plateau_exit_events.csv")
        severe_updates = _read_csv(paths["results"] / "severe_direction_updates.csv")
        severe_events = _read_csv(paths["results"] / "severe_candidate_events.csv", ["protocol_id", "target_dataset", "event", "cycle", "S_severe_candidate"])
        print("v3: reused completed target-state cache", flush=True)
    else:
        target_a, plateau_a, exit_a, updates_a, meta_a = run_target_state(exp2, exp1, model_a.features, model_a.feature_strength, model_a.plateau_prior, None, model_a.protocol_id, config)
        print("v3: Protocol A target stream complete", flush=True)
        target_b, plateau_b, exit_b, updates_b, meta_b = run_target_state(exp1, exp2, model_b.features, model_b.feature_strength, model_b.plateau_prior, model_b.severe_direction, model_b.protocol_id, config)
        print("v3: Protocol B target stream complete", flush=True)
        states = pd.concat([target_a, target_b], ignore_index=True); plateau_events = pd.concat([plateau_a, plateau_b], ignore_index=True); exit_events = pd.concat([exit_a, exit_b], ignore_index=True); severe_updates = pd.concat([updates_a, updates_b], ignore_index=True); severe_events = pd.concat([_severe_events(meta_a), _severe_events(meta_b)], ignore_index=True)
        states.to_csv(state_cache, index=False, encoding="utf-8-sig"); plateau_events.to_csv(paths["results"] / "plateau_events.csv", index=False, encoding="utf-8-sig"); exit_events.to_csv(paths["results"] / "plateau_exit_events.csv", index=False, encoding="utf-8-sig"); severe_updates.to_csv(paths["results"] / "severe_direction_updates.csv", index=False, encoding="utf-8-sig"); severe_events.to_csv(paths["results"] / "severe_candidate_events.csv", index=False, encoding="utf-8-sig")
    forecast_cache = paths["results"] / "online_forecast_predictions_v3.csv"
    if forecast_cache.exists() and (paths["results"] / "online_forecast_metrics_v3.csv").exists() and (paths["results"] / "safe_ensemble_weights.csv").exists():
        forecasts = _read_csv(forecast_cache); metrics = _read_csv(paths["results"] / "online_forecast_metrics_v3.csv"); alphas = _read_csv(paths["results"] / "safe_ensemble_weights.csv")
        print("v3: reused completed forecast cache", flush=True)
    else:
        frozen_a = train_frozen_models(model_a.source_states, config); frozen_b = train_frozen_models(model_b.source_states, config)
        forecast_a, metrics_a, alpha_a = run_online_forecasts(target_a, frozen_a, model_a.protocol_id, config)
        print("v3: Protocol A forecasts complete", flush=True)
        forecast_b, metrics_b, alpha_b = run_online_forecasts(target_b, frozen_b, model_b.protocol_id, config)
        print("v3: Protocol B forecasts complete", flush=True)
        forecasts = pd.concat([forecast_a, forecast_b], ignore_index=True); metrics = pd.concat([metrics_a, metrics_b], ignore_index=True); alphas = pd.concat([alpha_a, alpha_b], ignore_index=True)
        forecasts.to_csv(forecast_cache, index=False, encoding="utf-8-sig"); metrics.to_csv(paths["results"] / "online_forecast_metrics_v3.csv", index=False, encoding="utf-8-sig"); alphas.to_csv(paths["results"] / "safe_ensemble_weights.csv", index=False, encoding="utf-8-sig")
    target_a["source_dataset"] = "Exp1"; target_a["target_dataset"] = "Exp2"
    target_b["source_dataset"] = "Exp2"; target_b["target_dataset"] = "Exp1"
    states = pd.concat([target_a, target_b], ignore_index=True)
    feature_audit = pd.concat([audit_a, audit_b], ignore_index=True); source_plateau = pd.concat([plateau_prior_a, plateau_prior_b], ignore_index=True); source_severe = pd.concat([severe_prior_a, severe_prior_b], ignore_index=True)
    # Save all label-free online artifacts before any post-hoc read of stage information.
    feature_audit.to_csv(paths["results"] / "feature_pruning_v3.csv", index=False, encoding="utf-8-sig")
    source_plateau.to_csv(paths["results"] / "source_plateau_prior.csv", index=False, encoding="utf-8-sig")
    source_severe.to_csv(paths["results"] / "source_severe_prior.csv", index=False, encoding="utf-8-sig")
    states.to_csv(paths["results"] / "state_window_scores_v3.csv", index=False, encoding="utf-8-sig")
    plateau_events.to_csv(paths["results"] / "plateau_events.csv", index=False, encoding="utf-8-sig")
    exit_events.to_csv(paths["results"] / "plateau_exit_events.csv", index=False, encoding="utf-8-sig")
    severe_updates.to_csv(paths["results"] / "severe_direction_updates.csv", index=False, encoding="utf-8-sig")
    severe_events.to_csv(paths["results"] / "severe_candidate_events.csv", index=False, encoding="utf-8-sig")
    forecasts.to_csv(paths["results"] / "online_forecast_predictions_v3.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(paths["results"] / "online_forecast_metrics_v3.csv", index=False, encoding="utf-8-sig")
    alphas.to_csv(paths["results"] / "safe_ensemble_weights.csv", index=False, encoding="utf-8-sig")
    prefix_a_path, prefix_b_path = cache_dir / "prefix_a.json", cache_dir / "prefix_b.json"
    prefix_a = json.loads(prefix_a_path.read_text(encoding="utf-8")) if prefix_a_path.exists() else prefix_causality(model_a, exp2, target_a, config)
    if not prefix_a_path.exists(): write_json(prefix_a_path, prefix_a)
    prefix_b = json.loads(prefix_b_path.read_text(encoding="utf-8")) if prefix_b_path.exists() else prefix_causality(model_b, exp1, target_b, config)
    if not prefix_b_path.exists(): write_json(prefix_b_path, prefix_b)
    prefixes = [prefix_a, prefix_b]
    diagnostics = implementation_diagnostics(states, forecasts, prefixes, config)
    benefit, benefit_json = forecast_benefit(metrics)
    ablation = ablation_summary(states, forecasts, metrics, benefit)
    strict = pd.DataFrame([{"protocol_id": model_a.protocol_id, "source_dataset": "Exp1", "target_dataset": "Exp2", "evaluation_mode": "strict_online_evaluation", "source_severe_prior": "NONE"}, {"protocol_id": model_b.protocol_id, "source_dataset": "Exp2", "target_dataset": "Exp1", "evaluation_mode": "strict_online_control", "source_severe_prior": "AVAILABLE" if model_b.severe_direction is not None else "NONE"}])
    reference_library = pd.DataFrame([{"library_mode": "Protocol_C_reference_library_only", "source_datasets": "Exp1+Exp2", "features_intersection": ";".join(sorted(set(features_a).intersection(features_b))), "not_an_independent_target_generalization_claim": True}, *source_plateau.to_dict(orient="records")])
    # Now and only now use historical stage fields for post-hoc physical comparison.
    physical = physical_evaluation_after_online_outputs(states, severe_events, config)
    science = _scientific_acceptance(states, physical, benefit)
    ablation.to_csv(paths["results"] / "ablation_summary_v3.csv", index=False, encoding="utf-8-sig")
    strict.to_csv(paths["results"] / "strict_protocol_summary.csv", index=False, encoding="utf-8-sig")
    physical.to_csv(paths["results"] / "physical_evaluation_v3.csv", index=False, encoding="utf-8-sig")
    reference_library.to_csv(paths["results"] / "future_reference_library.csv", index=False, encoding="utf-8-sig")
    pytest_paths = sorted(str(path) for path in Path("tests").glob("test_csv3_*.py")); test = subprocess.run([sys.executable, "-m", "pytest", "-q", *pytest_paths], capture_output=True, text=True)
    (paths["diagnostics"] / "pytest_summary.txt").write_text((test.stdout or "") + (test.stderr or ""), encoding="utf-8")
    diagnostics["implementation_acceptance"]["pytest_exit_code"] = test.returncode
    diagnostics["implementation_acceptance"]["status"] = "PASS" if diagnostics["implementation_acceptance"]["status"] == "PASS" and test.returncode == 0 else "FAIL"
    for name, payload in diagnostics.items(): write_json(paths["diagnostics"] / f"{name}.json", payload)
    write_json(paths["diagnostics"] / "scientific_acceptance.json", science); write_json(paths["diagnostics"] / "safe_ensemble_rollback_check.json", diagnostics["safe_ensemble_rollback_check"])
    make_figures(states, forecasts, metrics, ablation, physical, paths["figures"])
    (paths["reports"] / "continuous_state_v3_report.md").write_text(make_report(physical, severe_updates, states, ablation, metrics, benefit, diagnostics), encoding="utf-8")
    print(f"Continuous State Monitoring v3 complete: implementation={diagnostics['implementation_acceptance']['status']}, scientific={science['status']}")


if __name__ == "__main__":
    main()
