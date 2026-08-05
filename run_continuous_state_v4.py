from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from continuous_state_v4.config import FEATURES, ContinuousStateV4Config
from continuous_state_v4.data import add_restart_guard, assert_label_free, load_window_table
from continuous_state_v4.evaluation import (
    evidence_summary,
    forecast_benefit,
    implementation_diagnostics,
    prefix_causality,
    write_json,
)
from continuous_state_v4.forecast import run_online_forecasts, train_frozen_models
from continuous_state_v4.plotting import make_figures
from continuous_state_v4.report import make_report
from continuous_state_v4.state_engine import EVIDENCE_NAMES, feature_subset, run_target_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path("continuous_state_v4").glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _event_cycle(events: pd.DataFrame, protocol: str, evidence: str) -> float:
    rows = events.loc[(events.protocol_id.eq(protocol)) & events.evidence_type.eq(evidence) & events.event.eq("algorithm_evidence_onset"), "cycle"]
    return float(rows.min()) if not rows.empty else float("nan")


def _ablation_summary(
    targets: dict[str, pd.DataFrame],
    reference_states: dict[str, pd.DataFrame],
    reference_events: dict[str, pd.DataFrame],
    config: ContinuousStateV4Config,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protocol, target in targets.items():
        for baseline_cycles in (250, 500, 1000):
            for distance_form in ("mahalanobis", "diagonal"):
                for removed in (None, "rx", "ry", "rs"):
                    features = feature_subset(FEATURES, removed)
                    same_as_reference = baseline_cycles == config.baseline_cycles and distance_form == config.distance_form and removed is None
                    if same_as_reference:
                        candidate, candidate_events = reference_states[protocol], reference_events[protocol]
                    else:
                        candidate_config = replace(config, baseline_cycles=baseline_cycles, distance_form=distance_form)
                        candidate, candidate_events, _ = run_target_state(target, protocol, features, candidate_config)
                    reference = reference_states[protocol].loc[:, ["window_index", "D_state"]]
                    merged = candidate.loc[:, ["window_index", "D_state"]].merge(reference, on="window_index", suffixes=("_candidate", "_reference"))
                    correlation = float(merged.D_state_candidate.corr(merged.D_state_reference, method="spearman"))
                    for evidence in EVIDENCE_NAMES:
                        reference_cycle = _event_cycle(reference_events[protocol], protocol, evidence)
                        candidate_cycle = _event_cycle(candidate_events, protocol, evidence)
                        same_presence = bool(np.isnan(reference_cycle) == np.isnan(candidate_cycle))
                        delta = candidate_cycle - reference_cycle if np.isfinite(candidate_cycle) and np.isfinite(reference_cycle) else np.nan
                        event_stable = bool(same_presence and (not np.isfinite(delta) or abs(delta) <= 500.0))
                        trajectory_stable = bool(np.isfinite(correlation) and correlation >= .80)
                        rows.append({"protocol_id": protocol, "target_dataset": str(target.dataset.iloc[0]), "baseline_cycles": baseline_cycles,
                                     "distance_form": distance_form, "removed_feature_group": removed or "none", "feature_count": len(features),
                                     "evidence_type": evidence, "trajectory_spearman": correlation,
                                     "reference_first_onset_cycle": reference_cycle, "ablation_first_onset_cycle": candidate_cycle,
                                     "event_position_delta_cycles": delta, "event_presence_matches": same_presence,
                                     "trajectory_stable": trajectory_stable, "event_position_stable": event_stable,
                                     "stability_status": "PASS" if trajectory_stable and event_stable else "FAIL"})
    return pd.DataFrame(rows)


def main() -> None:
    config = ContinuousStateV4Config()
    paths = config.paths()
    config_payload = config.jsonable()
    input_path = Path(config.z_table_path)
    fingerprint = {"code_version": "csv4", "config": config_payload, "input_file_sha256": _sha256(input_path), "code_sha256": _code_hash()}
    (paths["configs"] / "continuous_state_v4_config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(paths["configs"] / "run_fingerprint.json", fingerprint)

    raw = add_restart_guard(load_window_table(config), config)
    exp1 = raw.loc[raw.dataset.eq("Exp1")].reset_index(drop=True)
    exp2 = raw.loc[raw.dataset.eq("Exp2")].reset_index(drop=True)
    assert_label_free(exp1); assert_label_free(exp2)
    protocols = {
        "A_Exp1_to_Exp2": (exp1, exp2),
        "B_Exp2_to_Exp1": (exp2, exp1),
    }
    audit_rows = []
    target_states: dict[str, pd.DataFrame] = {}
    source_states: dict[str, pd.DataFrame] = {}
    event_frames: dict[str, pd.DataFrame] = {}
    predictions: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    updates: list[pd.DataFrame] = []
    for protocol, (source, target) in protocols.items():
        source_state, _, _ = run_target_state(source, protocol + "_source", FEATURES, config)
        target_state, events, _ = run_target_state(target, protocol, FEATURES, config)
        source_states[protocol], target_states[protocol], event_frames[protocol] = source_state, target_state, events
        audit_rows.extend({"protocol_id": protocol, "feature_name": feature, "kept": 1,
                           "selection_rule": "PRE_REGISTERED_LABEL_FREE_CANDIDATE", "stage_used": False} for feature in FEATURES)
        frozen = train_frozen_models(source_state, config)
        forecast, metric, update = run_online_forecasts(target_state, frozen, protocol, config)
        predictions.append(forecast); metrics.append(metric); updates.append(update)
        print(f"v4 protocol complete: {protocol}", flush=True)

    states = pd.concat(list(target_states.values()), ignore_index=True)
    events = pd.concat(list(event_frames.values()), ignore_index=True)
    forecasts = pd.concat(predictions, ignore_index=True)
    forecast_metrics = pd.concat(metrics, ignore_index=True)
    forecast_updates = pd.concat(updates, ignore_index=True)
    evidence = evidence_summary(states, events)
    ablations = _ablation_summary({key: value[1] for key, value in protocols.items()}, target_states, event_frames, config)
    forecast_summary, forecast_payload = forecast_benefit(forecast_metrics)

    states.to_csv(paths["results"] / "state_window_scores_v4.csv", index=False, encoding="utf-8-sig")
    events.to_csv(paths["results"] / "evidence_events_v4.csv", index=False, encoding="utf-8-sig")
    ablations.to_csv(paths["results"] / "ablation_summary_v4.csv", index=False, encoding="utf-8-sig")
    forecast_metrics.to_csv(paths["results"] / "forecast_metrics_v4.csv", index=False, encoding="utf-8-sig")
    forecasts.to_csv(paths["results"] / "forecast_predictions_v4.csv", index=False, encoding="utf-8-sig")
    forecast_updates.to_csv(paths["results"] / "forecast_update_log_v4.csv", index=False, encoding="utf-8-sig")
    evidence.to_csv(paths["results"] / "evidence_summary_v4.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(paths["results"] / "feature_protocol_audit_v4.csv", index=False, encoding="utf-8-sig")
    forecast_summary.to_csv(paths["results"] / "forecast_best_static_comparison_v4.csv", index=False, encoding="utf-8-sig")

    prefixes = [prefix_causality(target, target_states[protocol], protocol, FEATURES, config) for protocol, (_, target) in protocols.items()]
    diagnostics = implementation_diagnostics(states, forecasts, forecast_updates, prefixes, config)
    # Run the repository suite as well as the v4-specific tests so the shipped
    # artifact records integration compatibility, not merely local unit tests.
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
    pytest_text = (tests.stdout or "") + (tests.stderr or "")
    (paths["diagnostics"] / "pytest_summary.txt").write_text(pytest_text, encoding="utf-8")
    diagnostics["pytest"] = {"status": "PASS" if tests.returncode == 0 else "FAIL", "exit_code": tests.returncode}
    if tests.returncode != 0:
        diagnostics["implementation_acceptance"]["status"] = "FAIL"
    diagnostics["cache_fingerprint_check"] = {"status": "PASS", "cache_reused": False, "reason": "v4 recomputed every output; no earlier-version cache was read"}
    for name, payload in diagnostics.items():
        write_json(paths["diagnostics"] / f"{name}.json", payload)
    write_json(paths["diagnostics"] / "forecast_best_static_comparison.json", forecast_payload)

    make_figures(states, events, ablations, forecast_metrics, paths["figures"])
    report = make_report(states, events, evidence, ablations, forecast_summary, forecast_payload, diagnostics, pytest_text)
    (paths["reports"] / "continuous_state_v4_report.md").write_text(report, encoding="utf-8")
    print(f"Continuous State Monitoring v4 complete: implementation={diagnostics['implementation_acceptance']['status']}, pytest={diagnostics['pytest']['status']}", flush=True)


if __name__ == "__main__":
    main()
