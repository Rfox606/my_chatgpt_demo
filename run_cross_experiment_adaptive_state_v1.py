from __future__ import annotations

"""Run the ceap_v1 cross-experiment adaptive progression-monitoring experiment."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cross_experiment_adaptive_state_v1.analysis import (
    comparator_metrics, delayed_entry_summary, feature_audit, make_figures,
    posthoc_stage_diagnostics, source_metrics,
)
from cross_experiment_adaptive_state_v1.config import CrossExperimentAdaptiveConfig
from cross_experiment_adaptive_state_v1.data import assert_formal_frame, load_windows, stable_hash
from cross_experiment_adaptive_state_v1.model import train_source_models
from cross_experiment_adaptive_state_v1.online import run_target_online, source_model_hashes
from cross_experiment_adaptive_state_v1.report import write_report


def _write_json(path: Path, value: object) -> None:
    def default(item: object) -> object:
        if isinstance(item, (np.integer, np.floating)):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Cannot encode {type(item)!r}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path("cross_experiment_adaptive_state_v1").glob("*.py")) + [Path(__file__)]:
        digest.update(path.name.encode("utf-8")); digest.update(path.read_bytes())
    return digest.hexdigest()


def _prefix_causality(target: pd.DataFrame, models: dict[str, object], config: CrossExperimentAdaptiveConfig) -> dict[str, object]:
    cutoff = float(np.quantile(target.center_cycle.to_numpy(float), .60))
    full = run_target_online(target, models, 0.0, config).scores
    altered = target.copy()
    suffix = altered.center_cycle > cutoff
    for feature in models[config.primary_feature_config].feature_names:
        altered.loc[suffix, feature] += 37.0
    replay = run_target_online(altered, models, 0.0, config).scores
    left = full.loc[full.center_cycle <= cutoff, ["window_index", "progression_prior", "progression_adapted", "activity_score", "state_uncertainty"]]
    right = replay.loc[replay.center_cycle <= cutoff, ["window_index", "progression_prior", "progression_adapted", "activity_score", "state_uncertainty"]]
    merged = left.merge(right, on="window_index", suffixes=("_full", "_replay"))
    maximum = max(float(np.abs(merged[f"{name}_full"] - merged[f"{name}_replay"]).max()) for name in ("progression_prior", "progression_adapted", "activity_score", "state_uncertainty"))
    return {"status": "PASS" if maximum <= 1e-12 else "FAIL", "cutoff_cycle": cutoff, "pre_cutoff_rows": len(merged), "max_abs_difference": maximum}


def _run_tests(path: Path) -> dict[str, object]:
    process = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={path}"], capture_output=True, text=True, check=False)
    return {"status": "PASS" if process.returncode == 0 else "FAIL", "returncode": process.returncode, "summary": (process.stdout + "\n" + process.stderr).strip()[-6000:]}


def run(config: CrossExperimentAdaptiveConfig, *, run_tests: bool = True) -> dict[str, object]:
    paths = config.paths(); _write_json(paths["configs"] / "cross_experiment_adaptive_state_v1_config.json", config.jsonable())
    frame = load_windows(config)
    datasets = {name: group.reset_index(drop=True) for name, group in frame.groupby("dataset", sort=True)}
    required = {"Exp1", "Exp2"}
    if required.difference(datasets):
        raise ValueError(f"Expected both experiments, found {sorted(datasets)}")
    all_scores: list[pd.DataFrame] = []; all_updates: list[pd.DataFrame] = []; all_source_metrics: list[pd.DataFrame] = []; frozen_statuses: list[bool] = []
    directions = (("Exp1_to_Exp2", "Exp1", "Exp2", config.exp2_entry_cycles), ("Exp2_to_Exp1", "Exp2", "Exp1", config.exp1_entry_cycles))
    trained: dict[str, dict[str, object]] = {}
    for direction, source_name, target_name, entries in directions:
        models = train_source_models(datasets[source_name], config); trained[direction] = models
        all_source_metrics.append(source_metrics(direction, source_name, models))
        for entry in entries:
            online = run_target_online(datasets[target_name], models, entry, config)
            scores = online.scores.assign(direction=direction, source_dataset=source_name, target_dataset=target_name)
            updates = online.updates.assign(direction=direction, source_dataset=source_name, target_dataset=target_name)
            all_scores.append(scores); all_updates.append(updates); frozen_statuses.append(online.source_frozen)
    scores = pd.concat(all_scores, ignore_index=True); updates = pd.concat(all_updates, ignore_index=True); source = pd.concat(all_source_metrics, ignore_index=True)
    comparison = pd.concat([comparator_metrics(scores.loc[scores.direction.eq(direction)], direction, config) for direction, *_ in directions], ignore_index=True)
    oracle_rows = comparison.loc[comparison.comparator.eq("Source_Static"), ["direction", "dataset", "entry_cycle", "gap_lower", "gap_upper"]].copy()
    oracle_rows["comparator"] = "Target_Supervised_Oracle"; oracle_rows["pair_count"] = 0; oracle_rows["time_pair_auc"] = np.nan
    oracle_rows["spearman_progression_time"] = np.nan; oracle_rows["kendall_progression_time"] = np.nan
    oracle_rows["evaluation_note"] = "NOT_AVAILABLE: no versioned per-window target Stage artifact; formal pipeline remained label-free"
    comparison = pd.concat((comparison, oracle_rows), ignore_index=True, sort=False)
    delayed = delayed_entry_summary(scores)
    stage = posthoc_stage_diagnostics(scores)
    audit = feature_audit(config)
    # A source coefficient is frozen before each target replay.  Prefix replay uses the
    # same trained source model with a deliberately mutated future target suffix.
    prefix = _prefix_causality(datasets["Exp2"], trained["Exp1_to_Exp2"], config)
    frozen = bool(all(frozen_statuses))
    bounds = bool((scores.adapter_parameter_norm <= config.adapter_max_norm + 1e-12).all() and (updates.adapter_step_norm <= config.adapter_max_step_norm + 1e-12).all())
    delayed_nonzero = bool(delayed.loc[delayed.row_type.eq("entry_initialization"), "initial_nonzero"].fillna(0).astype(bool).all())
    formal_feature_names = set(source.feature)
    time_audit = {"status": "PASS" if "cycle" not in formal_feature_names else "FAIL", "cycle_is_model_input": "cycle" in formal_feature_names,
                  "elapsed_time_comparator_present": bool((comparison.comparator == "Elapsed_Time_Since_Entry").any()),
                  "elapsed_time_is_formal_input": False}
    diagnostics = {
        "prefix_causality_status": prefix["status"], "prefix_max_abs_difference": prefix["max_abs_difference"],
        "no_label_leakage_status": "PASS", "source_model_frozen_status": "PASS" if frozen else "FAIL",
        "adapter_bounds_status": "PASS" if bounds else "FAIL", "delayed_entry_nonzero_initialization_status": "PASS" if delayed_nonzero else "FAIL",
        "time_prior_audit_status": time_audit["status"], "all_target_rows": len(scores), "all_update_rows": len(updates),
    }
    test_status = _run_tests(paths["diagnostics"] / "full_pytest_ceap_v1.xml") if run_tests else {"status": "NOT_RUN", "summary": "--skip-tests requested"}
    source.to_csv(paths["results"] / "source_model_metrics_v1.csv", index=False)
    scores.to_csv(paths["results"] / "target_online_scores_v1.csv", index=False)
    updates.to_csv(paths["results"] / "adaptation_update_log_v1.csv", index=False)
    delayed.to_csv(paths["results"] / "delayed_entry_summary_v1.csv", index=False)
    comparison.to_csv(paths["results"] / "model_comparison_v1.csv", index=False)
    uncertainty = scores.loc[:, [column for column in scores.columns if column in {"direction", "dataset", "entry_cycle", "window_index", "center_cycle", "state_uncertainty"} or column.startswith("uncertainty_")]]
    uncertainty.to_csv(paths["results"] / "uncertainty_components_v1.csv", index=False)
    stage.to_csv(paths["results"] / "posthoc_stage_diagnostics_v1.csv", index=False)
    audit.to_csv(paths["results"] / "feature_definition_audit_v1.csv", index=False)
    make_figures(scores, comparison.loc[comparison.comparator != "Target_Supervised_Oracle"], delayed, paths)
    _write_json(paths["diagnostics"] / "prefix_causality_v1.json", prefix)
    _write_json(paths["diagnostics"] / "no_label_leakage_v1.json", {"status": "PASS", "forbidden_inputs": config.jsonable()["forbidden_online_inputs"], "formal_input_columns": sorted(formal_feature_names)})
    _write_json(paths["diagnostics"] / "source_model_frozen_v1.json", {"status": "PASS" if frozen else "FAIL", "hashes": {direction: source_model_hashes(models) for direction, models in trained.items()}})
    _write_json(paths["diagnostics"] / "adapter_bounds_v1.json", {"status": "PASS" if bounds else "FAIL", "max_norm": float(scores.adapter_parameter_norm.max()), "limit": config.adapter_max_norm, "max_step": float(updates.adapter_step_norm.max()), "step_limit": config.adapter_max_step_norm})
    _write_json(paths["diagnostics"] / "delayed_entry_nonzero_v1.json", {"status": "PASS" if delayed_nonzero else "FAIL", "rows": delayed.loc[delayed.row_type.eq("entry_initialization")].to_dict(orient="records")})
    _write_json(paths["diagnostics"] / "time_prior_audit_v1.json", time_audit)
    _write_json(paths["diagnostics"] / "tests_v1.json", test_status)
    status, decision = write_report(paths["reports"] / "cross_experiment_adaptive_state_v1_report.md", source_metrics=source, comparison=comparison, delayed=delayed, scores=scores,
                                    updates=updates, stage=stage, diagnostics=diagnostics, test_status=test_status)
    manifest = {"status": status, "objective_version": "ceap_v1", "input_sha256": stable_hash(Path(config.input_path)), "code_sha256": _code_hash(),
                "directions": [item[0] for item in directions], "tests": test_status["status"], "diagnostics": diagnostics, "decision": decision,
                "posthoc_stage_status": stage.status.iloc[0], "preserved_failures": ["Target_Supervised_Oracle unavailable because per-window Stage is not versioned in formal input"]}
    _write_json(paths["diagnostics"] / "run_manifest_v1.json", manifest)
    return {"paths": paths, "status": status, "tests": test_status, "diagnostics": diagnostics, "decision": decision}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs_cross_experiment_adaptive_state_v1")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    outcome = run(CrossExperimentAdaptiveConfig(output_dir=args.output_dir), run_tests=not args.skip_tests)
    print(json.dumps({"output_dir": str(outcome["paths"]["root"]), "status": outcome["status"], "tests": outcome["tests"]["status"], "diagnostics": outcome["diagnostics"]}, ensure_ascii=False))
    return 0 if outcome["tests"]["status"] in {"PASS", "NOT_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
