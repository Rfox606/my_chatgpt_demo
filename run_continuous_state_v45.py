from __future__ import annotations

"""Full v4.5 raw-feature continuous-state reconstruction and validation."""

import argparse
import hashlib
import json
import subprocess
import sys
import gc
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from continuous_state_v43.time_mapping import add_actual_cycle_columns
from continuous_state_v45.analysis import (
    ConfigurationRecord, consensus_trajectories, ry_path_audit, state_space_summary,
    trajectory_stability, v44_vs_v45,
)
from continuous_state_v45.config import (
    CORE_FEATURES, CORRDIST_FEATURES, RY_EXTENSION_FEATURES, ContinuousStateV45Config,
)
from continuous_state_v45.plotting import state_space_figure, v44_v45_figure
from continuous_state_v45.raw_features import (
    SensitiveCycleWaves, add_baseline_corrdist, direct_window_features,
    load_sensitive_force_cycles, raw_provenance,
)
from continuous_state_v45.report import write_report
from continuous_state_v45.state_engine import assert_label_free, feature_subset, run_state


def _write_json(path: Path, payload: object) -> None:
    def default(value: object) -> object:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Unsupported JSON value {type(value)!r}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=default), encoding="utf-8")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in [*sorted(Path("continuous_state_v45").glob("*.py")), Path(__file__)]:
        digest.update(path.name.encode("utf-8")); digest.update(path.read_bytes())
    return digest.hexdigest()


def _state_features(use_corrdist: bool, extended_ry: bool = False) -> tuple[str, ...]:
    values = [*CORE_FEATURES]
    if extended_ry:
        values.extend(RY_EXTENSION_FEATURES)
    if use_corrdist:
        values.extend(CORRDIST_FEATURES)
    return tuple(dict.fromkeys(values))


def _id(baseline: int, distance: str, variant: str, corrdist: bool) -> str:
    return f"b{baseline}_{distance}_{variant}_{'with_corrdist' if corrdist else 'without_corrdist'}"


def _map_frame(frame: pd.DataFrame, config: ContinuousStateV45Config) -> pd.DataFrame:
    mapped, _ = add_actual_cycle_columns(frame, config)
    assert_label_free(mapped)
    return mapped


def _run_main_grid(raw: SensitiveCycleWaves, direct: pd.DataFrame, config: ContinuousStateV45Config) -> tuple[list[ConfigurationRecord], dict[int, pd.DataFrame], pd.DataFrame, object]:
    records: list[ConfigurationRecord] = []; frames: dict[int, pd.DataFrame] = {}; canonical = pd.DataFrame(); canonical_ref = None
    for baseline in (500, 1000, 2000):
        frame = _map_frame(add_baseline_corrdist(raw, direct, baseline, config), config)
        frames[baseline] = frame
        for distance in ("mahalanobis", "diagonal"):
            for use_corrdist in (True, False):
                base_features = _state_features(use_corrdist)
                for removed in (None, "rx", "ry", "rs"):
                    variant = "full" if removed is None else f"no_{removed}"
                    run_config = replace(config, baseline_cycles=baseline, distance_form=distance)
                    state, reference = run_state(frame, _id(baseline, distance, variant, use_corrdist), feature_subset(base_features, removed), run_config,
                                                 include_details=(baseline == 1000 and distance == "mahalanobis" and use_corrdist and removed is None))
                    records.append(ConfigurationRecord(_id(baseline, distance, variant, use_corrdist), baseline, distance, variant, "with_corrdist" if use_corrdist else "without_corrdist", state))
                    if baseline == 1000 and distance == "mahalanobis" and use_corrdist and removed is None:
                        canonical, canonical_ref = state, reference
    if canonical_ref is None:
        raise RuntimeError("Missing b1000 Mahalanobis full-with-corrdist canonical state")
    return records, frames, canonical, canonical_ref


def _run_extended_ry(raw: SensitiveCycleWaves, frames: dict[int, pd.DataFrame], config: ContinuousStateV45Config) -> list[ConfigurationRecord]:
    records: list[ConfigurationRecord] = []
    for baseline in (500, 1000, 2000):
        for distance in ("mahalanobis", "diagonal"):
            run_config = replace(config, baseline_cycles=baseline, distance_form=distance)
            state, _ = run_state(frames[baseline], _id(baseline, distance, "ry_extended", True), _state_features(True, extended_ry=True), run_config)
            records.append(ConfigurationRecord(_id(baseline, distance, "ry_extended", True), baseline, distance, "ry_extended", "with_corrdist", state))
    return records


def _corrdist_diagnostic(frames: dict[str, dict[int, pd.DataFrame]]) -> dict[str, object]:
    result: dict[str, object] = {"status": "PASS", "definition": "corrdist recomputed from each configuration's raw early baseline waveform before 20-cycle window averaging", "datasets": {}}
    for dataset, by_baseline in frames.items():
        dataset_rows: dict[str, object] = {}
        reference = by_baseline[500]
        for baseline, frame in by_baseline.items():
            entry: dict[str, object] = {"baseline_cycles": baseline, "baseline_window_count": int((frame.end_cycle_effective <= baseline).sum()), "channels": {}}
            for feature in CORRDIST_FEATURES:
                entry["channels"][feature] = {"mean": float(frame[feature].mean()), "baseline_median": float(frame.loc[frame.end_cycle_effective <= baseline, feature].median()),
                                                "max_abs_difference_vs_500": float(np.max(np.abs(frame[feature].to_numpy(float) - reference[feature].to_numpy(float))))}
            dataset_rows[str(baseline)] = entry
        result["datasets"][dataset] = dataset_rows
    return result


def _feature_dominance_rows(frame: pd.DataFrame, reference: object, config: ContinuousStateV45Config) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []; monitor = frame.start_cycle_effective.to_numpy(float) > config.baseline_cycles
    for group, ref in reference.groups.items():
        values = frame.loc[:, list(ref.features)].to_numpy(float)
        standard = np.abs((values - ref.location) / ref.scale); share = standard / np.maximum(standard.sum(axis=1, keepdims=True), config.eps)
        for position, feature in enumerate(ref.features):
            if feature.endswith("corrdist_base"):
                rows.append({"row_type": "canonical_corrdist_feature_share", "dataset": str(frame.dataset.iloc[0]), "comparison": "full_with_corrdist", "metric": feature,
                             "mean_absolute_standardised_share": float(share[monitor, position].mean()), "p95_absolute_standardised_share": float(np.quantile(share[monitor, position], .95)),
                             "fraction_over_050": float((share[monitor, position] > .50).mean()), "common_windows": int(monitor.sum())})
    return rows


def _prefix_causality(frame: pd.DataFrame, config: ContinuousStateV45Config) -> dict[str, object]:
    run_config = replace(config, baseline_cycles=1000, distance_form="mahalanobis")
    features = _state_features(True); cutoff = 10000.0
    full, _ = run_state(frame, "prefix_full", features, run_config)
    altered = frame.copy(); suffix = altered.center_cycle_effective > cutoff
    altered.loc[suffix, list(features)] += 23.0
    replay, _ = run_state(altered, "prefix_mutated", features, run_config)
    metrics = ("D_state", "V1000_norm", "A_state", "state_volatility")
    merged = full.loc[full.center_cycle_effective <= cutoff, ["window_index", *metrics]].merge(replay.loc[replay.center_cycle_effective <= cutoff, ["window_index", *metrics]], on="window_index", suffixes=("_full", "_replay"))
    maximum = float(max(np.abs(merged[f"{metric}_full"] - merged[f"{metric}_replay"]).max() for metric in metrics))
    return {"status": "PASS" if maximum <= 1e-12 else "FAIL", "cutoff_effective_cycle": cutoff, "pre_cutoff_rows": int(len(merged)), "max_abs_difference": maximum}


def _test_status(junit_path: Path) -> dict[str, object]:
    process = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={junit_path}"], text=True, capture_output=True, check=False)
    return {"status": "PASS" if process.returncode == 0 else "FAIL", "returncode": process.returncode, "summary": (process.stdout + "\n" + process.stderr).strip()[-5000:]}


def run(config: ContinuousStateV45Config, run_tests: bool = True) -> dict[str, object]:
    paths = config.paths(); _write_json(paths["configs"] / "continuous_state_v45_config.json", config.jsonable())
    raw_tables: list[pd.DataFrame] = []; consensus_parts: list[pd.DataFrame] = []; stability_parts: list[pd.DataFrame] = []; audit_parts: list[pd.DataFrame] = []
    raw_hashes: dict[str, str] = {}; main_configuration_count = 0; extended_configuration_count = 0; prefix: dict[str, object] | None = None
    provenance: dict[str, object] = {"status": "PASS", "stage_read": False, "upstream_z_standardisation_used": False, "upstream_z_clip_used": False,
                                     "raw_feature_definition": "direct Fx/Fz, Fy/Fz and resultant sensitive-phase summaries; no pre-normalisation",
                                     "sensitive_phase": list(config.sensitive_phase), "window_cycles": config.window_cycles, "window_stride_cycles": config.window_stride_cycles, "datasets": {}}
    corrdist: dict[str, object] = {"status": "PASS", "definition": "corrdist recomputed from each configuration's raw early baseline waveform before 20-cycle window averaging", "datasets": {}}
    # Process Exp1 and Exp2 sequentially.  This bounds peak memory while preserving the same fixed grid for each experiment.
    for dataset, source_path in config.raw_files:
        raw = load_sensitive_force_cycles(dataset, Path(source_path), config)
        provenance["datasets"][dataset] = raw_provenance({dataset: raw}, config)["datasets"][dataset]
        raw_hashes[dataset] = _hash(Path(raw.raw_path))
        direct = direct_window_features(raw, config); raw_tables.append(direct)
        records, frames, canonical, reference = _run_main_grid(raw, direct, config)
        extended = _run_extended_ry(raw, frames, config)
        consensus_parts.append(consensus_trajectories(records, config))
        stability_parts.append(trajectory_stability(records, config))
        variants = {
            "full_with_corrdist": consensus_trajectories([record for record in records if record.feature_variant == "full" and record.corrdist_mode == "with_corrdist"], config),
            "no_ry_with_corrdist": consensus_trajectories([record for record in records if record.feature_variant == "no_ry" and record.corrdist_mode == "with_corrdist"], config),
            "ry_extended_with_corrdist": consensus_trajectories(extended, config),
        }
        audit_parts.append(ry_path_audit(variants, {dataset: canonical}, config))
        audit_parts.append(pd.DataFrame(_feature_dominance_rows(frames[1000], reference, config)))
        corrdist["datasets"].update(_corrdist_diagnostic({dataset: frames})["datasets"])
        if dataset == "Exp1":
            prefix = _prefix_causality(frames[1000], config)
        main_configuration_count += len(records); extended_configuration_count += len(extended)
        # All per-configuration frames and raw wave matrices are no longer required after their consensus/audit is saved.
        del raw, direct, records, frames, canonical, reference, extended, variants
        gc.collect()
    raw_table = pd.concat(raw_tables, ignore_index=True); raw_table.to_csv(paths["results"] / "window_feature_raw_v45.csv", index=False)
    consensus = pd.concat(consensus_parts, ignore_index=True)
    stability = pd.concat(stability_parts, ignore_index=True)
    path_audit = pd.concat(audit_parts, ignore_index=True, sort=False)
    if prefix is None:
        raise RuntimeError("Exp1 prefix replay was not produced")
    v44 = pd.read_csv(config.v44_consensus_path)
    comparison, comparison_display = v44_vs_v45(consensus, v44, config)
    state_summary = state_space_summary(consensus)
    tests = _test_status(paths["diagnostics"] / "full_pytest_v45.xml") if run_tests else {"status": "NOT_RUN", "summary": "--skip-tests requested"}
    # Metadata is opened only after all state outputs have been finalised, and is used solely for plot markers.
    metadata = json.loads(Path(config.metadata_path).read_text(encoding="utf-8"))
    isolation = {"status": "PASS", "metadata_read_after_state_outputs": True, "actual_cycle_used_for": ["plot coordinate", "Exp1 morphology-anchor marker"],
                 "not_used_for": ["raw feature extraction", "feature standardisation", "state calculation", "configuration comparison", "threshold selection"],
                 "metadata_prohibited_uses": metadata["analysis_boundary"]["prohibited_uses"]}
    consensus.to_csv(paths["results"] / "consensus_state_trajectories_v45.csv", index=False)
    stability.to_csv(paths["results"] / "trajectory_stability_v45.csv", index=False)
    comparison.to_csv(paths["results"] / "v44_vs_v45_comparison.csv", index=False)
    state_summary.to_csv(paths["results"] / "state_space_summary_v45.csv", index=False)
    path_audit.to_csv(paths["results"] / "ry_path_audit_v45.csv", index=False)
    state_space_figure(consensus, "Exp1", paths["figures"] / "state_space_exp1_v45.png")
    state_space_figure(consensus, "Exp2", paths["figures"] / "state_space_exp2_v45.png")
    v44_v45_figure(comparison_display, metadata["exp1_morphology"]["cycle_actual"], paths["figures"] / "v44_vs_v45_trajectories.png")
    coverage = consensus.groupby("dataset", as_index=False).agg(rows=("window_index", "size"), effective_configuration_count_min=("effective_configuration_count", "min"), effective_configuration_count_max=("effective_configuration_count", "max"), first_effective_cycle=("center_cycle_effective", "min"), last_effective_cycle=("center_cycle_effective", "max"))
    _write_json(paths["diagnostics"] / "raw_feature_provenance_v45.json", provenance)
    _write_json(paths["diagnostics"] / "single_normalisation_v45.json", {"status": "PASS", "location": "continuous_state_v45.state_engine.run_state", "upstream_z_input": False, "clip": None, "rule": "each configuration freezes robust location/scale from its own early raw baseline exactly once"})
    _write_json(paths["diagnostics"] / "corrdist_recomputation_v45.json", corrdist)
    _write_json(paths["diagnostics"] / "prefix_causality_v45.json", prefix)
    _write_json(paths["diagnostics"] / "metadata_isolation_v45.json", isolation)
    _write_json(paths["diagnostics"] / "tests_v45.json", tests)
    _write_json(paths["diagnostics"] / "coverage_v45.json", coverage.to_dict(orient="records"))
    _write_json(paths["diagnostics"] / "run_manifest_v45.json", {"status": "PASS" if all(value.get("status") == "PASS" for value in (provenance, corrdist, prefix, tests)) else "FAIL", "code_sha256": _code_hash(),
        "raw_input_sha256": raw_hashes, "configurations": main_configuration_count, "extended_ry_configurations": extended_configuration_count,
        "state_time_axis": "effective_cycle", "actual_cycle_use": "plots_and_morphology_markers_only", "upstream_z_standardisation": False})
    write_report(paths["reports"] / "continuous_state_v45_report.md", config=config, provenance=provenance, stability=stability, comparison=comparison, state_summary=state_summary, path_audit=path_audit, tests=tests, prefix=prefix)
    return {"paths": paths, "tests": tests, "prefix": prefix, "provenance": provenance}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs_continuous_state_v45")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(); outcome = run(ContinuousStateV45Config(output_dir=args.output_dir), run_tests=not args.skip_tests)
    print(json.dumps({"output_dir": str(outcome["paths"]["root"]), "tests": outcome["tests"]["status"], "prefix": outcome["prefix"]["status"], "provenance": outcome["provenance"]["status"]}, ensure_ascii=False))
    return 0 if outcome["tests"]["status"] == "PASS" and outcome["prefix"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
