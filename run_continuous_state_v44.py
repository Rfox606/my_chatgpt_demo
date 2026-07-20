from __future__ import annotations

"""Run the v4.4 trajectory-first, label-free physical validation workflow."""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from continuous_state_v44.analysis import (
    ConfigurationRecord,
    consensus_trajectories,
    exp_pattern_comparison,
    input_provenance,
    morphology_interval_alignment,
    trajectory_stability,
)
from continuous_state_v44.config import BASE_FEATURES, EXTENDED_RY_FEATURES, ContinuousStateV44Config
from continuous_state_v44.data import assert_label_free, load_window_table
from continuous_state_v44.plotting import consensus_trajectory_figure, morphology_interval_figure
from continuous_state_v44.report import write_report
from continuous_state_v44.ry_audit import ry_group_audit
from continuous_state_v44.state_engine import feature_subset, run_target_state, state_columns


def _write_json(path: Path, payload: object) -> None:
    def default(value: object) -> object:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialise {type(value)!r}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=default), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in [*sorted(Path("continuous_state_v44").glob("*.py")), Path(__file__)]:
        digest.update(path.name.encode("utf-8")); digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_id(baseline: int, distance: str, variant: str) -> str:
    return f"b{baseline}_{distance}_{variant}"


def _run_grid(frame: pd.DataFrame, base_config: ContinuousStateV44Config) -> tuple[list[ConfigurationRecord], dict[str, pd.DataFrame]]:
    """Run the fixed 3 x 2 x 4 state grid.  No physical outcome is consulted here."""
    records: list[ConfigurationRecord] = []
    canonical: dict[str, pd.DataFrame] = {}
    for baseline in (500, 1000, 2000):
        for distance in ("mahalanobis", "diagonal"):
            for removed in (None, "rx", "ry", "rs"):
                features = feature_subset(BASE_FEATURES, removed)
                run_config = replace(base_config, baseline_cycles=baseline, distance_form=distance)
                variant = f"no_{removed}" if removed else "full_p2p"
                is_canonical = baseline == 1000 and distance == "mahalanobis" and removed is None
                state, _ = run_target_state(frame, _config_id(baseline, distance, variant), features, run_config, include_velocity_details=is_canonical)
                record = ConfigurationRecord(_config_id(baseline, distance, variant), baseline, distance, variant, removed or "none", state)
                records.append(record)
                if is_canonical:
                    canonical["p2p"] = state
    return records, canonical


def _run_extended_grid(frame: pd.DataFrame, base_config: ContinuousStateV44Config) -> tuple[list[ConfigurationRecord], pd.DataFrame]:
    """Apply the same fixed baseline/distance grid with the predeclared expanded ry group."""
    records: list[ConfigurationRecord] = []
    canonical = pd.DataFrame()
    for baseline in (500, 1000, 2000):
        for distance in ("mahalanobis", "diagonal"):
            run_config = replace(base_config, baseline_cycles=baseline, distance_form=distance)
            is_canonical = baseline == 1000 and distance == "mahalanobis"
            state, _ = run_target_state(frame, _config_id(baseline, distance, "full_ry_extended"), EXTENDED_RY_FEATURES, run_config, include_velocity_details=is_canonical)
            records.append(ConfigurationRecord(_config_id(baseline, distance, "full_ry_extended"), baseline, distance, "full_ry_extended", "none", state))
            if is_canonical:
                canonical = state
    return records, canonical


def _prefix_causality(frame: pd.DataFrame, config: ContinuousStateV44Config) -> dict[str, object]:
    """Change only a suffix, then prove frozen-baseline states before that suffix are unchanged."""
    cutoff = min(10000.0, float(frame.center_cycle_effective.quantile(.45)))
    full, _ = run_target_state(frame, "causality_full", BASE_FEATURES, config)
    altered = frame.copy()
    suffix = altered.center_cycle_effective > cutoff
    altered.loc[suffix, list(BASE_FEATURES)] += 137.0
    replay, _ = run_target_state(altered, "causality_suffix_changed", BASE_FEATURES, config)
    columns = list(state_columns())
    merged = full.loc[full.center_cycle_effective <= cutoff, ["window_index", *columns]].merge(
        replay.loc[replay.center_cycle_effective <= cutoff, ["window_index", *columns]], on="window_index", suffixes=("_full", "_replay"))
    differences = [np.abs(merged[f"{column}_full"] - merged[f"{column}_replay"]).max() for column in columns] if not merged.empty else [np.inf]
    maximum = float(np.max(differences))
    return {"status": "PASS" if maximum <= 1e-12 else "FAIL", "cutoff_effective_cycle": cutoff, "pre_cutoff_rows": int(len(merged)), "max_abs_difference": maximum,
            "rule": "only suffix feature values changed; all pre-cutoff frozen-baseline states must agree"}


def _test_status(junit_path: Path) -> dict[str, object]:
    """Run the repository suite, retaining a machine-readable result alongside v4.4 diagnostics."""
    command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit_path}"]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    summary = (process.stdout + "\n" + process.stderr).strip()
    return {"status": "PASS" if process.returncode == 0 else "FAIL", "returncode": process.returncode, "summary": summary[-4000:]}


def run(config: ContinuousStateV44Config, run_tests: bool = True) -> dict[str, object]:
    paths = config.paths()
    _write_json(paths["configs"] / "continuous_state_v44_config.json", config.jsonable())
    # The reader intentionally excludes stage and morphology columns before any state feature matrix exists.
    combined = load_window_table(config, EXTENDED_RY_FEATURES)
    assert_label_free(combined)
    frames = {dataset: group.sort_values(["center_cycle_effective", "window_index"]).reset_index(drop=True) for dataset, group in combined.groupby("dataset", sort=True)}
    all_records: list[ConfigurationRecord] = []
    consensus_frames: list[pd.DataFrame] = []
    velocity_details: list[pd.DataFrame] = []
    variant_consensus: dict[str, list[pd.DataFrame]] = {"p2p_only": [], "ry_extended": [], "no_ry": []}
    extended_canonical: dict[str, pd.DataFrame] = {}
    for dataset, frame in frames.items():
        records, canonical = _run_grid(frame, config)
        extended_records, extended_state = _run_extended_grid(frame, config)
        all_records.extend(records)
        consensus, _ = consensus_trajectories(records, config)
        consensus_frames.append(consensus)
        extended_consensus, _ = consensus_trajectories(extended_records, config)
        p2p_records = [record for record in records if record.feature_variant == "full_p2p"]
        no_ry_records = [record for record in records if record.feature_variant == "no_ry"]
        p2p_consensus, _ = consensus_trajectories(p2p_records, config)
        no_ry_consensus, _ = consensus_trajectories(no_ry_records, config)
        variant_consensus["p2p_only"].append(p2p_consensus)
        variant_consensus["ry_extended"].append(extended_consensus)
        variant_consensus["no_ry"].append(no_ry_consensus)
        extended_canonical[dataset] = extended_state
        if "p2p" not in canonical:
            raise RuntimeError(f"Canonical p2p state missing for {dataset}")
        detail_columns = [column for column in canonical["p2p"].columns if column.startswith("velocity_") or column.startswith("D_")]
        velocity_details.append(canonical["p2p"].loc[:, ["dataset", "window_id", "window_index", "center_cycle_effective", "center_cycle_actual", *detail_columns]])
    consensus = pd.concat(consensus_frames, ignore_index=True)
    variants = {name: pd.concat(parts, ignore_index=True) for name, parts in variant_consensus.items()}
    stability = trajectory_stability(all_records, config)
    # State grid and consensus are frozen before metadata is opened for post-hoc analysis.
    metadata = json.loads(Path(config.metadata_path).read_text(encoding="utf-8"))
    morphology, morphology_diagnostic = morphology_interval_alignment(consensus.loc[consensus.dataset.eq("Exp1")].copy(), metadata, config)
    patterns = exp_pattern_comparison(consensus, config)
    ry_audit = ry_group_audit(frames, extended_canonical, variants, config)
    provenance = input_provenance(config)
    causality = _prefix_causality(frames["Exp1"], config)
    tests = _test_status(paths["diagnostics"] / "full_pytest_v44.xml") if run_tests else {"status": "NOT_RUN", "summary": "--skip-tests requested"}
    metadata_isolation = {
        "status": "PASS", "metadata_read_after_state_computation": True,
        "state_input_columns": ["window identifiers", "feature_name", "z_value"],
        "prohibited_uses": metadata.get("analysis_boundary", {}).get("prohibited_uses", []),
    }
    consensus.to_csv(paths["results"] / "consensus_state_trajectories_v44.csv", index=False)
    pd.concat(velocity_details, ignore_index=True).to_csv(paths["results"] / "canonical_velocity_vectors_v44.csv", index=False)
    stability.to_csv(paths["results"] / "trajectory_stability_v44.csv", index=False)
    morphology.to_csv(paths["results"] / "morphology_interval_alignment_v44.csv", index=False)
    patterns.to_csv(paths["results"] / "exp1_exp2_pattern_comparison_v44.csv", index=False)
    ry_audit.to_csv(paths["results"] / "ry_group_audit_v44.csv", index=False)
    pd.DataFrame([provenance]).to_csv(paths["results"] / "input_provenance_v44.csv", index=False)
    consensus_trajectory_figure(consensus, paths["figures"] / "consensus_trajectories_v44.png")
    morphology_interval_figure(morphology, paths["figures"] / "morphology_interval_alignment_v44.png")
    configuration_coverage = (consensus.groupby("dataset", as_index=False).agg(rows=("window_index", "size"), min_effective_configuration_count=("effective_configuration_count", "min"), max_effective_configuration_count=("effective_configuration_count", "max"), first_effective_cycle=("center_cycle_effective", "min"), last_effective_cycle=("center_cycle_effective", "max")))
    _write_json(paths["diagnostics"] / "input_provenance_v44.json", provenance)
    _write_json(paths["diagnostics"] / "prefix_causality_v44.json", causality)
    _write_json(paths["diagnostics"] / "morphology_posthoc_v44.json", morphology_diagnostic)
    _write_json(paths["diagnostics"] / "metadata_isolation_v44.json", metadata_isolation)
    _write_json(paths["diagnostics"] / "tests_v44.json", tests)
    _write_json(paths["diagnostics"] / "configuration_coverage_v44.json", configuration_coverage.to_dict(orient="records"))
    _write_json(paths["diagnostics"] / "run_manifest_v44.json", {
        "status": "PASS" if all(item.get("status") == "PASS" for item in (provenance, causality, tests)) else "FAIL",
        "code_sha256": _code_hash(), "input_sha256": _sha256(Path(config.z_table_path)),
        "metadata_sha256": _sha256(Path(config.metadata_path),), "configurations": len(all_records),
        "extended_ry_configurations": 12, "state_time_axis": "effective_cycle", "actual_cycle_use": "plotting_and_posthoc_only",
    })
    write_report(paths["reports"] / "continuous_state_v44_report.md", config_payload=config.jsonable(), provenance=provenance, stability=stability,
                 alignment=morphology, patterns=patterns, ry_audit=ry_audit, test_status=tests, causal_status=causality)
    return {"paths": paths, "tests": tests, "causality": causality, "provenance": provenance}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs_continuous_state_v44")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    outcome = run(ContinuousStateV44Config(output_dir=args.output_dir), run_tests=not args.skip_tests)
    print(json.dumps({"output_dir": str(outcome["paths"]["root"]), "tests": outcome["tests"]["status"], "causality": outcome["causality"]["status"], "provenance": outcome["provenance"]["status"]}, ensure_ascii=False))
    return 0 if outcome["tests"]["status"] == "PASS" and outcome["causality"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
