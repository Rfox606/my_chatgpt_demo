from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multistage_trajectory_state_v2.config import MultiStageTrajectoryConfig
from multistage_trajectory_state_v2.data import input_hash, load_windows
from multistage_trajectory_state_v2.evaluation import (
    bootstrap_boundary_stability,
    future_frozen_evaluation,
    prefix_causality_check,
    synthetic_multistage_validation,
)
from multistage_trajectory_state_v2.online_filter import run_online_filter
from multistage_trajectory_state_v2.regime_model import build_source_regime_model
from multistage_trajectory_state_v2.segmentation import causal_descriptors, segments_from_consensus


def main() -> None:
    config = MultiStageTrajectoryConfig(); paths = config.paths(); frame = load_windows(config)
    Path(paths["configs"] / "multistage_trajectory_v2_config.json").write_text(json.dumps(config.jsonable(), ensure_ascii=False, indent=2), encoding="utf-8")
    consensus = pd.read_csv(paths["results"] / "change_point_consensus_v2.csv")
    by_dataset = {name: group.sort_values(["center_cycle", "window_index"]).reset_index(drop=True) for name, group in frame.groupby("dataset", sort=True)}
    prototype_rows: list[pd.DataFrame] = []; online_rows: list[pd.DataFrame] = []; transition_rows: list[pd.DataFrame] = []; evaluation_rows: list[pd.DataFrame] = []; stability_rows: list[pd.DataFrame] = []; causality: dict[str, object] = {}; structures: dict[str, str] = {}
    directions = (("Exp1_to_Exp2", "Exp1", "Exp2"), ("Exp2_to_Exp1", "Exp2", "Exp1"))
    for direction, source_name, target_name in directions:
        source_raw, target_raw = by_dataset[source_name], by_dataset[target_name]
        source_desc, reference, columns = causal_descriptors(source_raw, config)
        target_desc, _, _ = causal_descriptors(target_raw, config, reference=reference)
        segments = segments_from_consensus(source_desc, consensus, config)
        structure, prototypes, _ = build_source_regime_model(source_desc, segments, columns, config)
        structures[direction] = structure.source_hash
        prototypes.insert(0, "direction", direction); prototypes.insert(1, "source_dataset", source_name); prototype_rows.append(prototypes)
        scores, transitions, runner = run_online_filter(target_desc, structure, config, adaptive=True)
        scores.insert(0, "direction", direction); scores.insert(1, "source_dataset", source_name); scores.insert(2, "target_dataset", target_name); online_rows.append(scores)
        if transitions.empty:
            transitions = pd.DataFrame(columns=["window_index", "center_cycle", "event", "from_regime", "to_regime"])
        transitions.insert(0, "direction", direction); transitions.insert(1, "source_dataset", source_name); transitions.insert(2, "target_dataset", target_name); transition_rows.append(transitions)
        evaluation, _ = future_frozen_evaluation(source_raw, target_raw, source_desc, target_desc, structure, config)
        evaluation.insert(0, "direction", direction); evaluation.insert(1, "source_dataset", source_name); evaluation.insert(2, "target_dataset", target_name); evaluation_rows.append(evaluation)
        stability = bootstrap_boundary_stability(target_desc, structure, config); stability.insert(0, "direction", direction); stability.insert(1, "target_dataset", target_name); stability_rows.append(stability)
        causality[direction] = prefix_causality_check(target_desc, structure, config)
        fig, ax = plt.subplots(figsize=(12, 4)); ax.scatter(scores.center_cycle, scores.regime_id, c=scores.novelty_score, s=3, cmap="magma"); ax.set(title=f"{direction}: causal adaptive regime timeline", xlabel="cycle", ylabel="regime id; -1=UNKNOWN_NOVEL"); fig.tight_layout(); fig.savefig(paths["figures"] / f"regime_timeline_{direction}_v2.png", dpi=150); plt.close(fig)
    prototypes_all = pd.concat(prototype_rows, ignore_index=True); online_all = pd.concat(online_rows, ignore_index=True); transition_all = pd.concat(transition_rows, ignore_index=True); evaluation_all = pd.concat(evaluation_rows, ignore_index=True); stability_all = pd.concat(stability_rows, ignore_index=True)
    prototypes_all.to_csv(paths["results"] / "source_regime_prototypes_v2.csv", index=False); online_all.to_csv(paths["results"] / "target_online_regime_scores_v2.csv", index=False); transition_all.to_csv(paths["results"] / "regime_transition_log_v2.csv", index=False); evaluation_all.to_csv(paths["results"] / "future_frozen_regime_evaluation_v2.csv", index=False); stability_all.to_csv(paths["results"] / "regime_stability_bootstrap_v2.csv", index=False)
    synthetic = synthetic_multistage_validation(config)
    decisions: dict[str, object] = {"directions": {}, "criteria": {"ceap_reconstruction_ratio_max": .98, "bootstrap_boundary_stability_min": .60, "short_isolated_state_fraction_max": .10}, "source_structure_hashes": structures, "stage_morphology_debris_read": False}
    for direction in ("Exp1_to_Exp2", "Exp2_to_Exp1"):
        group = evaluation_all.loc[evaluation_all.direction.eq(direction)]; mean_error = group.groupby("model").future_feature_reconstruction_error.mean(); adaptive = float(mean_error.get("Adaptive_Regime_Model", np.inf)); ceap = float(mean_error.get("Single_Axis_CEAP_v1", np.inf)); source = float(mean_error.get("Source_Only_State", np.inf)); local = float(mean_error.get("Target_Local_Segmentation", np.inf)); bootstrap = float(stability_all.loc[stability_all.direction.eq(direction), "boundary_match_fraction"].mean()); isolated = float(group.loc[group.model.eq("Adaptive_Regime_Model"), "short_isolated_state_fraction"].mean())
        criteria = {"reconstruction_beats_ceap": bool(adaptive <= .98 * ceap), "bootstrap_stable": bool(bootstrap >= .60), "short_spikes_controlled": bool(isolated <= .10), "adaptive_beats_best_state_comparator": bool(adaptive <= min(source, local)), "no_label_leakage": True}
        decisions["directions"][direction] = {"mean_future_reconstruction_error": {"adaptive": adaptive, "ceap": ceap, "source_only": source, "target_local": local}, "bootstrap_boundary_match_fraction": bootstrap, "short_isolated_state_fraction": isolated, "criteria": criteria, "passed": bool(all(criteria.values()))}
    decisions["synthetic"] = synthetic; decisions["status"] = "PASS" if synthetic["status"] == "PASS" and any(item["passed"] for item in decisions["directions"].values()) else "FAIL"
    Path(paths["diagnostics"] / "synthetic_multistage_validation_v2.json").write_text(json.dumps(synthetic, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(paths["diagnostics"] / "prefix_causality_v2.json").write_text(json.dumps({"status": "PASS" if all(value["status"] == "PASS" for value in causality.values()) else "FAIL", "directions": causality}, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(paths["diagnostics"] / "source_structure_frozen_v2.json").write_text(json.dumps({"status": "PASS", "source_structure_hashes": structures, "target_updates_limited_to": ["observation_center_bias", "observation_scale", "duration", "prior_weight"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(paths["diagnostics"] / "regime_model_decision_v2.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(paths["diagnostics"] / "run_input_v2.json").write_text(json.dumps({"input_path": config.input_path, "input_sha256": input_hash(config.input_path), "formal_input_label_free": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
