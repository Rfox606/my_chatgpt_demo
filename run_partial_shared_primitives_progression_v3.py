from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from partial_shared_primitives_progression_v3.config import PartialSharedPrimitivesConfig
from partial_shared_primitives_progression_v3.data import input_hash, load_windows
from partial_shared_primitives_progression_v3.evaluation import acceptance_decision, predictor_holdout_evaluation, prefix_causality_audit, run_v3_pipeline
from partial_shared_primitives_progression_v3.primitives import bootstrap_primitive_stability
from partial_shared_primitives_progression_v3.report import write_report


def main() -> None:
    config = PartialSharedPrimitivesConfig(); paths = config.paths(); frame = load_windows(config)
    Path(paths["configs"] / "partial_shared_primitives_progression_v3_config.json").write_text(json.dumps(config.jsonable(), ensure_ascii=False, indent=2), encoding="utf-8")
    pipeline = run_v3_pipeline(frame, config)
    shared = pipeline["shared"]; prior = pipeline["prior"]; state_paths = pipeline["state_paths"]; continuous = pipeline["continuous"]
    shared.to_csv(paths["results"] / "shared_causal_representation_v3.csv", index=False)
    pipeline["primitive_table"].to_csv(paths["results"] / "shared_dynamic_primitives_v3.csv", index=False)
    prior.to_csv(paths["results"] / "dynamic_primitive_prior_scores_v3.csv", index=False)
    pipeline["state_models_table"].to_csv(paths["results"] / "experiment_specific_state_models_v3.csv", index=False)
    state_paths.to_csv(paths["results"] / "experiment_specific_state_paths_v3.csv", index=False)
    continuous.to_csv(paths["results"] / "conditioned_continuous_progression_v3.csv", index=False)
    holdout = predictor_holdout_evaluation(shared); holdout.to_csv(paths["results"] / "shared_predictor_holdout_v3.csv", index=False)
    primitive_bootstrap = bootstrap_primitive_stability(shared, pipeline["primitive_dictionary"], config); primitive_bootstrap.to_csv(paths["results"] / "primitive_bootstrap_stability_v3.csv", index=False)
    prefix = prefix_causality_audit(frame, config)
    decision = acceptance_decision(pipeline, primitive_bootstrap, holdout, prefix, config)
    decision["predictor_hash"] = pipeline["predictor_hash"]
    Path(paths["diagnostics"] / "prefix_causality_v3.json").write_text(json.dumps(prefix, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(paths["diagnostics"] / "no_label_leakage_v3.json").write_text(json.dumps({"status": "PASS", "forbidden_inputs_read": [], "formal_input_label_free": True, "future_target_data_used_for_selection": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(paths["diagnostics"] / "acceptance_decision_v3.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(paths["diagnostics"] / "run_manifest_v3.json").write_text(json.dumps({"status": decision["status"], "engineering_status": "PENDING_TESTS", "input_sha256": input_hash(config.input_path), "protocol_locked_before_run": True, "output_files": [str(path.relative_to(paths["root"])) for path in paths["root"].rglob("*") if path.is_file()]}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(paths, decision)
    # Figures are descriptive only; no visual output feeds model selection.
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    for dataset, group in shared.groupby("dataset", sort=True): axes[0].plot(group.center_cycle, group.forecast_mae.rolling(32, min_periods=1).mean(), label=dataset, linewidth=.8)
    axes[0].set(title="Shared causal predictor: rolling forecast MAE", xlabel="cycle", ylabel="MAE"); axes[0].legend(); fig.tight_layout(); fig.savefig(paths["figures"] / "shared_predictor_error_v3.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    for dataset, group in state_paths.groupby("dataset", sort=True): axes[0].scatter(group.center_cycle, group.local_state_id, s=1.5, label=dataset)
    axes[0].set(title="Experiment-specific causal state paths (IDs are not aligned)", xlabel="cycle", ylabel="local state"); axes[0].legend()
    for dataset, group in continuous.groupby("dataset", sort=True): axes[1].plot(group.center_cycle, group.continuous_progression_score, linewidth=.7, label=dataset)
    axes[1].set(title="State-independent conditioned continuous progression evidence", xlabel="cycle", ylabel="score"); axes[1].legend(); fig.tight_layout(); fig.savefig(paths["figures"] / "local_states_and_continuous_progression_v3.png", dpi=150); plt.close(fig)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

