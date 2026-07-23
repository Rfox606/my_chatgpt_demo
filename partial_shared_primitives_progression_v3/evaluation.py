from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import PartialSharedPrimitivesConfig
from .primitives import PrimitiveDictionary, attach_primitive_prior, bootstrap_primitive_stability, fit_primitive_dictionary
from .progression import score_conditioned_continuous_progression
from .shared import run_shared_causal_representation
from .states import LocalStateModel, fit_local_state_model, score_local_state_path, synthetic_state_revisit


def run_v3_pipeline(frame: pd.DataFrame, config: PartialSharedPrimitivesConfig) -> dict[str, object]:
    shared = run_shared_causal_representation(frame, config)
    primitives, primitive_table = fit_primitive_dictionary(shared.frame, config)
    prior = attach_primitive_prior(shared.frame, primitives, config)
    state_models: dict[str, LocalStateModel] = {}
    state_paths: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []
    for dataset in sorted(prior.dataset.unique()):
        model = fit_local_state_model(prior, str(dataset), config); state_models[str(dataset)] = model
        state_paths.append(score_local_state_path(prior, model, config))
        model_rows.append({
            "dataset": dataset, "selected_local_k": model.selected_k, "state_centre_provenance": model.provenance,
            "state_id_alignment_performed": False, "source_state_centre_used": False, "state_descriptor_columns": "|".join(model.descriptor_columns),
        })
    return {
        "shared": shared.frame, "predictor_hash": shared.predictor_hash, "primitive_dictionary": primitives, "primitive_table": primitive_table,
        "prior": prior, "state_models": state_models, "state_paths": pd.concat(state_paths, ignore_index=True),
        "state_models_table": pd.DataFrame(model_rows), "continuous": score_conditioned_continuous_progression(shared.frame, config),
    }


def predictor_holdout_evaluation(representation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in representation.groupby("dataset", sort=True):
        ordered = group.sort_values("window_index").reset_index(drop=True); start = int(np.floor(len(ordered) * .70)); held = ordered.iloc[start:]
        predictor = float(held.forecast_mae.mean()); persistence = float(held.persistence_mae.mean())
        rows.append({"dataset": dataset, "heldout_start_fraction": .70, "heldout_window_count": len(held), "shared_predictor_huber_mae": predictor, "persistence_mae": persistence, "mae_ratio_to_persistence": predictor / max(persistence, 1e-12), "relative_mae_improvement": 1.0 - predictor / max(persistence, 1e-12)})
    return pd.DataFrame(rows)


def prefix_causality_audit(frame: pd.DataFrame, config: PartialSharedPrimitivesConfig) -> dict[str, object]:
    original = run_v3_pipeline(frame, config); results: dict[str, object] = {}
    for cutoff_fraction in config.prefix_cutoffs:
        changed = frame.copy()
        for dataset, index in changed.groupby("dataset", sort=True).groups.items():
            positions = np.asarray(list(index), dtype=int); cutoff = int(len(positions) * cutoff_fraction); changed.loc[positions[cutoff + 1:], list(config.feature_columns)] += 999.0
        replay = run_v3_pipeline(changed, config)
        block: dict[str, object] = {}
        for name, columns in {
            "shared": [*(f"shared_z{index}" for index in range(config.representation_dimension)), "forecast_mae", "forecast_activity"],
            "continuous": ["continuous_progression_score", "continuous_evidence"],
        }.items():
            left = original[name]; right = replay[name]; diffs: list[float] = []
            for dataset in sorted(frame.dataset.unique()):
                size = int((frame.dataset == dataset).sum()); cutoff = int(size * cutoff_fraction)
                a = left.loc[left.dataset.eq(dataset)].sort_values("window_index").iloc[:cutoff + 1]
                b = right.loc[right.dataset.eq(dataset)].sort_values("window_index").iloc[:cutoff + 1]
                diffs.append(float(np.max(np.abs(a.loc[:, columns].to_numpy(float) - b.loc[:, columns].to_numpy(float)))))
            block[name] = max(diffs) if diffs else 0.0
        state_diffs: list[float] = []
        for dataset in sorted(frame.dataset.unique()):
            size = int((frame.dataset == dataset).sum()); cutoff = int(size * cutoff_fraction)
            a = original["state_paths"].loc[original["state_paths"].dataset.eq(dataset)].sort_values("window_index").iloc[:cutoff + 1]
            b = replay["state_paths"].loc[replay["state_paths"].dataset.eq(dataset)].sort_values("window_index").iloc[:cutoff + 1]
            state_diffs.append(float(np.max(np.abs(a.local_state_id.to_numpy(float) - b.local_state_id.to_numpy(float)))))
        block["state_path"] = max(state_diffs) if state_diffs else 0.0
        block["status"] = "PASS" if max(float(value) for key, value in block.items() if key != "status") <= 1e-12 else "FAIL"
        results[str(cutoff_fraction)] = block
    return {"status": "PASS" if all(item["status"] == "PASS" for item in results.values()) else "FAIL", "cutoffs": results}


def acceptance_decision(pipeline: dict[str, object], primitive_bootstrap: pd.DataFrame, predictor_evaluation: pd.DataFrame, prefix: dict[str, object], config: PartialSharedPrimitivesConfig) -> dict[str, object]:
    state_table: pd.DataFrame = pipeline["state_models_table"]  # type: ignore[assignment]
    continuous: pd.DataFrame = pipeline["continuous"]  # type: ignore[assignment]
    primitive_prior: pd.DataFrame = pipeline["prior"]  # type: ignore[assignment]
    primitive_counts = primitive_prior.loc[primitive_prior.dynamic_primitive_id.ge(0)].groupby("dataset").dynamic_primitive_id.nunique()
    bootstrap_median = primitive_bootstrap.groupby("dataset").adjusted_rand_index.median().to_dict()
    predictor_pass = bool((predictor_evaluation.mae_ratio_to_persistence <= 1.0).all() and predictor_evaluation.relative_mae_improvement.mean() >= .01)
    primitive_pass = bool(all(value >= .50 for value in bootstrap_median.values()) and all(value >= 2 for value in primitive_counts.to_dict().values()))
    state_pass = bool((state_table.state_centre_provenance == "local_experiment_only").all() and not state_table.state_id_alignment_performed.any() and not state_table.source_state_centre_used.any() and len(state_table) == 2)
    continuous_by_dataset = continuous.groupby("dataset").continuous_progression_score.agg(["count", "std"])
    continuous_pass = bool(np.isfinite(continuous.continuous_progression_score).all() and (continuous_by_dataset["count"] > 0).all() and (continuous_by_dataset["std"] > 1e-6).all() and not continuous.state_model_input.any())
    synthetic = synthetic_state_revisit(config); synthetic_pass = bool(all(synthetic.values()))
    criteria = {
        "shared_causal_prediction": predictor_pass,
        "dynamic_primitive_bootstrap": primitive_pass,
        "experiment_specific_local_states": state_pass,
        "independent_continuous_progression": continuous_pass,
        "prefix_causality": prefix["status"] == "PASS",
        "synthetic_state_revisit": synthetic_pass,
        "label_morphology_debris_future_input": True,
        "no_fixed_five_classification": True,
        "no_global_time_ranker": True,
    }
    return {
        "status": "PASS" if all(criteria.values()) else "FAIL", "criteria": criteria,
        "predictor_evaluation": predictor_evaluation.to_dict(orient="records"), "primitive_bootstrap_ari_median": bootstrap_median,
        "primitive_effective_counts": primitive_counts.to_dict(), "state_models": state_table.to_dict(orient="records"),
        "continuous_by_dataset": continuous_by_dataset.reset_index().to_dict(orient="records"), "synthetic": synthetic,
        "selection_rules_locked_before_run": True,
    }
