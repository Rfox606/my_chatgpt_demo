from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from continuous_state_v1.baseline_distance import fit_baseline_distance, score_baseline_distance
from continuous_state_v1.candidate_selection import select_physical_validation_candidates
from continuous_state_v1.config import ContinuousStateV1Config
from continuous_state_v1.data import assert_label_free, load_window_table
from continuous_state_v1.diagnostics import (
    add_target_support_scores,
    baseline_stability,
    no_stage_leakage_check,
    source_support_table,
    target_temporal_concordance,
    write_json,
)
from continuous_state_v1.guards import add_restart_guard
from continuous_state_v1.pair_sampling import pair_split_check, split_source_windows
from continuous_state_v1.plotting import make_figures
from continuous_state_v1.rank_head import (
    coefficient_table,
    fit_final_rank_model,
    pair_auc_by_gap,
    select_rank_C,
)
from continuous_state_v1.target_anchor import score_awr, source_awr_scale


def _add_contributions(frame: pd.DataFrame, config: ContinuousStateV1Config, weight: np.ndarray) -> pd.DataFrame:
    assert_label_free(frame)
    result = frame.copy()
    for feature, value in zip(config.stable_plus_features, weight, strict=True):
        result[f"contrib_{feature}"] = float(value) * result[feature].to_numpy(float)
    return result


def _score_dataset(
    base: pd.DataFrame,
    role: str,
    direction_id: str,
    source_dataset: str,
    target_dataset: str,
    normalized_weight: np.ndarray,
    source_scale: float,
    config: ContinuousStateV1Config,
    source_support: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Score a frame using only source-fixed weights and its own initial baseline."""
    assert_label_free(base)
    awr_scored, anchor = score_awr(base, normalized_weight, source_scale, config)
    distance_model = fit_baseline_distance(awr_scored, config)
    scored = score_baseline_distance(awr_scored, distance_model, config)
    scored["direction_id"] = direction_id
    scored["source_dataset"] = source_dataset
    scored["target_dataset"] = target_dataset
    scored["dataset_role"] = role
    scored = _add_contributions(scored, config, normalized_weight)
    if source_support is not None:
        scored, _ = add_target_support_scores(scored, source_support, config)
    else:
        # Cross-domain support is only meaningful for a target window.  Source rows carry
        # explicit zero values so the unified score table remains numeric and non-ambiguous.
        scored["oos_feature_count"] = 0
        scored["oos_fraction"] = 0.0
        scored["max_abs_target_z"] = scored.loc[:, list(config.stable_plus_features)].abs().max(axis=1)
    return scored, anchor.baseline_mask, anchor.anchor_method


def _prefix_causality_check(
    target: pd.DataFrame,
    full_scores: pd.DataFrame,
    context: dict[str, object],
    config: ContinuousStateV1Config,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    comparison_columns = ["AWR_raw", "AWR_rel", "AWR_scaled", "BD", "BD_diag", "oos_fraction"]
    for cutoff in (500, 1000, 2000):
        prefix = target.loc[target["center_cycle"].to_numpy(float) <= cutoff].copy()
        if prefix.empty:
            continue
        prefix_scores, _, _ = _score_dataset(
            prefix,
            "target",
            str(context["direction_id"]),
            str(context["source_dataset"]),
            str(context["target_dataset"]),
            np.asarray(context["weight"]),
            float(context["source_scale"]),
            config,
            context["source_support"],
        )
        reference = full_scores.loc[full_scores["center_cycle"].to_numpy(float) <= cutoff]
        joined = prefix_scores.merge(reference[["window_index", *comparison_columns]], on="window_index", suffixes=("_prefix", "_full"), validate="one_to_one")
        maximum = 0.0
        for column in comparison_columns:
            maximum = max(
                maximum,
                float(np.max(np.abs(joined[f"{column}_prefix"].to_numpy(float) - joined[f"{column}_full"].to_numpy(float)))),
            )
        checks.append({"prefix_center_cycle": cutoff, "window_count": int(len(joined)), "max_abs_difference": maximum, "pass": maximum < 1e-10})
    # A full-prefix replay is a useful deterministic control and is still free of target labels.
    replay, _, _ = _score_dataset(
        target,
        "target",
        str(context["direction_id"]),
        str(context["source_dataset"]),
        str(context["target_dataset"]),
        np.asarray(context["weight"]),
        float(context["source_scale"]),
        config,
        context["source_support"],
    )
    joined = replay.merge(full_scores[["window_index", *comparison_columns]], on="window_index", suffixes=("_prefix", "_full"), validate="one_to_one")
    maximum = max(
        float(np.max(np.abs(joined[f"{column}_prefix"].to_numpy(float) - joined[f"{column}_full"].to_numpy(float))))
        for column in comparison_columns
    )
    checks.append({"prefix_center_cycle": "all", "window_count": int(len(joined)), "max_abs_difference": maximum, "pass": maximum < 1e-10})
    return {"status": "PASS" if all(item["pass"] for item in checks) else "FAIL", "checks": checks}


def _main_score_columns(config: ContinuousStateV1Config) -> list[str]:
    return [
        "direction_id", "source_dataset", "target_dataset", "dataset_role", "window_id", "window_index",
        "start_cycle", "end_cycle", "center_cycle", "baseline_window", "is_restart_guard", "AWR_raw",
        "AWR_rel", "AWR_scaled", "BD", "BD_diag", "bd_method", "oos_feature_count", "oos_fraction",
        "max_abs_target_z", *[f"contrib_{feature}" for feature in config.stable_plus_features],
    ]


def _report(
    source_summary: pd.DataFrame,
    temporal: pd.DataFrame,
    stability: pd.DataFrame,
    coefficients: pd.DataFrame,
    oos: pd.DataFrame,
    candidates: pd.DataFrame,
    implementation: dict[str, object],
    scientific: dict[str, object],
) -> str:
    auc = source_summary.loc[:, ["direction_id", "source_pair_auc", "selected_C"]].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    concordance = temporal.loc[:, ["direction_id", "target_long_gap_concordance", "spearman_AWR_cycle", "spearman_BD_cycle"]].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    weights = coefficients.loc[:, ["direction_id", "feature_name", "normalized_weight", "rank"]].to_string(index=False, float_format=lambda value: f"{value:.5f}")
    signs = coefficients.pivot(index="feature_name", columns="direction_id", values="normalized_weight")
    if signs.shape[1] == 2:
        agreement = signs.index[(np.sign(signs.iloc[:, 0]) == np.sign(signs.iloc[:, 1]))].tolist()
    else:
        agreement = []
    dominant_oos = (
        oos.groupby("feature_name", as_index=False)["oos_fraction"].mean().sort_values("oos_fraction", ascending=False).head(3)
    )
    candidate_text = "No candidate rows were produced."
    if not candidates.empty:
        candidate_text = candidates.loc[:, ["direction_id", "candidate_type", "center_cycle", "AWR_rel", "BD", "oos_fraction"]].head(30).to_string(index=False, float_format=lambda value: f"{value:.4f}")
    stable = stability.loc[:, ["direction_id", "baseline_AWR_rel_median", "baseline_AWR_rel_IQR", "baseline_BD_median", "baseline_BD_p95"]].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    source_fail = source_summary.loc[source_summary.source_pair_auc < 0.60, "direction_id"].tolist()
    target_fail = temporal.loc[temporal.target_long_gap_concordance < 0.55, "direction_id"].tolist()
    failure_reason = (
        f"Source directional evidence below the fixed criterion: {source_fail}; target long-gap time consistency below the fixed criterion: {target_fail}."
        if scientific["status"] == "FAIL"
        else "Both fixed source and target time-consistency criteria are met; this supports, but does not prove, a transferable continuous evolution direction."
    )
    return f"""# Continuous State Monitoring v1

Implementation acceptance: **{implementation['status']}**. Scientific acceptance: **{scientific['status']}**.

## Required answers

1. Source validation pair AUC and selected C:

```text
{auc}
```

2. The final ten feature weights are listed below. They are normalised L1 rank weights, not calibrated risks.

```text
{weights}
```

3. Features with the same learned direction in both transfer directions: {", ".join(agreement) if agreement else "none"}.
4. Target long-gap pair concordance is:

```text
{concordance}
```

The fixed target criterion is 0.55.
5. Target initial-baseline stability is:

```text
{stable}
```

The baseline AWR relative median is an implementation check and is expected to be approximately zero.
6. Mean target out-of-support fractions are greatest for:

```text
{dominant_oos.to_string(index=False, float_format=lambda value: f"{value:.4f}")}
```

High out-of-support fractions reduce confidence in source-prior transfer; they do not establish a high-wear conclusion.
7. The AWR/BD disagreement and increase candidates are the rows listed below; their cycles identify the follow-up inspection locations.

```text
{candidate_text}
```

8. The highest-priority physical checks are candidates marked `high_AWR_high_BD`, `high_AWR_low_BD`, `low_AWR_high_BD`, and the two increase types. These are diagnostic-only offline selections, not online alarms.
9. {failure_reason}
10. If this result is not scientifically accepted, the fixed diagnostics distinguish source-direction instability, cross-domain reversal in target concordance, and support loss. The current outcome is: {failure_reason}
11. AWR is a continuous, source-learned temporal ranking score. It is **not** a Stage5 probability, a wear percentage, or a failure probability. BD is only distance from that experiment's initial feature-state baseline.
12. No Stage1–Stage5 labels were used for training, C selection, scoring, candidate selection, or output generation in this experiment.

## Interpretation boundary

The two primary outputs are AWR_rel and BD. They are intentionally complementary: a high AWR can coexist with low initial-state distance, and a high BD can occur without a high learned temporal-direction score. Any causal or physical wear conclusion requires the planned surface-morphology, debris, or experimental-observation correspondence.
"""


def main() -> None:
    config = ContinuousStateV1Config()
    paths = config.paths()
    (paths["configs"] / "continuous_state_v1_config.json").write_text(
        json.dumps(config.jsonable(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    all_windows = add_restart_guard(load_window_table(config), config)
    assert_label_free(all_windows)

    score_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    source_summaries: list[dict[str, object]] = []
    source_gap_frames: list[pd.DataFrame] = []
    temporal_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    support_frames: list[pd.DataFrame] = []
    target_oos_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    prefix_audits: dict[str, object] = {}
    pair_audits: dict[str, object] = {}
    baseline_audits: dict[str, object] = {}

    for direction_id, source_dataset, target_dataset in (
        ("Exp1_to_Exp2", "Exp1", "Exp2"),
        ("Exp2_to_Exp1", "Exp2", "Exp1"),
    ):
        source = all_windows.loc[all_windows.dataset.eq(source_dataset)].copy()
        target = all_windows.loc[all_windows.dataset.eq(target_dataset)].copy()
        assert_label_free(source)
        assert_label_free(target)
        source_train, source_validation, source_gap = split_source_windows(source, config)
        selected_C, grid, train_pairs, validation_pairs = select_rank_C(source_train, source_validation, config)
        rank_model, all_source_pairs, validation_metrics = fit_final_rank_model(
            source, validation_pairs, selected_C, config
        )
        source_scale = source_awr_scale(source, rank_model.normalized_weight, config)
        support = source_support_table(source, direction_id, config)
        source_scored, source_baseline_mask, _ = _score_dataset(
            source,
            "source",
            direction_id,
            source_dataset,
            target_dataset,
            rank_model.normalized_weight,
            source_scale,
            config,
        )
        target_scored, target_baseline_mask, target_anchor_method = _score_dataset(
            target,
            "target",
            direction_id,
            source_dataset,
            target_dataset,
            rank_model.normalized_weight,
            source_scale,
            config,
            support,
        )
        _, target_oos = add_target_support_scores(target_scored, support, config)
        context = {
            "direction_id": direction_id,
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "weight": rank_model.normalized_weight,
            "source_scale": source_scale,
            "source_support": support,
        }
        prefix_audits[direction_id] = _prefix_causality_check(target, target_scored, context, config)
        pair_audits[direction_id] = pair_split_check(
            train_pairs,
            validation_pairs,
            config,
            train_window_ids=set(source_train["window_id"]),
            validation_window_ids=set(source_validation["window_id"]),
        )
        baseline_audits[direction_id] = {
            "status": "PASS"
            if abs(float(np.median(source_scored.loc[source_baseline_mask, "AWR_rel"]))) < 1e-10
            and abs(float(np.median(target_scored.loc[target_baseline_mask, "AWR_rel"]))) < 1e-10
            else "FAIL",
            "anchor_method": target_anchor_method,
            "source_baseline_window_count": int(source_baseline_mask.sum()),
            "target_baseline_window_count": int(target_baseline_mask.sum()),
            "source_baseline_AWR_rel_median": float(np.median(source_scored.loc[source_baseline_mask, "AWR_rel"])),
            "target_baseline_AWR_rel_median": float(np.median(target_scored.loc[target_baseline_mask, "AWR_rel"])),
        }
        coefficient = coefficient_table(rank_model, config.stable_plus_features)
        coefficient["direction_id"] = direction_id
        coefficient["source_dataset"] = source_dataset
        coefficient["target_dataset"] = target_dataset
        coefficient["selected_C"] = selected_C
        coefficient["orientation_flipped"] = int(rank_model.orientation_flipped)
        coefficient_frames.append(coefficient)
        gap_summary = pair_auc_by_gap(rank_model, validation_pairs)
        gap_summary["direction_id"] = direction_id
        gap_summary["source_dataset"] = source_dataset
        gap_summary["target_dataset"] = target_dataset
        source_gap_frames.append(gap_summary)
        source_summaries.append(
            {
                "direction_id": direction_id,
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "source_pair_auc": validation_metrics["source_pair_auc"],
                "source_pair_accuracy": validation_metrics["source_pair_accuracy"],
                "source_pair_logloss": validation_metrics["source_pair_logloss"],
                "selected_C": selected_C,
                "pair_count_train": train_pairs.pair_count,
                "pair_count_validation": validation_pairs.pair_count,
                "pair_count_final_source": all_source_pairs.pair_count,
                "source_train_window_count": len(source_train),
                "source_gap_window_count": len(source_gap),
                "source_validation_window_count": len(source_validation),
                "source_scale": source_scale,
                "orientation_flipped": int(rank_model.orientation_flipped),
            }
        )
        temporal_rows.append(target_temporal_concordance(target_scored, config, config.pair_random_seed + 10))
        stability_rows.append(baseline_stability(target_scored, target_baseline_mask))
        candidate_frames.append(select_physical_validation_candidates(target_scored, config))
        score_frames.extend([source_scored, target_scored])
        support_frames.append(support)
        target_oos_frames.append(target_oos)

    scores = pd.concat(score_frames, ignore_index=True).loc[:, _main_score_columns(config)]
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    source_summary = pd.DataFrame(source_summaries)
    source_gaps = pd.concat(source_gap_frames, ignore_index=True)
    temporal = pd.DataFrame(temporal_rows)
    stability = pd.DataFrame(stability_rows)
    support = pd.concat(support_frames, ignore_index=True)
    target_oos = pd.concat(target_oos_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True) if any(not frame.empty for frame in candidate_frames) else pd.DataFrame()
    leakage = no_stage_leakage_check(scores, coefficients, source_summary, source_gaps, temporal, stability, support, target_oos, candidates)

    scores.to_csv(paths["results"] / "continuous_window_scores_v1.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(paths["results"] / "pairwise_rank_coefficients.csv", index=False, encoding="utf-8-sig")
    source_summary.to_csv(paths["results"] / "source_pair_validation_summary.csv", index=False, encoding="utf-8-sig")
    source_gaps.to_csv(paths["results"] / "source_pair_gap_summary.csv", index=False, encoding="utf-8-sig")
    temporal.to_csv(paths["results"] / "target_temporal_concordance_summary.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(paths["results"] / "target_baseline_stability.csv", index=False, encoding="utf-8-sig")
    support.to_csv(paths["results"] / "source_feature_support.csv", index=False, encoding="utf-8-sig")
    target_oos.to_csv(paths["results"] / "target_oos_summary.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(paths["results"] / "physical_validation_candidates.csv", index=False, encoding="utf-8-sig")

    prefix_ok = all(audit["status"] == "PASS" for audit in prefix_audits.values())
    pairs_ok = all(audit["status"] == "PASS" for audit in pair_audits.values())
    anchors_ok = all(audit["status"] == "PASS" for audit in baseline_audits.values())
    test_paths = sorted(str(path) for path in Path("tests").glob("test_csv1_*.py"))
    pytest_run = subprocess.run([sys.executable, "-m", "pytest", "-q", *test_paths], capture_output=True, text=True)
    (paths["diagnostics"] / "pytest_summary.txt").write_text(
        (pytest_run.stdout or "") + (pytest_run.stderr or ""), encoding="utf-8"
    )
    implementation = {
        "status": "PASS" if pytest_run.returncode == 0 and leakage["status"] == "PASS" and prefix_ok and pairs_ok and anchors_ok else "FAIL",
        "pytest_exit_code": pytest_run.returncode,
        "both_directions_completed": sorted(temporal.direction_id.tolist()),
        "main_score_columns_are_label_free": leakage["status"] == "PASS",
    }
    scientific = {
        "status": "PASS"
        if bool((source_summary.source_pair_auc >= config.scientific_source_pair_auc_min).all())
        and bool((temporal.target_long_gap_concordance >= config.scientific_target_concordance_min).all())
        else "FAIL",
        "source_pair_auc_minimum": config.scientific_source_pair_auc_min,
        "target_long_gap_concordance_minimum": config.scientific_target_concordance_min,
    }
    direction_summary = source_summary.merge(temporal, on=["direction_id", "source_dataset", "target_dataset"], how="left").merge(
        stability, on=["direction_id", "source_dataset", "target_dataset"], how="left"
    )
    direction_summary["implementation_acceptance"] = implementation["status"]
    direction_summary["scientific_acceptance"] = scientific["status"]
    direction_summary.to_csv(paths["results"] / "direction_summary.csv", index=False, encoding="utf-8-sig")
    write_json(paths["diagnostics"] / "implementation_acceptance.json", implementation)
    write_json(paths["diagnostics"] / "scientific_acceptance.json", scientific)
    write_json(paths["diagnostics"] / "no_stage_leakage_check.json", leakage)
    write_json(paths["diagnostics"] / "prefix_causality_check.json", {"status": "PASS" if prefix_ok else "FAIL", "directions": prefix_audits})
    write_json(paths["diagnostics"] / "pair_split_check.json", {"status": "PASS" if pairs_ok else "FAIL", "directions": pair_audits})
    write_json(paths["diagnostics"] / "baseline_anchor_check.json", {"status": "PASS" if anchors_ok else "FAIL", "directions": baseline_audits})
    make_figures(scores, coefficients, source_gaps, candidates, paths["figures"])
    (paths["reports"] / "continuous_state_v1_report.md").write_text(
        _report(source_summary, temporal, stability, coefficients, target_oos, candidates, implementation, scientific),
        encoding="utf-8",
    )
    print(f"Continuous State Monitoring v1 complete: implementation={implementation['status']}, scientific={scientific['status']}")


if __name__ == "__main__":
    main()
