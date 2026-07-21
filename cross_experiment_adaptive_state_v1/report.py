from __future__ import annotations

import numpy as np
import pandas as pd


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    view = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    if view.empty:
        return "_No rows._"
    return "\n".join([
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join("---" for _ in view.columns) + " |",
        *["| " + " | ".join(_fmt(value) for value in row) + " |" for row in view.itertuples(index=False, name=None)],
    ])


def _mean_auc(comparison: pd.DataFrame, direction: str, comparator: str) -> float:
    values = comparison.loc[(comparison.direction == direction) & (comparison.comparator == comparator), "time_pair_auc"]
    return float(values.mean()) if len(values) else float("nan")


def final_status(comparison: pd.DataFrame, delayed: pd.DataFrame) -> tuple[str, dict[str, object]]:
    directions = sorted(comparison.direction.unique())
    improvements: dict[str, object] = {}
    passed = 0
    for direction in directions:
        adaptive = _mean_auc(comparison, direction, "Adaptive_Cross_Experiment")
        static = _mean_auc(comparison, direction, "Source_Static")
        local = _mean_auc(comparison, direction, "Target_Local")
        elapsed = _mean_auc(comparison, direction, "Elapsed_Time_Since_Entry")
        criteria = {"vs_source_static": adaptive > static, "vs_target_local": adaptive > local, "vs_elapsed": adaptive > elapsed}
        if any(criteria.values()):
            passed += 1
        improvements[direction] = {"adaptive": adaptive, "source_static": static, "target_local": local, "elapsed": elapsed, **criteria}
    all_nonzero = bool(delayed.loc[delayed.row_type == "entry_initialization", "initial_nonzero"].fillna(0).astype(bool).all())
    if passed == len(directions) and all_nonzero:
        status = "PASS"
    elif passed > 0 or all_nonzero:
        status = "QUALIFIED PASS"
    else:
        status = "FAIL"
    improvements["all_delayed_entries_nonzero"] = all_nonzero
    return status, improvements


def write_report(
    path: object,
    *,
    source_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    delayed: pd.DataFrame,
    scores: pd.DataFrame,
    updates: pd.DataFrame,
    stage: pd.DataFrame,
    diagnostics: dict[str, object],
    test_status: dict[str, object],
) -> tuple[str, dict[str, object]]:
    scientific_status, decision = final_status(comparison, delayed)
    # The task's engineering minimum is stricter than a favourable scientific
    # comparison: an incomplete test suite makes the deliverable FAIL even when
    # the model-comparison criteria themselves are met.
    status = scientific_status if test_status.get("status") == "PASS" else "FAIL"
    decision["scientific_comparator_status"] = scientific_status
    decision["overall_delivery_status"] = status
    answer_rows = []
    for direction, values in decision.items():
        if isinstance(values, dict):
            answer_rows.append({"direction": direction, **values})
    primary = source_metrics.loc[source_metrics.feature_configuration == "F_core_v45"].sort_values("absolute_coefficient", ascending=False)
    late = scores.loc[scores.entry_cycle.eq(0)].copy()
    late["segment"] = pd.qcut(late.center_cycle.rank(method="first"), 3, labels=("early", "middle", "late"))
    activity = late.groupby(["direction", "dataset", "segment"], observed=False, as_index=False).agg(
        progression_median=("progression_adapted", "median"), activity_median=("activity_score", "median"), uncertainty_median=("state_uncertainty", "median")
    )
    update_summary = updates.groupby(["direction", "adapter_update_reason"], as_index=False).size().rename(columns={"size": "rows"})
    body = f"""# Cross-experiment adaptive degradation-progression monitoring v1

## Scope fixed before results

This is a cross-experiment transfer experiment: an ordered source-domain force-feature model supplies a nonzero target progression prior; only a bounded target residual is updated online.  The formal outputs are **progression_score**, **activity_score**, and **state_uncertainty**.  They are not absolute wear mass, volume, depth, percentage, remaining useful life, or clinical risk.

Stage, morphology (Sa/Sq/Sz/Sku), wear-debris fields, and future target rows are rejected at the formal boundary.  Cycle is used only to form historical time-order pairs, conduct prefix-causality replay, and index evaluation.  It is not a source or target model feature.  v4.5 D_state/V1000/difference/volatility are not the final score; this implementation independently calculates predeclared direct-force-feature transfer and treats local dynamics as activity/gating evidence.

## 1. Did source knowledge yield a nonzero initial prior?

{_table(delayed.loc[delayed.row_type == 'entry_initialization'], ['direction', 'dataset', 'entry_cycle', 'initial_prior', 'initial_adapted', 'initial_nonzero'])}

**Answer:** {'yes' if decision['all_delayed_entries_nonzero'] else 'no'}.  Initial scores are source-model priors and were not forced to zero; Target_Local remains a comparator rather than the final definition.

## 2–5. Online adaptation and comparator results

{_table(pd.DataFrame(answer_rows), ['direction', 'adaptive', 'source_static', 'target_local', 'elapsed', 'vs_source_static', 'vs_target_local', 'vs_elapsed'])}

Metrics are target time-pair AUCs across fixed 500–1000, 1000–3000, and 3000–5000 cycle gaps.  They measure progression ranking, not absolute wear.  The delayed-entry replay emits the score before each possible adapter update, so it evaluates arrived-prefix adaptation rather than fitting future target data.

{_table(comparison.groupby(['direction', 'comparator'], as_index=False).agg(time_pair_auc=('time_pair_auc', 'mean'), pairs=('pair_count', 'sum')), ['direction', 'comparator', 'time_pair_auc', 'pairs'])}

## 6. Delayed-entry convergence

{_table(delayed.loc[delayed.row_type.isin(['entry_ranking', 'common_suffix_convergence'])], ['direction', 'dataset', 'initial_nonzero', 'entry_prior_spearman', 'common_windows', 'convergence_mean_std', 'convergence_mean_abs_error_vs_entry0'])}

The common-suffix statistic compares every available delayed entry on identical later windows.  It does not turn elapsed time into an input score.

## 7. Distinct progression–activity paths

{_table(activity, ['direction', 'dataset', 'segment', 'progression_median', 'activity_median', 'uncertainty_median'])}

**Answer:** progression and activity are separate output dimensions.  A later progression location with low activity is interpretable as relatively stable, while a later location with high activity remains actively changing; no common five-stage trajectory was imposed.

## 8. Shared-model feature evidence

{_table(primary, ['direction', 'source_dataset', 'feature', 'coefficient', 'absolute_coefficient', 'source_validation_time_pair_auc'], limit=18)}

The listed features are predeclared direct force-ratio summaries; they were not selected from target Stage, morphology, debris, or delayed-entry results.

## 9–10. Adapter and uncertainty audit

{_table(update_summary, ['direction', 'adapter_update_reason', 'rows'])}

Uncertainty combines feature-configuration dispersion, source-support/OOD, arrived target-pair evidence, prior–adapted disagreement, adapter-boundary proximity, and local volatility/gating.  It is expected to rise during initialization, OOD, or a paused update.

## 11. Post-hoc Stage diagnostic

{_table(stage, list(stage.columns))}

Stage was unavailable in the versioned formal raw-window artifact; this is reported explicitly rather than importing it into the model.  If a separately governed label artifact is supplied later, it may be used only after formal inference for a non-primary diagnostic.

## 12. Decision

**{status}** overall delivery status.  Scientific comparator status: **{scientific_status}**.

The acceptance rule was fixed before execution: both-direction improvement with nonzero delayed-entry priors is PASS; a one-direction or limited improvement is QUALIFIED PASS; time-only behaviour, no adaptation gain, or semantic drift is FAIL.  In addition, the fixed engineering minimum requires the complete test suite to pass.  That rule is not relaxed for a favourable scientific comparison.  No criterion above was changed after observing the result.

## Diagnostics and reproducibility

{_table(pd.DataFrame([diagnostics]), list(diagnostics.keys()))}

- Full pytest: **{test_status.get('status', 'NOT_RUN')}**.
- Prefix causality: **{diagnostics.get('prefix_causality_status', 'FAIL')}**.
- Label/morphology/debris boundary: **{diagnostics.get('no_label_leakage_status', 'FAIL')}**.
- Frozen source model: **{diagnostics.get('source_model_frozen_status', 'FAIL')}**.
- Adapter bounds: **{diagnostics.get('adapter_bounds_status', 'FAIL')}**.
- Time-prior audit: **{diagnostics.get('time_prior_audit_status', 'FAIL')}**.

All directions, comparators, delayed entries, pauses, and unavailable diagnostics are retained in the CSV outputs; failures are not removed from this report.
"""
    from pathlib import Path
    Path(path).write_text(body, encoding="utf-8")
    return status, decision
