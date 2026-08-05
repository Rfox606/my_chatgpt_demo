from __future__ import annotations

import numpy as np
import pandas as pd

from .state_engine import EVIDENCE_NAMES


def _table(frame: pd.DataFrame) -> str:
    return "```text\n" + frame.to_string(index=False) + "\n```" if not frame.empty else "No rows."


def make_report(
    states: pd.DataFrame,
    evidence: pd.DataFrame,
    ablations: pd.DataFrame,
    forecast_summary: pd.DataFrame,
    forecast_benefit: dict[str, object],
    diagnostics: dict[str, dict[str, object]],
    v4_comparison: dict[str, object],
    pytest_text: str,
) -> str:
    exp1_near = ablations.loc[(ablations.target_dataset.eq("Exp1")) & ablations.evidence_type.isin(EVIDENCE_NAMES)]
    exp1_near = exp1_near.loc[exp1_near.ablation_events_near_8000.notna()]
    exp1_rate = float(exp1_near.event_collection_stable.mean()) if not exp1_near.empty else np.nan
    exp2 = evidence.loc[evidence.target_dataset.eq("Exp2") & evidence.evidence_type.isin(["acceleration_evidence", "abrupt_change_evidence"])]
    stable_trajectory = float(ablations.trajectory_stable.mean()) if not ablations.empty else np.nan
    stable_events = float(ablations.event_collection_stable.mean()) if not ablations.empty else np.nan
    online_improved = int(forecast_benefit.get("online_better_output_horizon_count", 0))
    online_total = int(forecast_benefit.get("evaluated_output_horizon_count", 0))
    numeric = diagnostics.get("numerical_stability_check", {})
    overload = diagnostics.get("event_density_check", {})
    lines = [
        "# Continuous State Monitoring v4.1 report",
        "",
        "## Causal protocol",
        "",
        "The first 1,000 cycles are calibration-only. They freeze target robust location/scale, Ledoit–Wolf group covariances, evidence thresholds, and the causal residual reference; no formal state, evidence, or forecast row is emitted before the post-baseline monitoring start. Stage columns are never read by online modules.",
        "",
        "`rs_absmean` is excluded. rs, rx, and ry are normalized and scored in separate subspaces; their distances are fused with equal group weights. `baseline_outlier_fraction` uses the frozen target calibration, while `source_support_oos` uses the source experiment support envelope. V100/V500/V1000 replace the short v4 velocity scales. Abrupt evidence is an online EWMA-residual CUSUM, not an acceleration derivative.",
        "",
        "## Required answers",
        "",
        f"1. **Exp1 around 8,000 cycles:** event-collection stability in v4.1 ablations is {exp1_rate:.3f} for configurations with comparable near-8,000 event records. This is {'PASS' if np.isfinite(exp1_rate) and exp1_rate >= .80 else 'FAIL'} under the descriptive matching rule; it is not a wear label.",
        f"2. **Exp2 late acceleration/abrupt evidence:** v4.1 late-event fraction={v4_comparison.get('v41_exp2_late_fraction', np.nan):.3f}, v4 fraction={v4_comparison.get('v4_exp2_late_fraction', np.nan):.3f}; concentrated-more-than-v4={v4_comparison.get('more_concentrated_than_v4', False)}. Exp2 current onsets: `{', '.join(exp2.evidence_type.tolist()) if not exp2.empty else 'none'}`.",
        f"3. **Trajectory and complete-event-set stability:** trajectory-stable rate={stable_trajectory:.3f}; complete-event-collection-stable rate={stable_events:.3f}. This is {'PASS' if stable_trajectory >= .80 and stable_events >= .80 else 'FAIL'} across baseline, distance, and feature-group ablations, without changing thresholds after inspection.",
        f"4. **Online against best simple baseline:** Online RLS is lower-MAE in {online_improved}/{online_total} output–horizon–protocol comparisons. Strict all-comparison superiority is **{'PASS' if online_total and online_improved == online_total else 'FAIL'}**. Safe Gate compares Online with the current best of Zero Delta, Local Linear, Kalman, and Frozen Ridge.",
        f"5. **Event excess or numeric instability:** event-density status={overload.get('status', 'FAIL')}; numerical-stability status={numeric.get('status', 'FAIL')}. Event onset density={overload.get('onsets_per_1000_monitoring_cycles', np.nan):.3f}/1,000 cycles; finite state/forecast values={numeric.get('all_finite', False)}; maximum Online RLS prediction magnitude={numeric.get('max_online_prediction_abs', np.nan):.3f}.",
        "",
        "## Evidence summary",
        "",
        _table(evidence),
        "",
        "## Forecast comparison",
        "",
        _table(forecast_summary),
        "",
        "## PASS/FAIL record",
        "",
        f"Implementation diagnostics: **{diagnostics.get('implementation_acceptance', {}).get('status', 'FAIL')}**.",
        "",
        "```text", pytest_text.strip(), "```",
        "",
        "All failures above are retained as observed; no post-hoc acceptance tuning was applied.",
    ]
    return "\n".join(lines) + "\n"
