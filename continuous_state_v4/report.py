from __future__ import annotations

import numpy as np
import pandas as pd

from .state_engine import EVIDENCE_NAMES


def _yes_no(value: bool) -> str:
    return "是" if value else "否"


def _table(frame: pd.DataFrame) -> str:
    """Render a dependency-free text table inside the Markdown artifact."""
    return "```text\n" + frame.to_string(index=False) + "\n```" if not frame.empty else "No rows."


def make_report(
    states: pd.DataFrame,
    events: pd.DataFrame,
    evidence: pd.DataFrame,
    ablations: pd.DataFrame,
    forecast_summary: pd.DataFrame,
    forecast_benefit: dict[str, object],
    diagnostics: dict[str, dict[str, object]],
    pytest_text: str,
) -> str:
    exp1 = evidence.loc[evidence.target_dataset.eq("Exp1")]
    exp2 = evidence.loc[evidence.target_dataset.eq("Exp2")]
    exp1_active = exp1.loc[exp1.active_at_8000, "evidence_type"].tolist()
    exp1_near = events.loc[(events.target_dataset.eq("Exp1")) & events.event.eq("algorithm_evidence_onset") &
                           events.cycle.between(7800.0, 8200.0), ["evidence_type", "cycle"]]
    exp1_near_text = ", ".join(f"{row.evidence_type}@{row.cycle:.1f}" for _, row in exp1_near.iterrows()) if not exp1_near.empty else "none"
    exp2_persistent = exp2.loc[(exp2.evidence_type.isin(["directed_change_evidence", "acceleration_evidence"])) & (exp2.onset_count > 0)]
    stable = ablations.loc[ablations.stability_status.eq("PASS")] if not ablations.empty else ablations
    event_stable_rate = float(ablations.event_position_stable.mean()) if not ablations.empty else np.nan
    trajectory_stable_rate = float(ablations.trajectory_stable.mean()) if not ablations.empty else np.nan
    online_improved = int(forecast_benefit.get("online_better_output_horizon_count", 0))
    online_total = int(forecast_benefit.get("evaluated_output_horizon_count", 0))
    ablation_status = "PASS" if len(stable) == len(ablations) else "FAIL"
    online_global_status = "PASS" if online_total and online_improved == online_total else "FAIL"
    acceptance = diagnostics.get("implementation_acceptance", {}).get("status", "FAIL")
    lines = [
        "# Continuous State Monitoring v4 report",
        "",
        "## Scope and causal protocol",
        "",
        "v4 replaces the ordered platform-lock → platform-exit → serious-event state machine with four independent algorithm-state evidence tracks: `low_activity_evidence`, `directed_change_evidence`, `acceleration_evidence`, and `abrupt_change_evidence`. These are algorithmic evidence only; they are not stability/wear/failure labels or probabilities.",
        "",
        "Each target stream freezes robust location, scale, covariance/precision, and all evidence thresholds from its initial non-guard baseline. The online stream reads no Stage columns, predicts before RLS updates, pauses evidence tracks in restart guards, and is checked by prefix replay. D, V20/V50/V100, direction consistency, A, volatility, weighted OOS, every per-feature velocity vector, and rs/rx/ry velocity contributions are exported.",
        "",
        "## Required answers",
        "",
        f"1. **Exp1 near 8,000 cycles:** No v4 serious-wear class or alarm is emitted, so the former class-level false-alarm question does not apply as an output. Active tracks at the nearest 8,000-cycle window: `{', '.join(exp1_active) if exp1_active else 'none'}`; onset events in 7,800–8,200 cycles: `{exp1_near_text}`. Any such event is label-free algorithm evidence, not a serious-wear label.",
        f"2. **Exp2 without a platform:** {_yes_no(not exp2_persistent.empty)}. At least one onset was observed for the following independent directional/acceleration tracks: `{', '.join(exp2_persistent.evidence_type.tolist()) if not exp2_persistent.empty else 'none'}`. This result does not depend on platform detection.",
        f"3. **Baseline/distance/feature-group stability:** **{ablation_status}**. {len(stable)}/{len(ablations)} evidence-specific ablation rows meet the pre-declared descriptive stability rule (D Spearman ≥ 0.80 and matching/move ≤500-cycle first onset). Trajectory-stable rate={trajectory_stable_rate:.3f}; event-position-stable rate={event_stable_rate:.3f}. This is an observation, not a threshold-selection step.",
        f"4. **Does Online truly beat the best simple baseline?** Strict all-comparison superiority: **{online_global_status}**. Online RLS is lower-MAE than the best of Zero Delta, Local Linear, Kalman, and Frozen Ridge in only {online_improved}/{online_total} output–horizon–protocol comparisons. It is therefore {'not universally superior' if online_improved != online_total else 'superior in every evaluated comparison'}. Safe Gate selects Online only when its current cycle-window MAE is no worse than the current best static baseline.",
        "",
        "## Evidence summary",
        "",
        _table(evidence),
        "",
        "## Forecast comparison to the current best static baseline",
        "",
        _table(forecast_summary),
        "",
        "## PASS/FAIL record",
        "",
        f"Implementation diagnostics: **{acceptance}**.",
        "",
        "```text",
        pytest_text.strip(),
        "```",
        "",
        "Failures, if any, are retained above and were not removed by post-hoc parameter adjustment.",
    ]
    return "\n".join(lines) + "\n"
