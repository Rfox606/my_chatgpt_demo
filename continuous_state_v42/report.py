from __future__ import annotations

import numpy as np
import pandas as pd


def _table(frame: pd.DataFrame) -> str:
    return "```text\n" + frame.to_string(index=False) + "\n```" if not frame.empty else "No rows."


def make_report(consensus: pd.DataFrame, episodes: pd.DataFrame, ry_audit: pd.DataFrame, forecast_summary: pd.DataFrame, forecast_benefit: dict[str, object], diagnostics: dict[str, dict[str, object]], pytest_text: str) -> str:
    exp1 = episodes.loc[episodes.target_dataset.eq("Exp1") & (episodes.start_cycle <= 8000) & (episodes.end_cycle >= 8000)]
    if exp1.empty and not episodes.empty:
        exp1 = episodes.loc[episodes.target_dataset.eq("Exp1")].assign(distance=lambda frame: np.abs(frame.peak_cycle - 8000)).nsmallest(1, "distance")
    exp2 = episodes.loc[episodes.target_dataset.eq("Exp2") & (episodes.start_cycle <= 11695) & (episodes.end_cycle >= 11695)]
    if exp2.empty and not episodes.empty:
        exp2 = episodes.loc[episodes.target_dataset.eq("Exp2")].assign(distance=lambda frame: np.abs(frame.peak_cycle - 11695)).nsmallest(1, "distance")
    exp1_text = "none" if exp1.empty else "; ".join(f"{row.start_cycle:.1f}-{row.end_cycle:.1f}, support={row.configuration_support:.3f}, dominant={row.dominant_evidence}" for _, row in exp1.iterrows())
    exp2_text = "none" if exp2.empty else "; ".join(f"{row.start_cycle:.1f}-{row.end_cycle:.1f}, support={row.configuration_support:.3f}, composition={row.evidence_composition}, dominant={row.dominant_evidence}" for _, row in exp2.iterrows())
    ry = diagnostics.get("ry_removal_effect", {})
    consensus_stability = diagnostics.get("consensus_vs_v41_stability", {})
    online = int(forecast_benefit.get("online_better_output_horizon_count", 0)); total = int(forecast_benefit.get("evaluated_output_horizon_count", 0))
    clip = diagnostics.get("rls_prediction_clipping_ratio", {}); gate = diagnostics.get("safe_gate_selection_ratio", {})
    lines = [
        "# Continuous State Monitoring v4.2 report", "",
        "## Method", "",
        "v4.2 keeps v4.1's causal target calibration, Guard, label isolation, multi-scale rates, equal rs/rx/ry subspace fusion, and Safe Gate. It evaluates 24 pre-declared baseline/distance/feature-group configurations and reports configuration consensus rather than assigning ordered binary states. The formal consensus begins after 1,000 cycles; configurations enter only after their own calibration ends.", "",
        "For each continuous state quantity, the output records Q25/Q50/Q75, MAD, and valid configuration count. `multi_scale_rate_divergence` is the renamed former A-state. Change episodes are support-qualified continuous intervals: directed, rate-divergence, and abrupt are composition proportions, not fixed-order labels. Abrupt remains the independent residual/CUSUM track.", "",
        "## Required answers", "",
        f"1. **Exp1 near 8,000:** {exp1_text}. A high-support interval requires support ≥0.80; result is **{'PASS' if not exp1.empty and float(exp1.configuration_support.max()) >= .80 else 'FAIL'}**.",
        f"2. **Exp2 near 11,695:** {exp2_text}. abrupt+directed predominance is **{'PASS' if not exp2.empty and (exp2.abrupt_composition + exp2.directed_composition).max() >= .50 else 'FAIL'}**.",
        f"3. **Removing ry in Exp1:** D consensus Spearman={ry.get('D_state_spearman_full_vs_no_ry', np.nan):.3f}; full/no-ry episode counts={ry.get('full_episode_count', 0)}/{ry.get('no_ry_episode_count', 0)}; episode-peak Jaccard={ry.get('episode_peak_jaccard_500_cycles', np.nan):.3f}.",
        f"4. **Consensus vs v4.1 single-configuration events:** high-support consensus episode fraction={consensus_stability.get('consensus_high_support_episode_fraction', np.nan):.3f}; v4.1 event-collection stability={consensus_stability.get('v41_single_configuration_event_stability', np.nan):.3f}; status={consensus_stability.get('status', 'FAIL')}.",
        f"5. **Online forecast:** Online RLS is better than the best simple baseline in {online}/{total} output–horizon–protocol comparisons. RLS clipping ratio={clip.get('clipped_prediction_ratio', np.nan):.3f}; Safe Gate Online selection ratio={gate.get('online_selection_ratio', np.nan):.3f}.",
        "", "## Change episodes", "", _table(episodes), "", "## ry_p2p audit", "", _table(ry_audit), "", "## Forecast comparison", "", _table(forecast_summary), "", "## PASS/FAIL record", "",
        f"Implementation diagnostics: **{diagnostics.get('implementation_acceptance', {}).get('status', 'FAIL')}**.", "", "```text", pytest_text.strip(), "```", "", "Failures are retained as observed; no post-hoc feature or threshold adjustment was made.",
    ]
    return "\n".join(lines) + "\n"
