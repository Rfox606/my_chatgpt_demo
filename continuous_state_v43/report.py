from __future__ import annotations

import numpy as np
import pandas as pd


def _table(frame: pd.DataFrame, limit: int = 30) -> str:
    return "```text\n" + frame.head(limit).to_string(index=False) + "\n```" if not frame.empty else "No rows."


def _near(episodes: pd.DataFrame, dataset: str, cycle: float) -> pd.DataFrame:
    hit = episodes.loc[episodes.target_dataset.eq(dataset) & (episodes.start_cycle_actual <= cycle) & (episodes.end_cycle_actual >= cycle)]
    if not hit.empty:
        return hit
    return episodes.loc[episodes.target_dataset.eq(dataset)].assign(distance=lambda frame: np.abs(frame.peak_cycle_actual - cycle)).nsmallest(1, "distance")


def make_report(episodes: pd.DataFrame, deconfounding: pd.DataFrame, time_diagnostic: dict[str, object], morphology: pd.DataFrame, ry_physical: pd.DataFrame, ry_feature: pd.DataFrame, forecast_summary: pd.DataFrame, forecast_benefit: dict[str, object], diagnostics: dict[str, dict[str, object]], pytest_text: str) -> str:
    exp1 = _near(episodes, "Exp1", 8000); exp2 = _near(episodes, "Exp2", 20000)
    summaries = deconfounding.loc[deconfounding.row_type.eq("summary")]
    focus_id = str(exp1.episode_id.iloc[0]) if not exp1.empty else ""
    focus_deconf = deconfounding.loc[(deconfounding.row_type.eq("original_episode_match")) & deconfounding.episode_id.eq(focus_id)]
    focus_retained = bool(focus_deconf.retained.astype(str).str.lower().eq("true").all()) if not focus_deconf.empty else False
    metric_summary = pd.DataFrame(time_diagnostic.get("metric_summary", []))
    ry_summary = ry_physical.loc[ry_physical.row_type.eq("summary")] if "row_type" in ry_physical else pd.DataFrame()
    improved = int(ry_summary.ry_improves_absolute_D_correspondence.astype(str).str.lower().eq("true").sum()) if not ry_summary.empty else 0
    exp1_ry = ry_feature.loc[ry_feature.dataset.eq("Exp1")].iloc[0] if not ry_feature.empty and ry_feature.dataset.eq("Exp1").any() else pd.Series(dtype=float)
    focus_text = "none" if focus_deconf.empty else "; ".join(
        f"+/-{row.stop_exclusion_half_width_actual}: retained={row.retained}, IoU={float(row.interval_iou_actual):.3f}, peak shift={float(row.peak_shift_actual):.1f}"
        for _, row in focus_deconf.iterrows()
    )
    deconf_text = "; ".join(
        f"+/-{row.stop_exclusion_half_width_actual}: retention={float(row.retention_rate):.3f}, mean IoU={float(row.mean_interval_iou_actual):.3f}"
        for _, row in summaries.iterrows()
    )
    online = int(forecast_benefit.get("online_better_output_horizon_count", 0)); total = int(forecast_benefit.get("evaluated_output_horizon_count", 0))
    lines = [
        "# Continuous State Monitoring v4.3 report", "",
        "## Method and scope", "",
        "v4.3 preserves v4.2's label-free effective-cycle state calculation, 1,000-effective-cycle calibration, grouped equal-weight distance, consensus construction, and Safe Gate. A pre-existing piecewise mapping configuration is used only because no row-level actual-cycle index was found in the repository search. Mapping is used for time coordinates, actual-cycle Guard, stop deconfounding, and post-hoc physical correspondence; stage and morphology fields are not model inputs, thresholds, or episode criteria.", "",
        "Actual stop locations are Exp1 8k/16k/24k/32k/40k/48k and Exp2 every 500 actual cycles from 500 to 23,500. The v4.2 effective-time state is the main result. An independently replayed canonical actual-time sensitivity run changes only time lags for V100/V500/V1000 and their downstream causal evidence timing; it does not reuse future values or replace the main result.", "",
        "## Required answers", "",
        f"1. **Exp1 effective 7,570 to 7,985 mapped to actual time / 8,000 stop:** {_table(exp1, 3)}. Focused stop exclusion: {focus_text}; this stop-adjacent interval retains a post-Guard remainder: **{'PASS' if focus_retained else 'FAIL'}**. This does not establish an independent physical mechanism: the full event set is stop-sensitive ({deconf_text}).",
        f"2. **Exp2 local change near actual 20,000:** {_table(exp2, 4)}. The pre-declared long-episode split rule is based only on consensus-score valleys, sustained support decline, or persistent dominant-evidence change; result is **{'PASS' if not exp2.empty and (exp2.start_cycle_actual.le(20000) & exp2.end_cycle_actual.ge(20000)).any() else 'FAIL'}**.",
        f"3. **Effective-time vs actual-time:** episode peak Jaccard (500 actual cycles)={time_diagnostic.get('episode_peak_jaccard_500_actual_cycles', np.nan):.3f}. Metric comparison is below; values quantify sensitivity rather than selecting the favorable time basis.",
        f"4. **ry_p2p and morphology:** full configuration improves absolute D-state morphology correspondence in {improved}/4 metrics (Sa/Sq/Sz/Sku). Exp1 ry dominance over 0.60 occurs in {float(exp1_ry.get('ry_subspace_dominance_fraction_over_060', np.nan)):.3f} of canonical monitoring rows; single-feature flag={exp1_ry.get('single_feature_dominance_flag', 'NA')}. The sparse audit therefore does not support ry_p2p as a physical improvement; no feature or threshold was changed from it.",
        f"5. **Forecast unchanged:** Online RLS is better than the best static baseline in {online}/{total} output-horizon-protocol comparisons. This remains a limited rather than global advantage.", "",
        "## Primary actual-cycle localized episodes", "", _table(episodes), "",
        "## Stop deconfounding", "", _table(summaries), "", _table(deconfounding.loc[deconfounding.row_type.eq("original_episode_match")], 20), "",
        "## Effective vs actual-time sensitivity", "", _table(metric_summary), "",
        "## Post-hoc morphology correlations", "", _table(morphology, 40), "",
        "## ry physical audit", "", _table(ry_summary), "",
        "## Forecast comparison", "", _table(forecast_summary), "",
        "## PASS/FAIL record", "",
        f"Cycle mapping diagnostic: **{diagnostics.get('cycle_mapping_check', {}).get('status', 'FAIL')}**. Implementation diagnostic: **{diagnostics.get('implementation_acceptance', {}).get('status', 'FAIL')}**. Tests: **{diagnostics.get('pytest', {}).get('status', 'FAIL')}**.", "",
        "```text", pytest_text.strip(), "```", "",
        "Observed FAIL results and sensitivity risks are retained. No stage labels, morphology anchors, stop-deconfounding result, or physical correlation was used to refit a baseline, threshold, feature set, or episode rule.",
    ]
    return "\n".join(lines) + "\n"
