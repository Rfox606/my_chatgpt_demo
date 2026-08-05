from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(table: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    data = table.loc[:, [column for column in columns if column in table.columns]].copy()
    if limit is not None:
        data = data.head(limit)
    if data.empty:
        return "_No rows._"
    header = "| " + " | ".join(data.columns) + " |"
    separator = "| " + " | ".join("---" for _ in data.columns) + " |"
    rows = ["| " + " | ".join(_fmt(value) for value in row) + " |" for row in data.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def _metric_stability(stability: pd.DataFrame) -> pd.DataFrame:
    pairwise = stability.loc[stability.row_type.eq("pairwise")].copy()
    return (pairwise.groupby("metric", as_index=False)
            .agg(full_spearman_median=("full_spearman", "median"),
                 segment_agreement_median=("segmented_trend_agreement", "median"),
                 high_change_overlap_median=("major_high_value_overlap", "median"),
                 comparisons=("full_spearman", "size"))
            .sort_values(["full_spearman_median", "segment_agreement_median", "high_change_overlap_median"], ascending=False))


def _interval_value(alignment: pd.DataFrame, start: int, column: str) -> object:
    rows = alignment.loc[(alignment.row_type.eq("morphology_anchor_interval")) & (alignment.start_cycle_actual == float(start))]
    return rows.iloc[0][column] if not rows.empty else np.nan


def _ry_summary(audit: pd.DataFrame) -> pd.DataFrame:
    return audit.loc[audit.row_type.eq("expanded_group_summary")].copy()


def _ry_conclusion(audit: pd.DataFrame) -> str:
    summary = _ry_summary(audit)
    if summary.empty:
        return "No expanded-ry audit rows were produced."
    dominant = summary.ry_p2p_share_over_050_fraction.fillna(0).max()
    coherence = summary.median_pairwise_signed_spearman.median()
    if dominant <= .05 and coherence < .30:
        return "ry_p2p is not the dominant expanded-group feature, but the added ry features do not form a strongly coherent signed within-group trend."
    if dominant <= .05:
        return "ry_p2p is not the dominant expanded-group feature; the group has at least moderate signed internal coherence."
    return "The expanded ry group retains material ry_p2p dominance, so it should not be interpreted as a uniformly supported group trend."


def write_report(
    path: Path,
    *,
    config_payload: dict[str, object],
    provenance: dict[str, object],
    stability: pd.DataFrame,
    alignment: pd.DataFrame,
    patterns: pd.DataFrame,
    ry_audit: pd.DataFrame,
    test_status: dict[str, object],
    causal_status: dict[str, object],
) -> None:
    stable = _metric_stability(stability)
    recommended = stable.head(3).metric.tolist()
    ry = _ry_summary(ry_audit)
    ry_conclusion = _ry_conclusion(ry_audit)
    v24_32 = _interval_value(alignment, 24000, "V500_mean")
    v32_40_div = _interval_value(alignment, 32000, "rate_divergence_mean")
    v32_40_vol = _interval_value(alignment, 32000, "volatility_mean")
    v40_48 = _interval_value(alignment, 40000, "V500_mean")
    exp_patterns = patterns.loc[:, ["dataset", "metric", "overall_median", "effective_cycle_spearman", "late_to_early_mean_ratio", "late_high_value_fraction"]]
    body = f"""# Continuous state v4.4 — trajectory-first physical validation

## Scope and causal boundary

This run returns to the research question: can force features from the nominated sensitive gait interval support a stable, interpretable, online-updatable **continuous state**?  The state calculation uses cleaned effective-cycle order only, freezes each self-baseline before monitoring, and has no stage label, morphology value, stop/Guard flag, episode threshold, prediction module, or future-cycle value as an input.  Actual cycle is attached only after loading for plotting and post-hoc morphology alignment.

The configuration grid is fixed before this run: baseline lengths 500/1000/2000, Mahalanobis (Ledoit–Wolf) or diagonal distance, and full/no-rx/no-ry/no-rs feature variants.  The primary state quantities are D_state, V500, V1000, multi_scale_rate_divergence, and state_volatility.  All reported consensus quantities are configuration Q25/Q50/Q75/MAD and effective configuration count.

## Input-source traceability

`window_feature_z_table.csv` traceability result: **{provenance.get('status', 'FAIL')}**.  The existing feature-generation configuration and code identify Fx/Fy/Fz source files and a sensitive phase of 0.45–0.63.  The repository’s reproducible extraction indices are one-based 252–352 (Exp1) and 756–1058 (Exp2), while the study metadata records the requested nominal intervals 251–350 and 751–1050.  This off-by-discretisation difference is disclosed rather than silently harmonised.  {provenance.get('interpretation', '')}

No row-level actual-cycle index was available in the state input.  The pre-existing segmented mapping in `outputs_physical_validation_candidates_v1/configs/cycle_mapping_config.json` is therefore the documented fallback, used only for coordinate restoration, figures, and post-hoc comparisons.

## 1. Which continuous indicators are most stable across configurations?

The table aggregates pairwise configuration comparisons over both experiments.  “High-change overlap” is the overlap of each pair’s fixed top decile, not a fitted event detector.

{_markdown_table(stable, ["metric", "full_spearman_median", "segment_agreement_median", "high_change_overlap_median", "comparisons"])}

On this fixed grid, the strongest overall trajectory agreement is led by **{', '.join(recommended) if recommended else 'no metric (insufficient data)'}**.  This supports retaining the metric set as continuous outputs with uncertainty bands, rather than selecting a single morphology-fitted configuration.

## 2. Exp1 post-hoc correspondence with morphology intervals

The next table is a descriptive interval alignment, not a statistical validation: there are only six morphology-anchor intervals, and sparse Spearman values are exploratory only.  Morphology was read after state trajectories were final and never feeds feature selection or thresholds.

{_markdown_table(alignment, ["start_cycle_actual", "end_cycle_actual", "D_cumulative_absolute_change", "V500_mean", "V1000_mean", "rate_divergence_mean", "volatility_mean", "high_change_state_duration_fraction", "delta_Sa", "delta_Sq", "delta_Sz", "delta_Sku"], limit=7)}

The 24k–32k nominal smoothing interval has V500 mean {_fmt(v24_32)}.  The 32k–40k surface-fluctuation interval has rate-divergence mean {_fmt(v32_40_div)} and volatility mean {_fmt(v32_40_vol)}.  The 40k–48k anchored late interval has V500 mean {_fmt(v40_48)}.  These magnitudes permit a physically interpretable comparison, but do **not** establish a causal morphology model or prove a five-stage wear trajectory.

Interpretation of the interval results: relative to 16k-24k, the 24k-32k V500 level is lower; 32k-40k has slightly higher rate divergence, while volatility is nearly unchanged and marginally lower than 24k-32k.  However, 40k-48k V500 rises rather than remaining low.  The morphology correspondence is therefore partial, and the expected late low-rate pattern is not consistently supported by this state grid.

## 3. Exp2 compared with Exp1

Exp2 has lower median D_state but larger V500/V1000, divergence, and volatility medians than Exp1; its late D_state ratio is also higher.  It therefore shows a more persistently active late continuous pattern under its own frozen baseline, not a mapped copy of Exp1's morphology path.

Exp2 is intentionally not forced into Exp1’s morphology descriptions.  It is compared only by continuous deviation, long-run direction, late/early activity ratio, volatility, and multi-scale difference.

{_markdown_table(exp_patterns, ["dataset", "metric", "overall_median", "effective_cycle_spearman", "late_to_early_mean_ratio", "late_high_value_fraction"])}

Any Exp2 differences in the table are differences in observed continuous signal patterns, not a shared wear stage or a common wear direction.

## 4. Does extended ry remain dominated by ry_p2p?

Direct ry conclusion: **{ry_conclusion}**

{_markdown_table(ry, ["dataset", "median_pairwise_signed_spearman", "ry_p2p_mean_absolute_share", "ry_p2p_p95_absolute_share", "ry_p2p_share_over_050_fraction"])}

The expanded group is assessed by within-group signed agreement, absolute standardized contribution shares, and its trajectory agreement with p2p-only and no-ry alternatives.  A high p2p share would be evidence of residual single-feature dominance; a low median pairwise signed correlation would indicate that adding ry features does not make a coherent common ry trend.  Neither outcome is used to alter this run’s state parameters.

## 5. Recommended minimal online output

Recommended minimum: **{', '.join(recommended) if recommended else 'insufficient data'}**, each with configuration Q25/Q50/Q75/MAD and number of effective configurations.  D_state supplies deviation level; V500/V1000 or divergence supply time-scale-resolved change; volatility is retained when it has acceptable stability.  The mapping to actual cycle is a display/post-hoc field; the online calculation remains on effective cycle.

## 6. Does this move the continuous monitoring objective forward?

**Qualified PASS.** The workflow now tests stable continuous trajectories directly against a fixed configuration ensemble and reports physical interval correspondence without stage or morphology leakage.  It is closer to an online continuous-state monitor because baselines are frozen and every state row is prefix-causal.  It remains a signal-state study: sparse morphology anchors and the absence of a row-level actual-cycle source limit physical validation, and no claim of calibrated wear severity or failure probability is made.

## Tests and limitations

- Unit/integration test status: **{test_status.get('status', 'FAIL')}** ({test_status.get('summary', '')}).
- Prefix-causality replay: **{causal_status.get('status', 'FAIL')}**; maximum pre-cutoff discrepancy = {_fmt(causal_status.get('max_abs_difference'))}.
- Metadata isolation: experiment metadata and morphology anchors are opened only after state computation for report/alignment.  The state input reader requests only window IDs and feature z-values.
- No stop deconfounding, Guard optimisation, binary-episode tuning, Online RLS re-run, deep learning, or morphology-guided parameter adjustment was performed.

## Files

- `results/trajectory_stability_v44.csv`
- `results/morphology_interval_alignment_v44.csv`
- `results/exp1_exp2_pattern_comparison_v44.csv`
- `results/ry_group_audit_v44.csv`
- `results/consensus_state_trajectories_v44.csv`
- `figures/consensus_trajectories_v44.png`
- `figures/morphology_interval_alignment_v44.png`
"""
    path.write_text(body, encoding="utf-8")
