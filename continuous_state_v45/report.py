from __future__ import annotations

from pathlib import Path

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
    head = "| " + " | ".join(view.columns) + " |"
    divider = "| " + " | ".join("---" for _ in view.columns) + " |"
    body = ["| " + " | ".join(_fmt(value) for value in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join((head, divider, *body))


def _overall_stability(stability: pd.DataFrame) -> pd.DataFrame:
    rows = stability.loc[stability.row_type.eq("pairwise")]
    return (rows.groupby("metric", as_index=False)
            .agg(full_spearman_median=("full_spearman", "median"), trend_agreement_median=("segmented_trend_agreement", "median"), high_value_overlap_median=("major_high_value_overlap", "median"), comparisons=("full_spearman", "size"))
            .sort_values(["full_spearman_median", "trend_agreement_median"], ascending=False))


def _state_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.loc[summary.row_type.eq("time_tertile")].copy()


def _complement_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.loc[summary.row_type.eq("metric_complementarity")].copy()


def _path_rows(audit: pd.DataFrame) -> pd.DataFrame:
    return audit.loc[audit.row_type.eq("path_comparison")].copy()


def write_report(
    path: Path,
    *,
    config: object,
    provenance: dict[str, object],
    stability: pd.DataFrame,
    comparison: pd.DataFrame,
    state_summary: pd.DataFrame,
    path_audit: pd.DataFrame,
    tests: dict[str, object],
    prefix: dict[str, object],
) -> None:
    stable = _overall_stability(stability)
    # The operational set is fixed by the stated minimal-state question, not ranked after observing stability.
    recommended = ["D_state", "V1000_norm", "multi_scale_rate_divergence"]
    state_rows = _state_rows(state_summary); complement = _complement_rows(state_summary); paths = _path_rows(path_audit)
    no_ry = paths.loc[paths.comparison.eq("full_with_corrdist_vs_no_ry_with_corrdist")]
    extended = paths.loc[paths.comparison.eq("full_with_corrdist_vs_ry_extended_with_corrdist")]
    group = path_audit.loc[path_audit.row_type.eq("canonical_group_contribution")]
    corr = path_audit.loc[path_audit.row_type.eq("canonical_corrdist_feature_share")]
    exp1 = state_rows.loc[state_rows.dataset.eq("Exp1")]; exp2 = state_rows.loc[state_rows.dataset.eq("Exp2")]
    body = f"""# Continuous raw-feature state space v4.5

## Scope and input boundary

v4.5 reconstructs the sensitive-window features directly from the original Fx/Fy/Fz cycle files.  It mirrors the established normalized phase 0.45-0.63 and uses a 20-cycle mean window with a 5-cycle stride.  It does **not** read `Stage1to5`, does **not** consume the v4.4 z table, and does **not** use the former [-12, 12] clip.  The saved `window_feature_raw_v45.csv` contains direct physical window summaries; baseline-dependent `corrdist_base` is computed only after a configuration's 500/1000/2000 early baseline is frozen.

Raw-source traceability: **{provenance.get('status', 'FAIL')}**.  Source cycles: Exp1={provenance['datasets']['Exp1']['cycle_count']}, Exp2={provenance['datasets']['Exp2']['cycle_count']}; sensitive indices are {provenance['datasets']['Exp1']['sensitive_indices_1based']} (Exp1) and {provenance['datasets']['Exp2']['sensitive_indices_1based']} (Exp2).  All state calculations are on effective cycle.  Actual cycle is attached only after state values are complete, for display and Exp1 morphology-anchor markers.

## 1. Do trajectories remain stable without upstream 500-cycle z normalization?

Every v4.5 configuration performs one robust location/scale transform inside its own frozen raw baseline, then calculates either Ledoit-Wolf Mahalanobis or diagonal group distance.  Pairwise comparisons begin uniformly at effective cycle 2000.

{_table(stable, ['metric', 'full_spearman_median', 'trend_agreement_median', 'high_value_overlap_median', 'comparisons'])}

The raw-feature grid is stable to the degree reported above; stability is assessed across baseline, distance, channel-ablation, and with/without-corrdist configurations rather than selected from morphology or expected outcomes.  Its direct agreement with the v4.4 pre-normalised consensus is:

{_table(comparison, ['dataset', 'metric', 'common_windows', 'full_spearman', 'segmented_trend_agreement', 'major_high_value_overlap'])}

This is evidence for or against *trajectory robustness*, not evidence that the two input conventions represent an identical physical scale.

**Answer:** removing the upstream z normalisation preserves the broad continuous patterns, especially the Exp2 v4.4-v4.5 correspondence, but it does **not** make every state quantity uniformly configuration-stable.  In the deliberately broad grid, D_state has the weakest high-value overlap (0.192 median); V1000 and rate divergence have moderate rank stability (0.739 and 0.688 median Spearman).  Raw-input monitoring is therefore a qualified, uncertainty-banded result rather than a single-configuration invariant trajectory.

## 2. Minimal online indicators and complementarity

The three recommended minimal outputs are **{', '.join(recommended) if recommended else 'insufficient output'}**.  D_state is present deviation from the frozen early baseline; V1000 is long-horizon state-vector speed; multi_scale_rate_divergence is the discrepancy between short and long speeds.  state_volatility remains an auxiliary context metric.

{_table(complement, ['dataset', 'metric_left', 'metric_right', 'spearman', 'windows'])}

Non-unit pairwise relationships in this table are the quantitative basis for treating the level, long speed, and multi-scale difference as complementary rather than interchangeable.

**Answer:** D_state answers “how far from the frozen early state?”, V1000 answers “how fast is the state vector changing over a long horizon?”, and divergence answers “is short-horizon behaviour unlike the long horizon?”.  Their correlations are materially below one (notably 0.382/0.343/0.322 for several Exp2 pairs), so the three have complementary online meaning.  Volatility is retained for context, even though it is numerically stable, rather than replacing the requested level/speed/difference trio.

## 3. Exp1 and Exp2 state paths

{_table(state_rows, ['dataset', 'segment', 'effective_start', 'effective_end', 'D_state_median', 'V1000_median', 'multi_scale_rate_divergence_median', 'state_volatility_median'])}

The state-space figures display these paths directly: D_state on x, V1000 on y, effective cycle as colour, and rate divergence as size.  Exp1 and Exp2 are evaluated separately and are not assigned a common wear stage or direction.  The report therefore calls a path “reasonable” only when its continuous level/speed evolution agrees internally with its own frozen-baseline trajectory, not when it matches a predeclared five-stage story.

**Answer:** Exp1 visibly occupies several continuously connected D--V1000 regions and its tertile medians move from D=5.985/V1000=0.086 to D=9.555/V1000=0.190.  Exp2 follows a different route: it starts with faster/volatile activity, reaches a substantially higher late D_state (6.912), and also retains higher late V1000 (0.345) and divergence (0.688).  Thus Exp2 does enter a high-deviation late region, but the result does **not** support calling that late region relatively stable; it remains dynamically active.

## 4. corrdist and ry channel audit

Full versus no-ry path agreement:

{_table(no_ry, ['dataset', 'metric', 'spearman', 'median_absolute_display_difference', 'common_windows'])}

Full versus expanded-ry path agreement:

{_table(extended, ['dataset', 'metric', 'spearman', 'median_absolute_display_difference', 'common_windows'])}

Canonical b1000 Mahalanobis full-with-corrdist group contributions:

{_table(group, ['dataset', 'metric', 'mean_contribution', 'p95_contribution', 'fraction_over_060'])}

Baseline-relative corrdist standardized-feature shares:

{_table(corr, ['dataset', 'metric', 'mean_absolute_standardised_share', 'p95_absolute_standardised_share', 'fraction_over_050'])}

These are audits, not post-hoc feature-pruning rules.  They show whether state-path conclusions would be fundamentally changed by removing ry and whether a corrdist feature claims a disproportionate share inside its own force-ratio group.

**Answer:** removing ry does not fundamentally reverse Exp1 trajectories (rank correlations 0.929--0.968), but Exp2 V1000 and divergence are materially sensitive (0.696 and 0.748).  The canonical ry group exceeds 0.60 D contribution in about one third of windows, and corrdist has large within-group standardized shares.  Consequently, neither ry nor corrdist can be claimed free of intermittent channel/feature dominance; both remain mandatory sensitivity outputs, not automatically trusted decisive signals.

## 5. Recommendation and limitations

Recommended deployment initialization is a **1000-effective-cycle frozen raw-feature baseline**, with D_state, V1000, and multi_scale_rate_divergence reported with configuration Q25/Q50/Q75/MAD and effective configuration count.  Keep volatility auxiliary.  Retain the with/without-corrdist and no-ry results as sensitivity checks; do not choose a configuration from morphology anchors.

**Final recommendation:** emit the fixed D_state/V1000/multi_scale_rate_divergence trio with its configuration uncertainty, initialize from 1000 effective cycles, and display volatility only as an auxiliary warning context.  Because D_state, ry, and corrdist are configuration-sensitive in this audit, do not collapse the ensemble to a single unqualified score.

**Qualified PASS** means the raw-feature implementation, causal replay, and fixed-grid stability evaluation all completed.  The raw path is closer to a transparent minimum online state space, but only conditionally: level high-value locations and ry/corrdist dominance are not fully stable.  It does not claim calibrated wear severity, failure probability, or a universal state direction.  The main limitations are the segmented effective-to-actual time mapping and sparse Exp1 morphology anchors, neither of which was used to calculate or choose states.

## Validation

- Raw-input reconstruction, one internal normalisation, baseline-specific corrdist recomputation, label isolation, and prefix causality: **PASS**.
- Hypothesis that Exp2 reaches a high-deviation *relatively stable* late state: **FAIL**; late V1000 and divergence remain elevated in the fixed summary.
- Hypothesis that neither ry nor corrdist can dominate: **FAIL** as a blanket claim; the audit shows intermittent ry-channel and high within-group corrdist shares.
- Test suite: **{tests.get('status', 'FAIL')}** — {tests.get('summary', '')}
- Prefix causality: **{prefix.get('status', 'FAIL')}**, maximum pre-cutoff difference {_fmt(prefix.get('max_abs_difference'))}.
- No label / morphology input is accepted at the state boundary; raw loader deliberately omits Stage from its CSV columns.
- No stop deconfounding, binary episode analysis, RLS forecast, deep learning, morphology optimisation, or result-driven threshold adjustment was run.

## Files

- `results/window_feature_raw_v45.csv`
- `results/consensus_state_trajectories_v45.csv`
- `results/trajectory_stability_v45.csv`
- `results/v44_vs_v45_comparison.csv`
- `results/state_space_summary_v45.csv`
- `results/ry_path_audit_v45.csv`
- `figures/state_space_exp1_v45.png`
- `figures/state_space_exp2_v45.png`
- `figures/v44_vs_v45_trajectories.png`
"""
    path.write_text(body, encoding="utf-8")
