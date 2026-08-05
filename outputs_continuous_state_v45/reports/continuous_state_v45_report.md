# Continuous raw-feature state space v4.5

## Scope and input boundary

v4.5 reconstructs the sensitive-window features directly from the original Fx/Fy/Fz cycle files.  It mirrors the established normalized phase 0.45-0.63 and uses a 20-cycle mean window with a 5-cycle stride.  It does **not** read `Stage1to5`, does **not** consume the v4.4 z table, and does **not** use the former [-12, 12] clip.  The saved `window_feature_raw_v45.csv` contains direct physical window summaries; baseline-dependent `corrdist_base` is computed only after a configuration's 500/1000/2000 early baseline is frozen.

Raw-source traceability: **PASS**.  Source cycles: Exp1=45591, Exp2=14100; sensitive indices are [252, 352] (Exp1) and [756, 1058] (Exp2).  All state calculations are on effective cycle.  Actual cycle is attached only after state values are complete, for display and Exp1 morphology-anchor markers.

## 1. Do trajectories remain stable without upstream 500-cycle z normalization?

Every v4.5 configuration performs one robust location/scale transform inside its own frozen raw baseline, then calculates either Ledoit-Wolf Mahalanobis or diagonal group distance.  Pairwise comparisons begin uniformly at effective cycle 2000.

| metric | full_spearman_median | trend_agreement_median | high_value_overlap_median | comparisons |
| --- | --- | --- | --- | --- |
| state_volatility | 0.797 | 0.800 | 0.414 | 2256.000 |
| V1000_norm | 0.739 | 0.800 | 0.335 | 2256.000 |
| multi_scale_rate_divergence | 0.688 | 0.800 | 0.341 | 2256.000 |
| D_state | 0.683 | 0.800 | 0.192 | 2256.000 |

The raw-feature grid is stable to the degree reported above; stability is assessed across baseline, distance, channel-ablation, and with/without-corrdist configurations rather than selected from morphology or expected outcomes.  Its direct agreement with the v4.4 pre-normalised consensus is:

| dataset | metric | common_windows | full_spearman | segmented_trend_agreement | major_high_value_overlap |
| --- | --- | --- | --- | --- | --- |
| Exp1 | D_state | 8717.000 | 0.897 | 1.000 | 0.366 |
| Exp1 | V1000_norm | 8717.000 | 0.842 | 0.800 | 0.556 |
| Exp1 | multi_scale_rate_divergence | 8717.000 | 0.679 | 0.600 | 0.324 |
| Exp1 | state_volatility | 8717.000 | 0.636 | 0.600 | 0.701 |
| Exp2 | D_state | 2419.000 | 0.960 | 1.000 | 0.592 |
| Exp2 | V1000_norm | 2419.000 | 0.943 | 0.800 | 0.597 |
| Exp2 | multi_scale_rate_divergence | 2419.000 | 0.969 | 0.600 | 0.779 |
| Exp2 | state_volatility | 2419.000 | 0.799 | 0.800 | 0.458 |

This is evidence for or against *trajectory robustness*, not evidence that the two input conventions represent an identical physical scale.

**Answer:** removing the upstream z normalisation preserves the broad continuous patterns, especially the Exp2 v4.4-v4.5 correspondence, but it does **not** make every state quantity uniformly configuration-stable.  In the deliberately broad grid, D_state has the weakest high-value overlap (0.192 median); V1000 and rate divergence have moderate rank stability (0.739 and 0.688 median Spearman).  Raw-input monitoring is therefore a qualified, uncertainty-banded result rather than a single-configuration invariant trajectory.

## 2. Minimal online indicators and complementarity

The three recommended minimal outputs are **D_state, V1000_norm, multi_scale_rate_divergence**.  D_state is present deviation from the frozen early baseline; V1000 is long-horizon state-vector speed; multi_scale_rate_divergence is the discrepancy between short and long speeds.  state_volatility remains an auxiliary context metric.

| dataset | metric_left | metric_right | spearman | windows |
| --- | --- | --- | --- | --- |
| Exp1 | D_state | V1000 | 0.535 | 8717.000 |
| Exp1 | D_state | multi_scale_rate_divergence | 0.499 | 8717.000 |
| Exp1 | D_state | state_volatility | 0.583 | 8717.000 |
| Exp1 | V1000 | multi_scale_rate_divergence | 0.623 | 8717.000 |
| Exp1 | V1000 | state_volatility | 0.780 | 8717.000 |
| Exp1 | multi_scale_rate_divergence | state_volatility | 0.658 | 8717.000 |
| Exp2 | D_state | V1000 | 0.673 | 2419.000 |
| Exp2 | D_state | multi_scale_rate_divergence | 0.382 | 2419.000 |
| Exp2 | D_state | state_volatility | 0.342 | 2419.000 |
| Exp2 | V1000 | multi_scale_rate_divergence | 0.343 | 2419.000 |
| Exp2 | V1000 | state_volatility | 0.322 | 2419.000 |
| Exp2 | multi_scale_rate_divergence | state_volatility | 0.414 | 2419.000 |

Non-unit pairwise relationships in this table are the quantitative basis for treating the level, long speed, and multi-scale difference as complementary rather than interchangeable.

**Answer:** D_state answers “how far from the frozen early state?”, V1000 answers “how fast is the state vector changing over a long horizon?”, and divergence answers “is short-horizon behaviour unlike the long horizon?”.  Their correlations are materially below one (notably 0.382/0.343/0.322 for several Exp2 pairs), so the three have complementary online meaning.  Volatility is retained for context, even though it is numerically stable, rather than replacing the requested level/speed/difference trio.

## 3. Exp1 and Exp2 state paths

| dataset | segment | effective_start | effective_end | D_state_median | V1000_median | multi_scale_rate_divergence_median | state_volatility_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1 | early | 2000.500 | 16525.500 | 5.985 | 0.086 | 0.342 | 0.439 |
| Exp1 | middle | 16530.500 | 31050.500 | 7.538 | 0.100 | 0.482 | 0.589 |
| Exp1 | late | 31055.500 | 45580.500 | 9.555 | 0.190 | 0.626 | 0.905 |
| Exp2 | early | 2000.500 | 6030.500 | 3.075 | 0.165 | 0.566 | 2.335 |
| Exp2 | middle | 6035.500 | 10060.500 | 2.027 | 0.218 | 0.701 | 1.806 |
| Exp2 | late | 10065.500 | 14090.500 | 6.912 | 0.345 | 0.688 | 3.554 |

The state-space figures display these paths directly: D_state on x, V1000 on y, effective cycle as colour, and rate divergence as size.  Exp1 and Exp2 are evaluated separately and are not assigned a common wear stage or direction.  The report therefore calls a path “reasonable” only when its continuous level/speed evolution agrees internally with its own frozen-baseline trajectory, not when it matches a predeclared five-stage story.

**Answer:** Exp1 visibly occupies several continuously connected D--V1000 regions and its tertile medians move from D=5.985/V1000=0.086 to D=9.555/V1000=0.190.  Exp2 follows a different route: it starts with faster/volatile activity, reaches a substantially higher late D_state (6.912), and also retains higher late V1000 (0.345) and divergence (0.688).  Thus Exp2 does enter a high-deviation late region, but the result does **not** support calling that late region relatively stable; it remains dynamically active.

## 4. corrdist and ry channel audit

Full versus no-ry path agreement:

| dataset | metric | spearman | median_absolute_display_difference | common_windows |
| --- | --- | --- | --- | --- |
| Exp1 | D_state | 0.938 | 0.377 | 8717.000 |
| Exp1 | V1000_norm | 0.959 | 0.095 | 8717.000 |
| Exp1 | multi_scale_rate_divergence | 0.929 | 0.152 | 8717.000 |
| Exp1 | state_volatility | 0.968 | 0.101 | 8717.000 |
| Exp2 | D_state | 0.832 | 0.147 | 2419.000 |
| Exp2 | V1000_norm | 0.696 | 0.368 | 2419.000 |
| Exp2 | multi_scale_rate_divergence | 0.748 | 0.347 | 2419.000 |
| Exp2 | state_volatility | 0.813 | 0.195 | 2419.000 |

Full versus expanded-ry path agreement:

| dataset | metric | spearman | median_absolute_display_difference | common_windows |
| --- | --- | --- | --- | --- |
| Exp1 | D_state | 0.980 | 0.216 | 8717.000 |
| Exp1 | V1000_norm | 0.975 | 0.055 | 8717.000 |
| Exp1 | multi_scale_rate_divergence | 0.980 | 0.072 | 8717.000 |
| Exp1 | state_volatility | 0.982 | 0.056 | 8717.000 |
| Exp2 | D_state | 0.984 | 0.044 | 2419.000 |
| Exp2 | V1000_norm | 0.977 | 0.173 | 2419.000 |
| Exp2 | multi_scale_rate_divergence | 0.975 | 0.128 | 2419.000 |
| Exp2 | state_volatility | 0.935 | 0.189 | 2419.000 |

Canonical b1000 Mahalanobis full-with-corrdist group contributions:

| dataset | metric | mean_contribution | p95_contribution | fraction_over_060 |
| --- | --- | --- | --- | --- |
| Exp1 | rx | 0.296 | 0.457 | 0.000 |
| Exp1 | ry | 0.466 | 0.748 | 0.335 |
| Exp1 | rs | 0.238 | 0.408 | 0.000 |
| Exp2 | rx | 0.287 | 0.482 | 0.000 |
| Exp2 | ry | 0.484 | 0.803 | 0.313 |
| Exp2 | rs | 0.230 | 0.401 | 0.000 |

Baseline-relative corrdist standardized-feature shares:

| dataset | metric | mean_absolute_standardised_share | p95_absolute_standardised_share | fraction_over_050 |
| --- | --- | --- | --- | --- |
| Exp1 | rx_corrdist_base | 0.767 | 0.963 | 0.843 |
| Exp1 | ry_corrdist_base | 0.603 | 0.909 | 0.602 |
| Exp1 | rs_corrdist_base | 0.741 | 0.949 | 0.818 |
| Exp2 | rx_corrdist_base | 0.390 | 0.824 | 0.216 |
| Exp2 | ry_corrdist_base | 0.832 | 0.991 | 0.924 |
| Exp2 | rs_corrdist_base | 0.551 | 0.877 | 0.650 |

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
- Test suite: **PASS** — ........................................................................ [ 54%]
............................................................             [100%]
============================== warnings summary ===============================
tests/test_csv1_no_stage_leakage.py: 6 warnings
tests/test_csv1_rank_direction.py: 7 warnings
tests/test_csv2_pre_refit_validation.py: 7 warnings
  D:\Program Files\Python313\Lib\site-packages\sklearn\linear_model\_logistic.py:1403: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
132 passed, 20 warnings in 35.26s
- Prefix causality: **PASS**, maximum pre-cutoff difference 0.000.
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
