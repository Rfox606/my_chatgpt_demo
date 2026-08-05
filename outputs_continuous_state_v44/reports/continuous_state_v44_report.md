# Continuous state v4.4 — trajectory-first physical validation

## Scope and causal boundary

This run returns to the research question: can force features from the nominated sensitive gait interval support a stable, interpretable, online-updatable **continuous state**?  The state calculation uses cleaned effective-cycle order only, freezes each self-baseline before monitoring, and has no stage label, morphology value, stop/Guard flag, episode threshold, prediction module, or future-cycle value as an input.  Actual cycle is attached only after loading for plotting and post-hoc morphology alignment.

The configuration grid is fixed before this run: baseline lengths 500/1000/2000, Mahalanobis (Ledoit–Wolf) or diagonal distance, and full/no-rx/no-ry/no-rs feature variants.  The primary state quantities are D_state, V500, V1000, multi_scale_rate_divergence, and state_volatility.  All reported consensus quantities are configuration Q25/Q50/Q75/MAD and effective configuration count.

## Input-source traceability

`window_feature_z_table.csv` traceability result: **PASS**.  The existing feature-generation configuration and code identify Fx/Fy/Fz source files and a sensitive phase of 0.45–0.63.  The repository’s reproducible extraction indices are one-based 252–352 (Exp1) and 756–1058 (Exp2), while the study metadata records the requested nominal intervals 251–350 and 751–1050.  This off-by-discretisation difference is disclosed rather than silently harmonised.  Confirmed from existing generator/configuration; stage labels are excluded from v4.4 state input.

No row-level actual-cycle index was available in the state input.  The pre-existing segmented mapping in `outputs_physical_validation_candidates_v1/configs/cycle_mapping_config.json` is therefore the documented fallback, used only for coordinate restoration, figures, and post-hoc comparisons.

## 1. Which continuous indicators are most stable across configurations?

The table aggregates pairwise configuration comparisons over both experiments.  “High-change overlap” is the overlap of each pair’s fixed top decile, not a fitted event detector.

| metric | full_spearman_median | segment_agreement_median | high_change_overlap_median | comparisons |
| --- | --- | --- | --- | --- |
| D_state | 0.934 | 0.800 | 0.453 | 552.000 |
| multi_scale_rate_divergence | 0.913 | 1.000 | 0.630 | 552.000 |
| V1000_norm | 0.891 | 0.800 | 0.480 | 552.000 |
| V500_norm | 0.887 | 0.800 | 0.591 | 552.000 |
| state_volatility | 0.858 | 0.800 | 0.458 | 552.000 |

On this fixed grid, the strongest overall trajectory agreement is led by **D_state, multi_scale_rate_divergence, V1000_norm**.  This supports retaining the metric set as continuous outputs with uncertainty bands, rather than selecting a single morphology-fitted configuration.

## 2. Exp1 post-hoc correspondence with morphology intervals

The next table is a descriptive interval alignment, not a statistical validation: there are only six morphology-anchor intervals, and sparse Spearman values are exploratory only.  Morphology was read after state trajectories were final and never feeds feature selection or thresholds.

| start_cycle_actual | end_cycle_actual | D_cumulative_absolute_change | V500_mean | V1000_mean | rate_divergence_mean | volatility_mean | high_change_state_duration_fraction | delta_Sa | delta_Sq | delta_Sz | delta_Sku |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000 | 8000.000 | 138.310 | 0.120 | 0.087 | 0.369 | 0.474 | 0.166 | 0.555 | 0.514 | -11.210 | -0.273 |
| 8000.000 | 16000.000 | 79.344 | 0.211 | 0.181 | 0.394 | 0.523 | 0.145 | -0.835 | -0.893 | 14.735 | 3.028 |
| 16000.000 | 24000.000 | 180.280 | 0.191 | 0.112 | 0.532 | 0.589 | 0.255 | -0.809 | -0.792 | 15.777 | 2.156 |
| 24000.000 | 32000.000 | 99.100 | 0.169 | 0.125 | 0.337 | 0.442 | 0.123 | 1.255 | 1.591 | -20.157 | -1.999 |
| 32000.000 | 40000.000 | 79.753 | 0.141 | 0.103 | 0.361 | 0.439 | 0.104 | -1.542 | -1.878 | 8.529 | 4.569 |
| 40000.000 | 48000.000 | 103.342 | 0.226 | 0.146 | 0.463 | 0.561 | 0.239 | 0.399 | 0.440 | -12.627 | -4.005 |
| 48000.000 | 52988.763 | 69.420 | 0.221 | 0.186 | 0.426 | 0.562 | 0.402 | NA | NA | NA | NA |

The 24k–32k nominal smoothing interval has V500 mean 0.169.  The 32k–40k surface-fluctuation interval has rate-divergence mean 0.361 and volatility mean 0.439.  The 40k–48k anchored late interval has V500 mean 0.226.  These magnitudes permit a physically interpretable comparison, but do **not** establish a causal morphology model or prove a five-stage wear trajectory.

Interpretation of the interval results: relative to 16k-24k, the 24k-32k V500 level is lower; 32k-40k has slightly higher rate divergence, while volatility is nearly unchanged and marginally lower than 24k-32k.  However, 40k-48k V500 rises rather than remaining low.  The morphology correspondence is therefore partial, and the expected late low-rate pattern is not consistently supported by this state grid.

## 3. Exp2 compared with Exp1

Exp2 has lower median D_state but larger V500/V1000, divergence, and volatility medians than Exp1; its late D_state ratio is also higher.  It therefore shows a more persistently active late continuous pattern under its own frozen baseline, not a mapped copy of Exp1's morphology path.

Exp2 is intentionally not forced into Exp1’s morphology descriptions.  It is compared only by continuous deviation, long-run direction, late/early activity ratio, volatility, and multi-scale difference.

| dataset | metric | overall_median | effective_cycle_spearman | late_to_early_mean_ratio | late_high_value_fraction |
| --- | --- | --- | --- | --- | --- |
| Exp1 | D_state | 7.890 | 0.705 | 1.317 | 0.052 |
| Exp1 | V500 | 0.117 | 0.032 | 1.245 | 0.177 |
| Exp1 | V1000 | 0.084 | 0.047 | 1.240 | 0.139 |
| Exp1 | multi_scale_rate_divergence | 0.304 | 0.005 | 1.065 | 0.125 |
| Exp1 | state_volatility | 0.410 | -0.089 | 1.049 | 0.166 |
| Exp2 | D_state | 2.977 | 0.734 | 2.366 | 0.433 |
| Exp2 | V500 | 0.315 | 0.262 | 1.224 | 0.164 |
| Exp2 | V1000 | 0.224 | 0.435 | 1.482 | 0.153 |
| Exp2 | multi_scale_rate_divergence | 0.572 | 0.147 | 0.893 | 0.086 |
| Exp2 | state_volatility | 1.468 | 0.193 | 0.864 | 0.034 |

Any Exp2 differences in the table are differences in observed continuous signal patterns, not a shared wear stage or a common wear direction.

## 4. Does extended ry remain dominated by ry_p2p?

Direct ry conclusion: **ry_p2p is not the dominant expanded-group feature, but the added ry features do not form a strongly coherent signed within-group trend.**

| dataset | median_pairwise_signed_spearman | ry_p2p_mean_absolute_share | ry_p2p_p95_absolute_share | ry_p2p_share_over_050_fraction |
| --- | --- | --- | --- | --- |
| Exp1 | -0.182 | 0.233 | 0.373 | 0.000 |
| Exp2 | 0.011 | 0.070 | 0.163 | 0.000 |

The expanded group is assessed by within-group signed agreement, absolute standardized contribution shares, and its trajectory agreement with p2p-only and no-ry alternatives.  A high p2p share would be evidence of residual single-feature dominance; a low median pairwise signed correlation would indicate that adding ry features does not make a coherent common ry trend.  Neither outcome is used to alter this run’s state parameters.

## 5. Recommended minimal online output

Recommended minimum: **D_state, multi_scale_rate_divergence, V1000_norm**, each with configuration Q25/Q50/Q75/MAD and number of effective configurations.  D_state supplies deviation level; V500/V1000 or divergence supply time-scale-resolved change; volatility is retained when it has acceptable stability.  The mapping to actual cycle is a display/post-hoc field; the online calculation remains on effective cycle.

## 6. Does this move the continuous monitoring objective forward?

**Qualified PASS.** The workflow now tests stable continuous trajectories directly against a fixed configuration ensemble and reports physical interval correspondence without stage or morphology leakage.  It is closer to an online continuous-state monitor because baselines are frozen and every state row is prefix-causal.  It remains a signal-state study: sparse morphology anchors and the absence of a row-level actual-cycle source limit physical validation, and no claim of calibrated wear severity or failure probability is made.

## Tests and limitations

- Unit/integration test status: **PASS** (........................................................................ [ 56%]
.......................................................                  [100%]
============================== warnings summary ===============================
tests/test_csv1_no_stage_leakage.py: 6 warnings
tests/test_csv1_rank_direction.py: 7 warnings
tests/test_csv2_pre_refit_validation.py: 7 warnings
  D:\Program Files\Python313\Lib\site-packages\sklearn\linear_model\_logistic.py:1403: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
127 passed, 20 warnings in 19.02s).
- Prefix-causality replay: **PASS**; maximum pre-cutoff discrepancy = 0.000.
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
