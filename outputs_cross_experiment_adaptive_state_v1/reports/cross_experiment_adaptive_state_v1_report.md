# Cross-experiment adaptive degradation-progression monitoring v1

## Scope fixed before results

This is a cross-experiment transfer experiment: an ordered source-domain force-feature model supplies a nonzero target progression prior; only a bounded target residual is updated online.  The formal outputs are **progression_score**, **activity_score**, and **state_uncertainty**.  They are not absolute wear mass, volume, depth, percentage, remaining useful life, or clinical risk.

Stage, morphology (Sa/Sq/Sz/Sku), wear-debris fields, and future target rows are rejected at the formal boundary.  Cycle is used only to form historical time-order pairs, conduct prefix-causality replay, and index evaluation.  It is not a source or target model feature.  v4.5 D_state/V1000/difference/volatility are not the final score; this implementation independently calculates predeclared direct-force-feature transfer and treats local dynamics as activity/gating evidence.

## 1. Did source knowledge yield a nonzero initial prior?

| direction | dataset | entry_cycle | initial_prior | initial_adapted | initial_nonzero |
| --- | --- | --- | --- | --- | --- |
| Exp1_to_Exp2 | Exp2 | 0.000 | 0.900 | 0.900 | 1.000 |
| Exp1_to_Exp2 | Exp2 | 3000.000 | 0.900 | 0.900 | 1.000 |
| Exp1_to_Exp2 | Exp2 | 6000.000 | 0.899 | 0.899 | 1.000 |
| Exp1_to_Exp2 | Exp2 | 9000.000 | 0.909 | 0.909 | 1.000 |
| Exp2_to_Exp1 | Exp1 | 0.000 | 0.948 | 0.948 | 1.000 |
| Exp2_to_Exp1 | Exp1 | 8000.000 | 0.751 | 0.751 | 1.000 |
| Exp2_to_Exp1 | Exp1 | 16000.000 | 0.614 | 0.614 | 1.000 |
| Exp2_to_Exp1 | Exp1 | 24000.000 | 0.577 | 0.577 | 1.000 |
| Exp2_to_Exp1 | Exp1 | 32000.000 | 0.824 | 0.824 | 1.000 |

**Answer:** yes.  Initial scores are source-model priors and were not forced to zero; Target_Local remains a comparator rather than the final definition.

## 2–5. Online adaptation and comparator results

| direction | adaptive | source_static | target_local | elapsed | vs_source_static | vs_target_local | vs_elapsed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1_to_Exp2 | 0.561 | 0.540 | 0.638 | 1.000 | 1.000 | 0.000 | 0.000 |
| Exp2_to_Exp1 | 0.609 | 0.473 | 0.596 | 1.000 | 1.000 | 1.000 | 0.000 |

Metrics are target time-pair AUCs across fixed 500–1000, 1000–3000, and 3000–5000 cycle gaps.  They measure progression ranking, not absolute wear.  The delayed-entry replay emits the score before each possible adapter update, so it evaluates arrived-prefix adaptation rather than fitting future target data.

| direction | comparator | time_pair_auc | pairs |
| --- | --- | --- | --- |
| Exp1_to_Exp2 | Adaptive_Cross_Experiment | 0.561 | 9219.000 |
| Exp1_to_Exp2 | Elapsed_Time_Since_Entry | 1.000 | 9219.000 |
| Exp1_to_Exp2 | Source_Static | 0.540 | 9219.000 |
| Exp1_to_Exp2 | Target_Local | 0.638 | 9219.000 |
| Exp1_to_Exp2 | Target_Supervised_Oracle | NA | 0.000 |
| Exp2_to_Exp1 | Adaptive_Cross_Experiment | 0.609 | 12000.000 |
| Exp2_to_Exp1 | Elapsed_Time_Since_Entry | 1.000 | 12000.000 |
| Exp2_to_Exp1 | Source_Static | 0.473 | 12000.000 |
| Exp2_to_Exp1 | Target_Local | 0.596 | 12000.000 |
| Exp2_to_Exp1 | Target_Supervised_Oracle | NA | 0.000 |

## 6. Delayed-entry convergence

| direction | dataset | initial_nonzero | entry_prior_spearman | common_windows | convergence_mean_std | convergence_mean_abs_error_vs_entry0 |
| --- | --- | --- | --- | --- | --- | --- |
| Exp1_to_Exp2 | Exp2 | 1.000 | 0.400 | NA | NA | NA |
| Exp1_to_Exp2 | Exp2 | NA | NA | 1019.000 | 0.044 | 0.063 |
| Exp2_to_Exp1 | Exp1 | 1.000 | -0.400 | NA | NA | NA |
| Exp2_to_Exp1 | Exp1 | NA | NA | 2717.000 | 0.171 | 0.113 |

The common-suffix statistic compares every available delayed entry on identical later windows.  It does not turn elapsed time into an input score.

## 7. Distinct progression–activity paths

| direction | dataset | segment | progression_median | activity_median | uncertainty_median |
| --- | --- | --- | --- | --- | --- |
| Exp1_to_Exp2 | Exp2 | early | 0.900 | 0.505 | 0.344 |
| Exp1_to_Exp2 | Exp2 | middle | 0.918 | 0.556 | 0.576 |
| Exp1_to_Exp2 | Exp2 | late | NA | NA | NA |
| Exp2_to_Exp1 | Exp1 | early | 0.812 | 0.448 | 0.476 |
| Exp2_to_Exp1 | Exp1 | middle | 0.950 | 0.465 | 0.547 |
| Exp2_to_Exp1 | Exp1 | late | 1.000 | 0.623 | 0.481 |

**Answer:** progression and activity are separate output dimensions.  A later progression location with low activity is interpretable as relatively stable, while a later location with high activity remains actively changing; no common five-stage trajectory was imposed.

## 8. Shared-model feature evidence

| direction | source_dataset | feature | coefficient | absolute_coefficient | source_validation_time_pair_auc |
| --- | --- | --- | --- | --- | --- |
| Exp2_to_Exp1 | Exp2 | rs_rms | 7.357 | 7.357 | 0.602 |
| Exp2_to_Exp1 | Exp2 | rs_mean | -5.110 | 5.110 | 0.602 |
| Exp1_to_Exp2 | Exp1 | rs_rms | -4.655 | 4.655 | 0.316 |
| Exp1_to_Exp2 | Exp1 | rs_mean | 2.555 | 2.555 | 0.316 |
| Exp1_to_Exp2 | Exp1 | ry_absmean | 1.938 | 1.938 | 0.316 |
| Exp1_to_Exp2 | Exp1 | rx_absmean | 1.750 | 1.750 | 0.316 |
| Exp1_to_Exp2 | Exp1 | rx_mean | 1.739 | 1.739 | 0.316 |
| Exp1_to_Exp2 | Exp1 | ry_mean | 1.497 | 1.497 | 0.316 |
| Exp1_to_Exp2 | Exp1 | rx_q05 | -1.207 | 1.207 | 0.316 |
| Exp2_to_Exp1 | Exp2 | ry_absmean | -0.816 | 0.816 | 0.602 |
| Exp2_to_Exp1 | Exp2 | ry_mean | 0.816 | 0.816 | 0.602 |
| Exp1_to_Exp2 | Exp1 | ry_q05 | -0.669 | 0.669 | 0.316 |
| Exp2_to_Exp1 | Exp2 | ry_p2p | 0.648 | 0.648 | 0.602 |
| Exp2_to_Exp1 | Exp2 | rx_q05 | -0.416 | 0.416 | 0.602 |
| Exp2_to_Exp1 | Exp2 | ry_q05 | -0.361 | 0.361 | 0.602 |
| Exp1_to_Exp2 | Exp1 | ry_p2p | -0.112 | 0.112 | 0.316 |
| Exp2_to_Exp1 | Exp2 | rx_mean | -0.065 | 0.065 | 0.602 |
| Exp2_to_Exp1 | Exp2 | rx_absmean | -0.065 | 0.065 | 0.602 |

The listed features are predeclared direct force-ratio summaries; they were not selected from target Stage, morphology, debris, or delayed-entry results.

## 9–10. Adapter and uncertainty audit

| direction | adapter_update_reason | rows |
| --- | --- | --- |
| Exp1_to_Exp2 | cadence_not_reached | 4491.000 |
| Exp1_to_Exp2 | high_volatility_suppressed | 111.000 |
| Exp1_to_Exp2 | initialization_freeze | 798.000 |
| Exp1_to_Exp2 | ood_suppressed | 2220.000 |
| Exp1_to_Exp2 | updated | 51.000 |
| Exp1_to_Exp2 | updated_clipped | 3.000 |
| Exp2_to_Exp1 | cadence_not_reached | 27015.000 |
| Exp2_to_Exp1 | high_volatility_suppressed | 1293.000 |
| Exp2_to_Exp1 | initialization_freeze | 998.000 |
| Exp2_to_Exp1 | updated | 210.000 |
| Exp2_to_Exp1 | updated_clipped | 67.000 |

Uncertainty combines feature-configuration dispersion, source-support/OOD, arrived target-pair evidence, prior–adapted disagreement, adapter-boundary proximity, and local volatility/gating.  It is expected to rise during initialization, OOD, or a paused update.

## 11. Post-hoc Stage diagnostic

| status | reason | stage_progression_spearman | ordinal_mae | confusion_matrix | rows |
| --- | --- | --- | --- | --- | --- |
| NOT_AVAILABLE_INPUT_NOT_VERSIONED | Stage is absent from the formal v4.5 raw-window input and was not fetched for CEAP v1 training/adaptation | NA | NA | not available | 37257.000 |

Stage was unavailable in the versioned formal raw-window artifact; this is reported explicitly rather than importing it into the model.  If a separately governed label artifact is supplied later, it may be used only after formal inference for a non-primary diagnostic.

## 12. Decision

**FAIL** overall delivery status.  Scientific comparator status: **PASS**.

The acceptance rule was fixed before execution: both-direction improvement with nonzero delayed-entry priors is PASS; a one-direction or limited improvement is QUALIFIED PASS; time-only behaviour, no adaptation gain, or semantic drift is FAIL.  In addition, the fixed engineering minimum requires the complete test suite to pass.  That rule is not relaxed for a favourable scientific comparison.  No criterion above was changed after observing the result.

## Diagnostics and reproducibility

| prefix_causality_status | prefix_max_abs_difference | no_label_leakage_status | source_model_frozen_status | adapter_bounds_status | delayed_entry_nonzero_initialization_status | time_prior_audit_status | all_target_rows | all_update_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | 0.000 | PASS | PASS | PASS | PASS | PASS | 37257.000 | 37257.000 |

- Full pytest: **FAIL**.
- Prefix causality: **PASS**.
- Label/morphology/debris boundary: **PASS**.
- Frozen source model: **PASS**.
- Adapter bounds: **PASS**.
- Time-prior audit: **PASS**.

All directions, comparators, delayed entries, pauses, and unavailable diagnostics are retained in the CSV outputs; failures are not removed from this report.
