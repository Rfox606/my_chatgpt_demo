# Weighted AWR-core signal-layer interpretation report

## Purpose

This run rebuilds AWR-score as a transparent continuous state score before physical closed-loop validation. It keeps the existing M0 `stable_plus` mean-z AWR as a baseline and adds M1/M2/M3 to make feature direction, feature weight, and Fx/Fz versus Fy/Fz channel contribution explicit. It does not retrain Stage1-Stage5 classification, and Stage5 is used only as a late-state proxy label for external evaluation.

## Model definitions

- M0: existing `stable_plus` mean-z AWR read from the v2 state table.
- M1: direction-corrected mean-z AWR, using source-only `direction_sign`.
- M2: direction-corrected AWR weighted by effect size, bootstrap direction stability, and redundancy factor.
- M3: channel-constrained weighted AWR, with Hx from rx=Fx/Fz and Hy from ry=Fy/Fz. `M3_equal` is the transparent default candidate and `M3_weighted` is a source-validation fusion candidate.

## Direction and stability

- Direction rows: 56
- Stable direction rows with stability >= 0.70: 55
- Features entering M2: 55
- Features entering M3: 39
- Bootstrap: 500 block resamples, block size 20 windows.

## Feature weights

| direction_id | feature_name | channel | normalized_weight | direction_stability |
| --- | --- | --- | --- | --- |
| Exp2_to_Exp1 | ry_std | ry | 0.05434 | 1 |
| Exp2_to_Exp1 | rs_corrdist_base | rs | 0.05434 | 1 |
| Exp2_to_Exp1 | rx_corrdist_base | rx | 0.05434 | 1 |
| Exp2_to_Exp1 | rx_q95 | rx | 0.05434 | 1 |
| Exp2_to_Exp1 | rx_p2p | rx | 0.05434 | 1 |
| Exp2_to_Exp1 | rx_absmean | rx | 0.05434 | 1 |
| Exp2_to_Exp1 | rx_q05 | rx | 0.05434 | 1 |
| Exp2_to_Exp1 | ry_p2p | ry | 0.05434 | 1 |
| Exp2_to_Exp1 | rs_p2p | rs | 0.05434 | 1 |
| Exp2_to_Exp1 | rs_q95 | rs | 0.05434 | 1 |

## Bidirectional comparison

| model_name | mean_AUROC | worst_AUROC | mean_AUPRC | worst_AUPRC | mean_Spearman | worst_Spearman | interpretability_level | recommended_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 0.9435 | 0.8966 | 0.8257 | 0.7079 | 0.8138 | 0.8092 | baseline | stable_plus mean-z reference |
| M1 | 0.8186 | 0.715 | 0.5285 | 0.4341 | 0.661 | 0.5709 | direction-transparent | direction-corrected contrast |
| M2 | 0.7883 | 0.6735 | 0.4247 | 0.3105 | 0.6269 | 0.59 | weighted-transparent | effect/stability weighted contrast |
| M3_equal | 0.8413 | 0.7312 | 0.6442 | 0.5087 | 0.7368 | 0.7126 | channel-constrained | default interpretable Hx/Hy fusion |
| M3_weighted | 0.7208 | 0.4703 | 0.5256 | 0.2093 | 0.4794 | 0.1577 | channel-constrained candidate | source-validation channel fusion candidate |

Detailed target-side metrics:

| model_name | direction_id | source_dataset | target_dataset | target_AUROC | target_AUPRC | target_AUPRC_baseline | target_Spearman_stage_AWR | Stage1_median_AWR | Stage5_median_AWR | ScoreGap | Stage5_high_AWR_rate | Stage5_high_AWR_high_BD_occupancy | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9905 | 0.9435 | 0.1704 | 0.8092 | -0.8834 | 4.494 | 5.378 | 0.09375 | 0.09375 | Stage5 is used only as a late-state proxy label for evaluation. |
| M1 | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9222 | 0.6229 | 0.1704 | 0.7511 | -0.3654 | 1.231 | 1.597 | 0 | 0 | Stage5 is used only as a late-state proxy label for evaluation. |
| M2 | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9031 | 0.5388 | 0.1704 | 0.6638 | -0.3076 | 0.9888 | 1.296 | 0 | 0 | Stage5 is used only as a late-state proxy label for evaluation. |
| M3_equal | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9514 | 0.7796 | 0.1704 | 0.7611 | -0.3387 | 1.72 | 2.059 | 0 | 0 | Stage5 is used only as a late-state proxy label for evaluation. |
| M3_weighted | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9713 | 0.8419 | 0.1704 | 0.8011 | -0.3486 | 2.087 | 2.436 | 0 | 0 | Stage5 is used only as a late-state proxy label for evaluation. |
| M0 | Exp2_to_Exp1 | Exp2 | Exp1 | 0.8966 | 0.7079 | 0.2413 | 0.8185 | -4.177 | 2.564 | 6.742 | 0.2847 | 0.2838 | Stage5 is used only as a late-state proxy label for evaluation. |
| M1 | Exp2_to_Exp1 | Exp2 | Exp1 | 0.715 | 0.4341 | 0.2413 | 0.5709 | -1.93 | 0.1152 | 2.045 | 0 | 0 | Stage5 is used only as a late-state proxy label for evaluation. |
| M2 | Exp2_to_Exp1 | Exp2 | Exp1 | 0.6735 | 0.3105 | 0.2413 | 0.59 | -1.682 | 0.4638 | 2.145 | 0 | 0 | Stage5 is used only as a late-state proxy label for evaluation. |
| M3_equal | Exp2_to_Exp1 | Exp2 | Exp1 | 0.7312 | 0.5087 | 0.2413 | 0.7126 | -1.709 | 1.692 | 3.401 | 0 | 0 | Stage5 is used only as a late-state proxy label for evaluation. |
| M3_weighted | Exp2_to_Exp1 | Exp2 | Exp1 | 0.4703 | 0.2093 | 0.2413 | 0.1577 | -2.756 | -2.229 | 0.5279 | 0 | 0 | Stage5 is used only as a late-state proxy label for evaluation. |

## Selected model

Selected model: **M0**

Reason: Weighted variants did not satisfy the conservative worst-direction guardrail against M0.

The selection uses worst-direction behavior, interpretability, and channel traceability. It is not a search for the highest one-direction score.

## State occupancy and TES

Stage5 high_AWR_high_BD occupancy rows:

| direction_id | dataset | group_type | group_value | state_region | count | rate | cycle_start | cycle_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1_to_Exp2 | Exp1 | stage | 5 | high_AWR_high_BD | 160 | 0.07276 | 3.458e+04 | 4.559e+04 |
| Exp1_to_Exp2 | Exp2 | stage | 5 | high_AWR_high_BD | 45 | 0.09375 | 1.169e+04 | 1.41e+04 |
| Exp2_to_Exp1 | Exp1 | stage | 5 | high_AWR_high_BD | 624 | 0.2838 | 3.458e+04 | 4.559e+04 |
| Exp2_to_Exp1 | Exp2 | stage | 5 | high_AWR_high_BD | 282 | 0.5875 | 1.169e+04 | 1.41e+04 |

TES events detected with selected AWR: 56, high-confidence events: 15.

## Dependency boundary

BD v2 is reused as an AWR-independent baseline deviation layer. RS depends on selected AWR trend, and TES partly depends on selected AWR volatility, so RS/TES should be interpreted as derived signal-layer descriptors rather than independent physical evidence. FEM/contact morphology/debris observations remain the next external validation layer.

## Current conclusion boundary

The outputs describe continuous signal-state structure and channel-level AWR contributions. They are not wear-depth prediction, not a failure-warning result, and not a replacement for physical closed-loop validation.
