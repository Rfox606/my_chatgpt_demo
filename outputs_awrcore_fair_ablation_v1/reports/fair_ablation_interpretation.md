# Fair AWR ablation interpretation

## Purpose

This run performs a fair ablation of the AWR score. It does not rebuild Stage1-Stage5 classification and does not change the research target. Stage5 is used only as a late-state proxy label for target-side evaluation.

## Main answers

1. Best stable_plus model: **M0_stable**.
2. M0_stable remains the best stable_plus model: **True**.
3. M1_stable is close to M0_stable under the worst-direction guardrail: **True**.
4. M2_stable shows incremental gain over M1_stable: **False**.
5. Corrdist dependency: **weak**.
6. Best channel/family model: **ry_only_direction_mean_z**.
7. Recommended AWR structure: Keep stable_plus as the main score family and use channel/family ablations for interpretation.

## Stable Plus Ablation

| model_name | feature_group | formulation | direction_id | source_dataset | target_dataset | target_AUROC | target_AUPRC | AUPRC_baseline | Spearman_stage_AWR | Stage1_median_AWR | Stage5_median_AWR | ScoreGap | source_threshold_P95 | target_Stage5_high_AWR_rate | target_Stage5_high_AWR_high_BD_occupancy | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0_stable | stable_plus | mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9906 | 0.943 | 0.1704 | 0.8237 | -0.8834 | 4.494 | 5.378 | 3.482 | 0.9958 | 0.9958 | Stage5 is used only as a late-state proxy label. |
| M1_stable | stable_plus | direction_mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9906 | 0.943 | 0.1704 | 0.8237 | -0.8834 | 4.494 | 5.378 | 3.482 | 0.9958 | 0.9958 | Stage5 is used only as a late-state proxy label. |
| M2_stable | stable_plus | weighted_direction | Exp1_to_Exp2 | Exp1 | Exp2 | 0.978 | 0.8795 | 0.1704 | 0.8349 | -0.862 | 4.583 | 5.445 | 5.02 | 0.2979 | 0.2979 | Stage5 is used only as a late-state proxy label. |
| M0_stable | stable_plus | mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.8443 | 0.619 | 0.2413 | 0.7777 | -4.177 | 0.9642 | 5.142 | 4.411 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| M1_stable | stable_plus | direction_mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.8443 | 0.619 | 0.2413 | 0.7777 | -4.177 | 0.9642 | 5.142 | 4.411 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| M2_stable | stable_plus | weighted_direction | Exp2_to_Exp1 | Exp2 | Exp1 | 0.8428 | 0.5957 | 0.2413 | 0.8052 | -3.349 | 2.397 | 5.746 | 5.049 | 0 | 0 | Stage5 is used only as a late-state proxy label. |

## Corrdist Ablation

| model_name | feature_group | formulation | direction_id | source_dataset | target_dataset | target_AUROC | target_AUPRC | AUPRC_baseline | Spearman_stage_AWR | Stage1_median_AWR | Stage5_median_AWR | ScoreGap | source_threshold_P95 | target_Stage5_high_AWR_rate | target_Stage5_high_AWR_high_BD_occupancy | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0_stable | stable_plus | mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9906 | 0.943 | 0.1704 | 0.8237 | -0.8834 | 4.494 | 5.378 | 3.482 | 0.9958 | 0.9958 | Stage5 is used only as a late-state proxy label. |
| M1_stable | stable_plus | direction_mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9906 | 0.943 | 0.1704 | 0.8237 | -0.8834 | 4.494 | 5.378 | 3.482 | 0.9958 | 0.9958 | Stage5 is used only as a late-state proxy label. |
| M2_stable | stable_plus | weighted_direction | Exp1_to_Exp2 | Exp1 | Exp2 | 0.978 | 0.8795 | 0.1704 | 0.8349 | -0.862 | 4.583 | 5.445 | 5.02 | 0.2979 | 0.2979 | Stage5 is used only as a late-state proxy label. |
| M0_stable_no_corrdist | stable_plus_no_corrdist | mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9958 | 0.9713 | 0.1704 | 0.8127 | -1.482 | 3.879 | 5.362 | 1.353 | 1 | 1 | Stage5 is used only as a late-state proxy label. |
| M1_stable_no_corrdist | stable_plus_no_corrdist | direction_mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9958 | 0.9713 | 0.1704 | 0.8127 | -1.482 | 3.879 | 5.362 | 1.353 | 1 | 1 | Stage5 is used only as a late-state proxy label. |
| M2_stable_no_corrdist | stable_plus_no_corrdist | weighted_direction | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9972 | 0.9845 | 0.1704 | 0.8418 | -1.573 | 4.002 | 5.574 | 2.591 | 0.9979 | 0.9979 | Stage5 is used only as a late-state proxy label. |
| M0_stable | stable_plus | mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.8443 | 0.619 | 0.2413 | 0.7777 | -4.177 | 0.9642 | 5.142 | 4.411 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| M1_stable | stable_plus | direction_mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.8443 | 0.619 | 0.2413 | 0.7777 | -4.177 | 0.9642 | 5.142 | 4.411 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| M2_stable | stable_plus | weighted_direction | Exp2_to_Exp1 | Exp2 | Exp1 | 0.8428 | 0.5957 | 0.2413 | 0.8052 | -3.349 | 2.397 | 5.746 | 5.049 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| M0_stable_no_corrdist | stable_plus_no_corrdist | mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.827 | 0.5902 | 0.2413 | 0.6897 | -5.619 | -1.795 | 3.824 | 3.882 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| M1_stable_no_corrdist | stable_plus_no_corrdist | direction_mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.827 | 0.5902 | 0.2413 | 0.6897 | -5.619 | -1.795 | 3.824 | 3.882 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| M2_stable_no_corrdist | stable_plus_no_corrdist | weighted_direction | Exp2_to_Exp1 | Exp2 | Exp1 | 0.8234 | 0.5672 | 0.2413 | 0.7107 | -5.144 | -1.156 | 3.988 | 4.522 | 0 | 0 | Stage5 is used only as a late-state proxy label. |

## Channel / Feature Family Ablation

| model_name | feature_group | formulation | direction_id | source_dataset | target_dataset | target_AUROC | target_AUPRC | AUPRC_baseline | Spearman_stage_AWR | Stage1_median_AWR | Stage5_median_AWR | ScoreGap | source_threshold_P95 | target_Stage5_high_AWR_rate | target_Stage5_high_AWR_high_BD_occupancy | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0_stable | stable_plus | mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9906 | 0.943 | 0.1704 | 0.8237 | -0.8834 | 4.494 | 5.378 | 3.482 | 0.9958 | 0.9958 | Stage5 is used only as a late-state proxy label. |
| M1_stable | stable_plus | direction_mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9906 | 0.943 | 0.1704 | 0.8237 | -0.8834 | 4.494 | 5.378 | 3.482 | 0.9958 | 0.9958 | Stage5 is used only as a late-state proxy label. |
| M2_stable | stable_plus | weighted_direction | Exp1_to_Exp2 | Exp1 | Exp2 | 0.978 | 0.8795 | 0.1704 | 0.8349 | -0.862 | 4.583 | 5.445 | 5.02 | 0.2979 | 0.2979 | Stage5 is used only as a late-state proxy label. |
| rx_only_mean_z | rx_only | mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9647 | 0.8591 | 0.1704 | 0.7737 | -0.6517 | 5.208 | 5.86 | 1.195 | 1 | 1 | Stage5 is used only as a late-state proxy label. |
| rx_only_direction_mean_z | rx_only | direction_mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9399 | 0.6165 | 0.1704 | 0.693 | -0.7237 | 2.896 | 3.62 | 2.645 | 0.8292 | 0.8292 | Stage5 is used only as a late-state proxy label. |
| rx_only_weighted_direction | rx_only | weighted_direction | Exp1_to_Exp2 | Exp1 | Exp2 | 0.6873 | 0.2543 | 0.1704 | 0.3765 | -0.2853 | 0.8332 | 1.118 | 6.232 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| ry_only_mean_z | ry_only | mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9596 | 0.8973 | 0.1704 | 0.7375 | 0.02128 | 2.484 | 2.463 | 1.48 | 0.8458 | 0.8458 | Stage5 is used only as a late-state proxy label. |
| ry_only_direction_mean_z | ry_only | direction_mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9438 | 0.7907 | 0.1704 | 0.7232 | -0.2745 | 2.353 | 2.628 | 6.868 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| ry_only_weighted_direction | ry_only | weighted_direction | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9685 | 0.8229 | 0.1704 | 0.7899 | -0.3557 | 2.358 | 2.714 | 9.165 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rs_only_mean_z | rs_only | mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9759 | 0.8338 | 0.1704 | 0.8878 | -0.3668 | 3.153 | 3.52 | 0.6941 | 1 | 1 | Stage5 is used only as a late-state proxy label. |
| rs_only_direction_mean_z | rs_only | direction_mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.1176 | 0.09606 | 0.1704 | -0.5392 | -0.1798 | -0.8694 | -0.6896 | 4.293 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rs_only_weighted_direction | rs_only | weighted_direction | Exp1_to_Exp2 | Exp1 | Exp2 | 0.1159 | 0.09582 | 0.1704 | -0.5467 | -0.1262 | -1.034 | -0.9076 | 5.095 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rx_ry_mean_z | rx_ry | mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9963 | 0.9799 | 0.1704 | 0.868 | -0.2393 | 3.595 | 3.835 | 0.8209 | 1 | 1 | Stage5 is used only as a late-state proxy label. |
| rx_ry_direction_mean_z | rx_ry | direction_mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9779 | 0.9141 | 0.1704 | 0.8324 | -0.438 | 2.722 | 3.16 | 4.047 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rx_ry_weighted_direction | rx_ry | weighted_direction | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9494 | 0.771 | 0.1704 | 0.7522 | -0.2578 | 1.507 | 1.764 | 6.45 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rx_ry_rs_mean_z | rx_ry_rs | mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9966 | 0.9814 | 0.1704 | 0.8812 | -0.2825 | 3.468 | 3.75 | 0.6851 | 1 | 1 | Stage5 is used only as a late-state proxy label. |
| rx_ry_rs_direction_mean_z | rx_ry_rs | direction_mean_z | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9459 | 0.6969 | 0.1704 | 0.7734 | -0.4156 | 1.602 | 2.018 | 4.101 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rx_ry_rs_weighted_direction | rx_ry_rs | weighted_direction | Exp1_to_Exp2 | Exp1 | Exp2 | 0.9033 | 0.5395 | 0.1704 | 0.6641 | -0.308 | 0.9911 | 1.299 | 5.829 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| M0_stable | stable_plus | mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.8443 | 0.619 | 0.2413 | 0.7777 | -4.177 | 0.9642 | 5.142 | 4.411 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| M1_stable | stable_plus | direction_mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.8443 | 0.619 | 0.2413 | 0.7777 | -4.177 | 0.9642 | 5.142 | 4.411 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| M2_stable | stable_plus | weighted_direction | Exp2_to_Exp1 | Exp2 | Exp1 | 0.8428 | 0.5957 | 0.2413 | 0.8052 | -3.349 | 2.397 | 5.746 | 5.049 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rx_only_mean_z | rx_only | mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.4483 | 0.2216 | 0.2413 | -0.04662 | -1.295 | -3.342 | -2.047 | 5.276 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rx_only_direction_mean_z | rx_only | direction_mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.3052 | 0.1665 | 0.2413 | -0.172 | -2.417 | -3.226 | -0.8083 | 5.433 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rx_only_weighted_direction | rx_only | weighted_direction | Exp2_to_Exp1 | Exp2 | Exp1 | 0.424 | 0.1971 | 0.2413 | 0.0431 | -2.531 | -2.567 | -0.03537 | 5.184 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| ry_only_mean_z | ry_only | mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.5448 | 0.5765 | 0.2413 | -0.1175 | 1.179 | 0.8651 | -0.3141 | 2.421 | 0.3624 | 0.362 | Stage5 is used only as a late-state proxy label. |
| ry_only_direction_mean_z | ry_only | direction_mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.8809 | 0.7473 | 0.2413 | 0.6274 | -1.048 | 5.249 | 6.297 | 2.578 | 1 | 0.9991 | Stage5 is used only as a late-state proxy label. |
| ry_only_weighted_direction | ry_only | weighted_direction | Exp2_to_Exp1 | Exp2 | Exp1 | 0.746 | 0.6031 | 0.2413 | 0.7376 | -0.7215 | 4.965 | 5.686 | 4.131 | 0.759 | 0.7581 | Stage5 is used only as a late-state proxy label. |
| rs_only_mean_z | rs_only | mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.515 | 0.3512 | 0.2413 | 0.03654 | -2.598 | -3.805 | -1.206 | 3.262 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rs_only_direction_mean_z | rs_only | direction_mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.515 | 0.3512 | 0.2413 | 0.03654 | -2.598 | -3.805 | -1.206 | 3.262 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rs_only_weighted_direction | rs_only | weighted_direction | Exp2_to_Exp1 | Exp2 | Exp1 | 0.4638 | 0.2403 | 0.2413 | -0.006319 | -1.442 | -3.241 | -1.799 | 4.045 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rx_ry_mean_z | rx_ry | mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.5315 | 0.2288 | 0.2413 | 0.0363 | -0.054 | -0.465 | -0.411 | 3.85 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rx_ry_direction_mean_z | rx_ry | direction_mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.7813 | 0.6137 | 0.2413 | 0.6435 | -1.689 | 1.351 | 3.04 | 4.008 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rx_ry_weighted_direction | rx_ry | weighted_direction | Exp2_to_Exp1 | Exp2 | Exp1 | 0.7077 | 0.4131 | 0.2413 | 0.6807 | -1.671 | 1.351 | 3.022 | 4.686 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rx_ry_rs_mean_z | rx_ry_rs | mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.5376 | 0.2412 | 0.2413 | 0.06709 | -0.7315 | -1.314 | -0.5829 | 3.672 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rx_ry_rs_direction_mean_z | rx_ry_rs | direction_mean_z | Exp2_to_Exp1 | Exp2 | Exp1 | 0.715 | 0.4341 | 0.2413 | 0.5709 | -1.93 | 0.1152 | 2.045 | 3.787 | 0 | 0 | Stage5 is used only as a late-state proxy label. |
| rx_ry_rs_weighted_direction | rx_ry_rs | weighted_direction | Exp2_to_Exp1 | Exp2 | Exp1 | 0.6841 | 0.3184 | 0.2413 | 0.616 | -1.648 | 0.6726 | 2.32 | 4.507 | 0 | 0 | Stage5 is used only as a late-state proxy label. |

## Overall Model Comparison

| model_name | feature_group | formulation | mean_AUROC | worst_AUROC | mean_AUPRC | worst_AUPRC | mean_Spearman | worst_Spearman | mean_ScoreGap | worst_ScoreGap | mean_Stage5_high_AWR_high_BD_occupancy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ry_only_direction_mean_z | ry_only | direction_mean_z | 0.9123 | 0.8809 | 0.769 | 0.7473 | 0.6753 | 0.6274 | 4.463 | 2.628 | 0.4995 |
| M0_stable | stable_plus | mean_z | 0.9174 | 0.8443 | 0.781 | 0.619 | 0.8007 | 0.7777 | 5.26 | 5.142 | 0.4979 |
| M1_stable | stable_plus | direction_mean_z | 0.9174 | 0.8443 | 0.781 | 0.619 | 0.8007 | 0.7777 | 5.26 | 5.142 | 0.4979 |
| M2_stable | stable_plus | weighted_direction | 0.9104 | 0.8428 | 0.7376 | 0.5957 | 0.82 | 0.8052 | 5.595 | 5.445 | 0.149 |
| M0_stable_no_corrdist | stable_plus_no_corrdist | mean_z | 0.9114 | 0.827 | 0.7808 | 0.5902 | 0.7512 | 0.6897 | 4.593 | 3.824 | 0.5 |
| M1_stable_no_corrdist | stable_plus_no_corrdist | direction_mean_z | 0.9114 | 0.827 | 0.7808 | 0.5902 | 0.7512 | 0.6897 | 4.593 | 3.824 | 0.5 |
| M2_stable_no_corrdist | stable_plus_no_corrdist | weighted_direction | 0.9103 | 0.8234 | 0.7758 | 0.5672 | 0.7763 | 0.7107 | 4.781 | 3.988 | 0.499 |
| rx_ry_direction_mean_z | rx_ry | direction_mean_z | 0.8796 | 0.7813 | 0.7639 | 0.6137 | 0.738 | 0.6435 | 3.1 | 3.04 | 0 |
| ry_only_weighted_direction | ry_only | weighted_direction | 0.8573 | 0.746 | 0.713 | 0.6031 | 0.7637 | 0.7376 | 4.2 | 2.714 | 0.379 |
| rx_ry_rs_direction_mean_z | rx_ry_rs | direction_mean_z | 0.8304 | 0.715 | 0.5655 | 0.4341 | 0.6722 | 0.5709 | 2.032 | 2.018 | 0 |
| rx_ry_weighted_direction | rx_ry | weighted_direction | 0.8286 | 0.7077 | 0.5921 | 0.4131 | 0.7164 | 0.6807 | 2.393 | 1.764 | 0 |
| rx_ry_rs_weighted_direction | rx_ry_rs | weighted_direction | 0.7937 | 0.6841 | 0.429 | 0.3184 | 0.64 | 0.616 | 1.81 | 1.299 | 0 |
| ry_only_mean_z | ry_only | mean_z | 0.7522 | 0.5448 | 0.7369 | 0.5765 | 0.31 | -0.1175 | 1.074 | -0.3141 | 0.6039 |
| rx_ry_rs_mean_z | rx_ry_rs | mean_z | 0.7671 | 0.5376 | 0.6113 | 0.2412 | 0.4742 | 0.06709 | 1.584 | -0.5829 | 0.5 |
| rx_ry_mean_z | rx_ry | mean_z | 0.7639 | 0.5315 | 0.6044 | 0.2288 | 0.4521 | 0.0363 | 1.712 | -0.411 | 0.5 |
| rs_only_mean_z | rs_only | mean_z | 0.7454 | 0.515 | 0.5925 | 0.3512 | 0.4622 | 0.03654 | 1.157 | -1.206 | 0.5 |
| rx_only_mean_z | rx_only | mean_z | 0.7065 | 0.4483 | 0.5403 | 0.2216 | 0.3635 | -0.04662 | 1.906 | -2.047 | 0.5 |
| rx_only_weighted_direction | rx_only | weighted_direction | 0.5556 | 0.424 | 0.2257 | 0.1971 | 0.2098 | 0.0431 | 0.5415 | -0.03537 | 0 |
| rx_only_direction_mean_z | rx_only | direction_mean_z | 0.6226 | 0.3052 | 0.3915 | 0.1665 | 0.2605 | -0.172 | 1.406 | -0.8083 | 0.4146 |
| rs_only_direction_mean_z | rs_only | direction_mean_z | 0.3163 | 0.1176 | 0.2236 | 0.09606 | -0.2513 | -0.5392 | -0.9479 | -1.206 | 0 |
| rs_only_weighted_direction | rs_only | weighted_direction | 0.2898 | 0.1159 | 0.1681 | 0.09582 | -0.2765 | -0.5467 | -1.353 | -1.799 | 0 |

## Interpretation Boundary

AUROC/AUPRC describe target-side ranking ability with Stage5 as a late-state proxy label. The high_AWR_high_BD occupancy describes how a source-only high-AWR state threshold transfers into the target dataset when combined with BD v2. These two outputs should be interpreted separately.

BD is reused from v2 as the AWR-independent baseline deviation layer. The ablation results are signal-layer evidence only; FEM/contact morphology/debris evidence remains the next physical closed-loop validation layer.
