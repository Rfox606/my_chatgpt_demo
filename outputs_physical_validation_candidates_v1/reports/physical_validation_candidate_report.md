# Physical validation candidate report

## Purpose

This run selects representative windows for FEM, surface morphology, and debris closed-loop validation. It does not tune AWR models.

## Main model

- Main AWR model: `M1_stable`.
- `M0_stable` remains the equal-weight baseline.
- `M2_stable` is kept as weighted sensitivity context, not as the main model.

## Candidate types

- `high_AWR_high_BD`: high late-state AWR form with clear baseline deviation.
- `high_AWR_low_BD`: local late-state form or short sensitive-phase change.
- `low_AWR_high_BD`: baseline deviation without high late-state AWR form.
- `AWR_rising`: rapid local AWR increase.
- `TES_high_confidence`: transition event neighborhood.
- `Exp1_late_stable_candidate`: Exp1 Stage5 stable late-state check.
- `Exp2_late_severe_candidate`: Exp2 Stage5 severe-state check.

## Candidate counts

| dataset | candidate_type | count |
| --- | --- | --- |
| Exp1 | AWR_rising | 5 |
| Exp1 | Exp1_late_stable_candidate | 5 |
| Exp1 | TES_high_confidence | 5 |
| Exp1 | high_AWR_high_BD | 5 |
| Exp1 | high_AWR_low_BD | 5 |
| Exp1 | low_AWR_high_BD | 5 |
| Exp2 | AWR_rising | 5 |
| Exp2 | Exp2_late_severe_candidate | 4 |
| Exp2 | TES_high_confidence | 4 |
| Exp2 | high_AWR_high_BD | 5 |
| Exp2 | high_AWR_low_BD | 5 |
| Exp2 | low_AWR_high_BD | 5 |

## Priority candidate list

| dataset | candidate_type | candidate_rank | center_cycle | stage | AWR | AWR_percentile_within_dataset | BDall_xy_v2 | BD_percentile_within_dataset | physical_validation_priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1 | TES_high_confidence | 1 | 4.14e+04 | 5 | 2.888 | 92.73 | 7.579 | 91.03 | High |
| Exp1 | TES_high_confidence | 2 | 3.463e+04 | 5 | 2.33 | 90.46 | 6.984 | 77.18 | High |
| Exp1 | TES_high_confidence | 3 | 1.06e+04 | 2 | 0.6746 | 66.96 | 5.922 | 61.97 | High |
| Exp1 | TES_high_confidence | 4 | 4.304e+04 | 5 | 3.392 | 96.96 | 7.673 | 91.68 | High |
| Exp1 | TES_high_confidence | 5 | 4.42e+04 | 5 | 3.075 | 94.17 | 6.981 | 77.08 | High |
| Exp1 | high_AWR_high_BD | 1 | 4.244e+04 | 5 | 3.671 | 99.37 | 7.49 | 90.05 | High |
| Exp1 | high_AWR_high_BD | 2 | 4.336e+04 | 5 | 3.416 | 97.24 | 7.653 | 91.55 | High |
| Exp1 | high_AWR_high_BD | 3 | 4.19e+04 | 5 | 3.473 | 97.63 | 7.388 | 88.78 | High |
| Exp1 | high_AWR_high_BD | 4 | 4.14e+04 | 5 | 2.923 | 92.99 | 7.585 | 91.09 | High |
| Exp1 | high_AWR_high_BD | 5 | 4.483e+04 | 5 | 2.78 | 92.13 | 7.589 | 91.16 | High |
| Exp2 | Exp2_late_severe_candidate | 1 | 1.265e+04 | 5 | 5.315 | 99.96 | 4.997 | 99.79 | High |
| Exp2 | Exp2_late_severe_candidate | 2 | 1.321e+04 | 5 | 4.956 | 99.04 | 4.687 | 98.19 | High |
| Exp2 | Exp2_late_severe_candidate | 3 | 1.399e+04 | 5 | 4.693 | 94.57 | 4.262 | 94.46 | High |
| Exp2 | Exp2_late_severe_candidate | 4 | 1.174e+04 | 5 | 4.903 | 98.3 | 4.099 | 89.99 | High |
| Exp2 | TES_high_confidence | 1 | 2730 | 1 | 1.264 | 46.65 | 1.384 | 28.75 | High |
| Exp2 | TES_high_confidence | 2 | 1.173e+04 | 5 | 4.908 | 98.37 | 4.068 | 89.28 | High |
| Exp2 | TES_high_confidence | 3 | 8726 | 4 | 3.078 | 75.68 | 3.09 | 76.43 | High |
| Exp2 | TES_high_confidence | 4 | 9916 | 4 | 4.291 | 86.23 | 3.964 | 88.46 | High |
| Exp2 | high_AWR_high_BD | 1 | 1.265e+04 | 5 | 5.315 | 99.96 | 4.997 | 99.79 | High |
| Exp2 | high_AWR_high_BD | 2 | 1.321e+04 | 5 | 4.956 | 99.04 | 4.687 | 98.19 | High |
| Exp2 | high_AWR_high_BD | 3 | 1.021e+04 | 4 | 4.883 | 97.98 | 4.291 | 94.82 | High |
| Exp2 | high_AWR_high_BD | 4 | 9606 | 4 | 4.451 | 90.13 | 4.958 | 99.33 | High |
| Exp2 | high_AWR_high_BD | 5 | 1.399e+04 | 5 | 4.693 | 94.57 | 4.262 | 94.46 | High |
| Exp1 | Exp1_late_stable_candidate | 1 | 4.097e+04 | 5 | -0.4059 | 40.3 | 6.514 | 66.8 | Medium |
| Exp1 | Exp1_late_stable_candidate | 2 | 3.877e+04 | 5 | -0.07169 | 49.73 | 7.054 | 79.57 | Medium |
| Exp1 | Exp1_late_stable_candidate | 3 | 4.022e+04 | 5 | 0.008525 | 52.33 | 6.748 | 69.22 | Medium |
| Exp1 | Exp1_late_stable_candidate | 4 | 3.961e+04 | 5 | 0.1409 | 57.02 | 7.671 | 91.66 | Medium |
| Exp1 | Exp1_late_stable_candidate | 5 | 3.825e+04 | 5 | 0.2042 | 58.97 | 7.045 | 79.17 | Medium |
| Exp1 | low_AWR_high_BD | 1 | 2.725e+04 | 3 | -1.912 | 21.51 | 8.087 | 99.95 | Medium |
| Exp1 | low_AWR_high_BD | 2 | 2.668e+04 | 3 | -1.788 | 24.55 | 8.069 | 99.63 | Medium |
| Exp1 | low_AWR_high_BD | 3 | 2.782e+04 | 3 | -1.897 | 21.67 | 7.988 | 97.17 | Medium |
| Exp1 | low_AWR_high_BD | 4 | 2.591e+04 | 3 | -2.071 | 19.25 | 7.949 | 95.85 | Medium |
| Exp1 | low_AWR_high_BD | 5 | 2.526e+04 | 3 | -1.579 | 30.62 | 8.131 | 99.98 | Medium |
| Exp2 | low_AWR_high_BD | 1 | 1.08e+04 | 4 | 2.541 | 67.77 | 4.471 | 96.66 | Medium |
| Exp2 | low_AWR_high_BD | 2 | 9296 | 4 | 2.57 | 68.26 | 3.486 | 81.29 | Medium |
| Exp2 | low_AWR_high_BD | 3 | 6900 | 3 | 0.8082 | 33.97 | 2.497 | 66.06 | Medium |
| Exp2 | low_AWR_high_BD | 4 | 1.14e+04 | 4 | 1.885 | 58.25 | 2.969 | 74.51 | Medium |
| Exp2 | low_AWR_high_BD | 5 | 8700 | 4 | 2.175 | 62.05 | 3.011 | 75.65 | Medium |
| Exp1 | AWR_rising | 1 | 7846 | 2 | 0.3033 | 61.58 | 4.091 | 28.39 | Medium-High |
| Exp1 | AWR_rising | 2 | 2.14e+04 | 3 | 1.123 | 81.96 | 5.238 | 35.71 | Medium-High |

## Channel / family contribution hints

| dataset | candidate_type | center_cycle | window_index | rx_contribution | ry_contribution | rs_contribution | corrdist_contribution | amplitude_contribution | dominant_channel | dominant_feature_family | interpretation_hint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1 | high_AWR_high_BD | 4.244e+04 | 8485 | 4.003 | 12 | 20.7 | 24 | 12.71 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | high_AWR_high_BD | 4.336e+04 | 8670 | 2.48 | 12 | 19.68 | 24 | 10.16 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | high_AWR_high_BD | 4.19e+04 | 8378 | 3.221 | 11.38 | 20.13 | 24 | 10.73 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | high_AWR_high_BD | 4.14e+04 | 8278 | 2.766 | 6.666 | 19.8 | 24 | 5.234 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | high_AWR_high_BD | 4.483e+04 | 8963 | 0.6024 | 10.62 | 16.58 | 24 | 3.796 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | high_AWR_low_BD | 3.448e+04 | 6893 | 16.2 | 6.108 | 18.32 | 24 | 16.63 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | high_AWR_low_BD | 10.5 | 0 | 12.31 | -0.1924 | 14.68 | 10.9 | 15.9 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | high_AWR_low_BD | 3.448e+04 | 6894 | 16.47 | 6.156 | 18.69 | 24 | 17.32 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | high_AWR_low_BD | 3.445e+04 | 6887 | 15.14 | 5.652 | 17.05 | 24 | 13.84 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | high_AWR_low_BD | 3.444e+04 | 6886 | 15 | 5.857 | 16.88 | 24 | 13.73 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | low_AWR_high_BD | 2.725e+04 | 5447 | -11.56 | 10.56 | -18.13 | 24 | -43.12 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | low_AWR_high_BD | 2.668e+04 | 5333 | -11.33 | 11.21 | -17.77 | 24 | -41.88 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | low_AWR_high_BD | 2.782e+04 | 5561 | -11.52 | 10.95 | -18.4 | 24 | -42.97 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | low_AWR_high_BD | 2.591e+04 | 5179 | -12.29 | 11.03 | -19.46 | 24 | -44.71 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | low_AWR_high_BD | 2.526e+04 | 5049 | -10.69 | 12 | -17.09 | 24 | -39.79 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | AWR_rising | 7846 | 1567 | 1.1 | 3.702 | -1.769 | 24 | -20.97 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | AWR_rising | 2.14e+04 | 4278 | 4.505 | 5.617 | 1.105 | 24 | -12.77 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | AWR_rising | 2.814e+04 | 5625 | 4.38 | 9.209 | 1.881 | 24 | -8.53 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | AWR_rising | 4.167e+04 | 8331 | 2.86 | 6.899 | 19.43 | 24 | 5.193 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | AWR_rising | 3.459e+04 | 6916 | 15.48 | 7.849 | 17.69 | 24 | 17.02 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | TES_high_confidence | 4.14e+04 | 8277 | 2.772 | 6.379 | 19.73 | 24 | 4.88 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | TES_high_confidence | 3.463e+04 | 6924 | 6.933 | 9.356 | 7.015 | 24 | -0.6962 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | TES_high_confidence | 1.06e+04 | 2118 | 1.629 | 5.394 | -0.2763 | 24 | -17.25 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | TES_high_confidence | 4.304e+04 | 8606 | 2.177 | 12 | 19.74 | 24 | 9.917 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | TES_high_confidence | 4.42e+04 | 8837 | 1.042 | 12 | 17.71 | 24 | 6.75 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | Exp1_late_stable_candidate | 4.097e+04 | 8192 | -2.867 | 4.739 | -5.932 | 24 | -28.06 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | Exp1_late_stable_candidate | 3.877e+04 | 7751 | -1.986 | 5.925 | -4.656 | 24 | -24.72 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | Exp1_late_stable_candidate | 4.022e+04 | 8042 | -1.211 | 4.92 | -3.624 | 24 | -23.91 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | Exp1_late_stable_candidate | 3.961e+04 | 7920 | -2.569 | 9.004 | -5.026 | 24 | -22.59 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | Exp1_late_stable_candidate | 3.825e+04 | 7647 | -1.301 | 6.943 | -3.6 | 24 | -21.96 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp2 | high_AWR_high_BD | 1.265e+04 | 2528 | 34.96 | 10.45 | 7.735 | 12.73 | 40.41 | rx | absmean | Dominant channel rx; dominant family absmean. |
| Exp2 | high_AWR_high_BD | 1.321e+04 | 2639 | 35.22 | 3.418 | 10.92 | 14.09 | 35.47 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_high_BD | 1.021e+04 | 2040 | 28.75 | 6.099 | 13.98 | 16.16 | 32.67 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_high_BD | 9606 | 1919 | 29.76 | 2.942 | 11.81 | 11.97 | 32.54 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_high_BD | 1.399e+04 | 2796 | 31.01 | 5.412 | 10.51 | 15.31 | 31.62 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_low_BD | 1.254e+04 | 2505 | 26.19 | 11.88 | 11.96 | 18.83 | 31.21 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_low_BD | 1.172e+04 | 2342 | 31.1 | 9.054 | 8.954 | 15.06 | 34.05 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_low_BD | 1.253e+04 | 2504 | 26.35 | 11.64 | 11.99 | 18.93 | 31.06 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_low_BD | 1.254e+04 | 2506 | 25.79 | 11.9 | 11.61 | 18.27 | 31.03 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_low_BD | 1.253e+04 | 2503 | 26.39 | 11.78 | 11.85 | 18.73 | 31.3 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |

## Warnings

- None.

## Physical-loop checks

- For high_AWR_high_BD windows, check surface damage, debris increase, and FEM high-contribution contact zones.
- For low_AWR_high_BD windows, check run-in, contact reorganization, or lubrication disturbance.
- For TES event windows, check whether transitions align with disassembly, lubrication change, measurement disturbance, or real force-signal jumps.
- For ry-dominant windows, check lateral contact migration or eccentric loading traces.