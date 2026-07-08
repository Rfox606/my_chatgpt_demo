# Physical validation candidate report

## Purpose

This run selects representative windows for FEM, surface morphology, and debris closed-loop validation. It does not tune AWR models.

## Main model

- Main AWR model: `M1_stable`.
- `M0_stable` remains the equal-weight baseline.
- `M2_stable` is kept as weighted sensitivity context, not as the main model.

## Cycle mapping

Candidate windows keep the original effective-cycle fields and add actual experimental-cycle fields.
Mapping uses piecewise linear interpolation inside each user-provided stage. For Exp2, actual cycles 1-500 are NaN, so effective cycle 1 maps to actual cycle 501.

| dataset | stage | effective_start | effective_end | actual_start | actual_end | slope_actual_per_effective | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1 | 1 | 1 | 7575 | 1 | 8000 | 1.056 | Stage 1 piecewise mapping |
| Exp1 | 2 | 7575 | 2.112e+04 | 8000 | 2.4e+04 | 1.181 | Stage 2 piecewise mapping |
| Exp1 | 3 | 2.112e+04 | 2.784e+04 | 2.4e+04 | 3.2e+04 | 1.191 | Stage 3 piecewise mapping |
| Exp1 | 4 | 2.784e+04 | 3.46e+04 | 3.2e+04 | 4e+04 | 1.183 | Stage 4 piecewise mapping |
| Exp1 | 5 | 3.46e+04 | 4.559e+04 | 4e+04 | 5.3e+04 | 1.183 | Stage 5 piecewise mapping |
| Exp2 | 1 | 1 | 3005 | 501 | 5500 | 1.664 | Stage 1 piecewise mapping; actual cycles 1-500 are NaN |
| Exp2 | 2 | 3005 | 6005 | 5500 | 1.05e+04 | 1.667 | Stage 2 piecewise mapping |
| Exp2 | 3 | 6005 | 8705 | 1.05e+04 | 1.5e+04 | 1.667 | Stage 3 piecewise mapping |
| Exp2 | 4 | 8705 | 1.17e+04 | 1.5e+04 | 2e+04 | 1.667 | Stage 4 piecewise mapping |
| Exp2 | 5 | 1.17e+04 | 1.41e+04 | 2e+04 | 2.4e+04 | 1.67 | Stage 5 piecewise mapping |

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
| Exp1 | high_AWR_low_BD | 3 |
| Exp1 | low_AWR_high_BD | 5 |
| Exp2 | AWR_rising | 5 |
| Exp2 | Exp2_late_severe_candidate | 5 |
| Exp2 | TES_high_confidence | 4 |
| Exp2 | high_AWR_high_BD | 5 |
| Exp2 | high_AWR_low_BD | 5 |
| Exp2 | low_AWR_high_BD | 5 |

## Priority candidate list

| dataset | candidate_type | candidate_rank | center_cycle_effective | center_cycle_actual | stage | AWR | AWR_percentile_within_dataset | BDall_xy_v2 | BD_percentile_within_dataset | physical_validation_priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1 | TES_high_confidence | 1 | 4.14e+04 | 4.804e+04 | 5 | 2.888 | 92.73 | 7.579 | 91.03 | High |
| Exp1 | TES_high_confidence | 2 | 3.463e+04 | 4.004e+04 | 5 | 2.33 | 90.46 | 6.984 | 77.18 | High |
| Exp1 | TES_high_confidence | 3 | 1.06e+04 | 1.157e+04 | 2 | 0.6746 | 66.96 | 5.922 | 61.97 | High |
| Exp1 | TES_high_confidence | 4 | 4.304e+04 | 4.998e+04 | 5 | 3.392 | 96.96 | 7.673 | 91.68 | High |
| Exp1 | TES_high_confidence | 5 | 4.42e+04 | 5.135e+04 | 5 | 3.075 | 94.17 | 6.981 | 77.08 | High |
| Exp1 | high_AWR_high_BD | 1 | 4.244e+04 | 4.927e+04 | 5 | 3.671 | 99.37 | 7.49 | 90.05 | High |
| Exp1 | high_AWR_high_BD | 2 | 4.336e+04 | 5.036e+04 | 5 | 3.416 | 97.24 | 7.653 | 91.55 | High |
| Exp1 | high_AWR_high_BD | 3 | 4.19e+04 | 4.864e+04 | 5 | 3.473 | 97.63 | 7.388 | 88.78 | High |
| Exp1 | high_AWR_high_BD | 4 | 4.145e+04 | 4.81e+04 | 5 | 2.983 | 93.45 | 7.621 | 91.39 | High |
| Exp1 | high_AWR_high_BD | 5 | 4.483e+04 | 5.21e+04 | 5 | 2.78 | 92.13 | 7.589 | 91.16 | High |
| Exp2 | Exp2_late_severe_candidate | 1 | 1.265e+04 | 2.158e+04 | 5 | 5.315 | 99.96 | 4.997 | 99.79 | High |
| Exp2 | Exp2_late_severe_candidate | 2 | 1.321e+04 | 2.251e+04 | 5 | 4.956 | 99.04 | 4.687 | 98.19 | High |
| Exp2 | Exp2_late_severe_candidate | 3 | 1.399e+04 | 2.382e+04 | 5 | 4.693 | 94.57 | 4.262 | 94.46 | High |
| Exp2 | Exp2_late_severe_candidate | 4 | 1.174e+04 | 2.005e+04 | 5 | 4.903 | 98.3 | 4.099 | 89.99 | High |
| Exp2 | Exp2_late_severe_candidate | 5 | 1.367e+04 | 2.328e+04 | 5 | 4.386 | 88.71 | 4.262 | 94.39 | High |
| Exp2 | TES_high_confidence | 1 | 2730 | 5043 | 1 | 1.264 | 46.65 | 1.384 | 28.75 | High |
| Exp2 | TES_high_confidence | 2 | 1.173e+04 | 2.004e+04 | 5 | 4.908 | 98.37 | 4.068 | 89.28 | High |
| Exp2 | TES_high_confidence | 3 | 8726 | 1.503e+04 | 4 | 3.078 | 75.68 | 3.09 | 76.43 | High |
| Exp2 | TES_high_confidence | 4 | 9916 | 1.702e+04 | 4 | 4.291 | 86.23 | 3.964 | 88.46 | High |
| Exp2 | high_AWR_high_BD | 1 | 1.265e+04 | 2.158e+04 | 5 | 5.315 | 99.96 | 4.997 | 99.79 | High |
| Exp2 | high_AWR_high_BD | 2 | 1.321e+04 | 2.251e+04 | 5 | 4.956 | 99.04 | 4.687 | 98.19 | High |
| Exp2 | high_AWR_high_BD | 3 | 1.021e+04 | 1.751e+04 | 4 | 4.883 | 97.98 | 4.291 | 94.82 | High |
| Exp2 | high_AWR_high_BD | 4 | 9906 | 1.7e+04 | 4 | 4.531 | 91.94 | 5.283 | 100 | High |
| Exp2 | high_AWR_high_BD | 5 | 9606 | 1.65e+04 | 4 | 4.451 | 90.13 | 4.958 | 99.33 | High |
| Exp1 | Exp1_late_stable_candidate | 1 | 4.097e+04 | 4.754e+04 | 5 | -0.4059 | 40.3 | 6.514 | 66.8 | Medium |
| Exp1 | Exp1_late_stable_candidate | 2 | 3.877e+04 | 4.493e+04 | 5 | -0.07169 | 49.73 | 7.054 | 79.57 | Medium |
| Exp1 | Exp1_late_stable_candidate | 3 | 4.055e+04 | 4.703e+04 | 5 | 0.01334 | 52.63 | 6.576 | 67.35 | Medium |
| Exp1 | Exp1_late_stable_candidate | 4 | 3.995e+04 | 4.633e+04 | 5 | 0.08396 | 55.1 | 6.835 | 71.2 | Medium |
| Exp1 | Exp1_late_stable_candidate | 5 | 3.833e+04 | 4.441e+04 | 5 | 0.1637 | 57.66 | 6.981 | 77.06 | Medium |
| Exp1 | low_AWR_high_BD | 1 | 2.725e+04 | 3.129e+04 | 3 | -1.912 | 21.51 | 8.087 | 99.95 | Medium |
| Exp1 | low_AWR_high_BD | 2 | 2.769e+04 | 3.182e+04 | 3 | -1.872 | 22.16 | 8.063 | 99.47 | Medium |
| Exp1 | low_AWR_high_BD | 3 | 2.668e+04 | 3.061e+04 | 3 | -1.788 | 24.55 | 8.069 | 99.63 | Medium |
| Exp1 | low_AWR_high_BD | 4 | 2.591e+04 | 2.97e+04 | 3 | -2.071 | 19.25 | 7.949 | 95.85 | Medium |
| Exp1 | low_AWR_high_BD | 5 | 2.526e+04 | 2.892e+04 | 3 | -1.579 | 30.62 | 8.131 | 99.98 | Medium |
| Exp2 | low_AWR_high_BD | 1 | 1.08e+04 | 1.848e+04 | 4 | 2.541 | 67.77 | 4.471 | 96.66 | Medium |
| Exp2 | low_AWR_high_BD | 2 | 1.11e+04 | 1.899e+04 | 4 | 2.669 | 69.36 | 3.537 | 82.07 | Medium |
| Exp2 | low_AWR_high_BD | 3 | 9296 | 1.598e+04 | 4 | 2.57 | 68.26 | 3.486 | 81.29 | Medium |
| Exp2 | low_AWR_high_BD | 4 | 6900 | 1.199e+04 | 3 | 0.8082 | 33.97 | 2.497 | 66.06 | Medium |
| Exp2 | low_AWR_high_BD | 5 | 1.14e+04 | 1.949e+04 | 4 | 1.885 | 58.25 | 2.969 | 74.51 | Medium |
| Exp1 | AWR_rising | 1 | 7846 | 8319 | 2 | 0.3033 | 61.58 | 4.091 | 28.39 | Medium-High |

## Channel / family contribution hints

| dataset | candidate_type | center_cycle | center_cycle_effective | center_cycle_actual | window_index | rx_contribution | ry_contribution | rs_contribution | corrdist_contribution | amplitude_contribution | dominant_channel | dominant_feature_family | interpretation_hint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1 | high_AWR_high_BD | 4.244e+04 | 4.244e+04 | 4.927e+04 | 8485 | 4.003 | 12 | 20.7 | 24 | 12.71 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | high_AWR_high_BD | 4.336e+04 | 4.336e+04 | 5.036e+04 | 8670 | 2.48 | 12 | 19.68 | 24 | 10.16 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | high_AWR_high_BD | 4.19e+04 | 4.19e+04 | 4.864e+04 | 8378 | 3.221 | 11.38 | 20.13 | 24 | 10.73 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | high_AWR_high_BD | 4.145e+04 | 4.145e+04 | 4.81e+04 | 8288 | 2.548 | 7.498 | 19.78 | 24 | 5.827 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | high_AWR_high_BD | 4.483e+04 | 4.483e+04 | 5.21e+04 | 8963 | 0.6024 | 10.62 | 16.58 | 24 | 3.796 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | high_AWR_low_BD | 3.448e+04 | 3.448e+04 | 3.985e+04 | 6893 | 16.2 | 6.108 | 18.32 | 24 | 16.63 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | high_AWR_low_BD | 10.5 | 10.5 | 11.03 | 0 | 12.31 | -0.1924 | 14.68 | 10.9 | 15.9 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | high_AWR_low_BD | 3.403e+04 | 3.403e+04 | 3.933e+04 | 6804 | 9.274 | 5.543 | 8.82 | 24 | -0.363 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | low_AWR_high_BD | 2.725e+04 | 2.725e+04 | 3.129e+04 | 5447 | -11.56 | 10.56 | -18.13 | 24 | -43.12 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | low_AWR_high_BD | 2.769e+04 | 2.769e+04 | 3.182e+04 | 5536 | -11.69 | 11.54 | -18.56 | 24 | -42.72 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | low_AWR_high_BD | 2.668e+04 | 2.668e+04 | 3.061e+04 | 5333 | -11.33 | 11.21 | -17.77 | 24 | -41.88 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | low_AWR_high_BD | 2.591e+04 | 2.591e+04 | 2.97e+04 | 5179 | -12.29 | 11.03 | -19.46 | 24 | -44.71 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | low_AWR_high_BD | 2.526e+04 | 2.526e+04 | 2.892e+04 | 5049 | -10.69 | 12 | -17.09 | 24 | -39.79 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | AWR_rising | 7846 | 7846 | 8319 | 1567 | 1.1 | 3.702 | -1.769 | 24 | -20.97 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | AWR_rising | 2.14e+04 | 2.14e+04 | 2.433e+04 | 4278 | 4.505 | 5.617 | 1.105 | 24 | -12.77 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | AWR_rising | 2.814e+04 | 2.814e+04 | 3.235e+04 | 5625 | 4.38 | 9.209 | 1.881 | 24 | -8.53 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | AWR_rising | 4.167e+04 | 4.167e+04 | 4.836e+04 | 8331 | 2.86 | 6.899 | 19.43 | 24 | 5.193 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | AWR_rising | 3.459e+04 | 3.459e+04 | 3.999e+04 | 6916 | 15.48 | 7.849 | 17.69 | 24 | 17.02 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | TES_high_confidence | 4.14e+04 | 4.14e+04 | 4.804e+04 | 8277 | 2.772 | 6.379 | 19.73 | 24 | 4.88 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | TES_high_confidence | 3.463e+04 | 3.463e+04 | 4.004e+04 | 6924 | 6.933 | 9.356 | 7.015 | 24 | -0.6962 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | TES_high_confidence | 1.06e+04 | 1.06e+04 | 1.157e+04 | 2118 | 1.629 | 5.394 | -0.2763 | 24 | -17.25 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | TES_high_confidence | 4.304e+04 | 4.304e+04 | 4.998e+04 | 8606 | 2.177 | 12 | 19.74 | 24 | 9.917 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | TES_high_confidence | 4.42e+04 | 4.42e+04 | 5.135e+04 | 8837 | 1.042 | 12 | 17.71 | 24 | 6.75 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp1 | Exp1_late_stable_candidate | 4.097e+04 | 4.097e+04 | 4.754e+04 | 8192 | -2.867 | 4.739 | -5.932 | 24 | -28.06 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | Exp1_late_stable_candidate | 3.877e+04 | 3.877e+04 | 4.493e+04 | 7751 | -1.986 | 5.925 | -4.656 | 24 | -24.72 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | Exp1_late_stable_candidate | 4.055e+04 | 4.055e+04 | 4.703e+04 | 8107 | -1.45 | 5.736 | -4.152 | 24 | -23.87 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | Exp1_late_stable_candidate | 3.995e+04 | 3.995e+04 | 4.633e+04 | 7988 | -1.135 | 5.472 | -3.498 | 24 | -23.16 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp1 | Exp1_late_stable_candidate | 3.833e+04 | 3.833e+04 | 4.441e+04 | 7663 | -1.227 | 6.436 | -3.571 | 24 | -22.36 | rs | corrdist_base | Dominant channel rs; dominant family corrdist_base. |
| Exp2 | high_AWR_high_BD | 1.265e+04 | 1.265e+04 | 2.158e+04 | 2528 | 34.96 | 10.45 | 7.735 | 12.73 | 40.41 | rx | absmean | Dominant channel rx; dominant family absmean. |
| Exp2 | high_AWR_high_BD | 1.321e+04 | 1.321e+04 | 2.251e+04 | 2639 | 35.22 | 3.418 | 10.92 | 14.09 | 35.47 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_high_BD | 1.021e+04 | 1.021e+04 | 1.751e+04 | 2040 | 28.75 | 6.099 | 13.98 | 16.16 | 32.67 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_high_BD | 9906 | 9906 | 1.7e+04 | 1979 | 33.82 | 0.3704 | 11.11 | 16.33 | 28.98 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_high_BD | 9606 | 9606 | 1.65e+04 | 1919 | 29.76 | 2.942 | 11.81 | 11.97 | 32.54 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_low_BD | 1.254e+04 | 1.254e+04 | 2.139e+04 | 2505 | 26.19 | 11.88 | 11.96 | 18.83 | 31.21 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_low_BD | 1.172e+04 | 1.172e+04 | 2.003e+04 | 2342 | 31.1 | 9.054 | 8.954 | 15.06 | 34.05 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_low_BD | 1.253e+04 | 1.253e+04 | 2.138e+04 | 2504 | 26.35 | 11.64 | 11.99 | 18.93 | 31.06 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_low_BD | 1.254e+04 | 1.254e+04 | 2.14e+04 | 2506 | 25.79 | 11.9 | 11.61 | 18.27 | 31.03 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | high_AWR_low_BD | 1.253e+04 | 1.253e+04 | 2.137e+04 | 2503 | 26.39 | 11.78 | 11.85 | 18.73 | 31.3 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | low_AWR_high_BD | 1.08e+04 | 1.08e+04 | 1.848e+04 | 2157 | 23.27 | -1.53 | 3.671 | 9.68 | 15.73 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |
| Exp2 | low_AWR_high_BD | 1.11e+04 | 1.11e+04 | 1.899e+04 | 2218 | 20.39 | 0.5272 | 5.779 | 12.1 | 14.59 | rx | corrdist_base | Dominant channel rx; dominant family corrdist_base. |

## Warnings

- None.

## Physical-loop checks

- For high_AWR_high_BD windows, check surface damage, debris increase, and FEM high-contribution contact zones.
- For low_AWR_high_BD windows, check run-in, contact reorganization, or lubrication disturbance.
- For TES event windows, check whether transitions align with disassembly, lubrication change, measurement disturbance, or real force-signal jumps.
- For ry-dominant windows, check lateral contact migration or eccentric loading traces.