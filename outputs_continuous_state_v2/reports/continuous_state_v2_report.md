# Continuous State Monitoring v2

## Required answers

1. Corrected source pre-refit validation AUC:

```text
direction_id  source_validation_auc_pre_refit  source_validation_auc_after_refit_replay
 Exp1_source                         0.918293                                  0.770025
 Exp2_source                         0.798206                                  0.872092
```

2. Restart guard now uses full window/guard-region intersection at 50, 100, and 150 cycles; the main analysis uses 100 cycles. Thus overlap candidates near each 500-cycle boundary are excluded rather than selected.
3. Repeated/highly correlated feature audit:

```text
direction_id     feature_name  kept                   drop_reason  correlated_with  single_feature_pair_auc
 Exp1_source       rs_absmean     0    EXACT_DUPLICATE_OF_rs_mean          rs_mean                      NaN
 Exp1_source           rs_q05     0 HIGH_CORRELATION_SOURCE_TRAIN           rx_q05                 0.481995
 Exp1_source           rs_rms     0 HIGH_CORRELATION_SOURCE_TRAIN          rx_mean                 0.452099
 Exp1_source       rx_absmean     0 HIGH_CORRELATION_SOURCE_TRAIN          rx_mean                 0.475393
 Exp1_source rx_corrdist_base     0 HIGH_CORRELATION_SOURCE_TRAIN rs_corrdist_base                 0.710977
 Exp2_source       rs_absmean     0    EXACT_DUPLICATE_OF_rs_mean          rs_mean                      NaN
 Exp2_source           rs_q05     0 HIGH_CORRELATION_SOURCE_TRAIN          rs_mean                 0.810741
 Exp2_source           rs_rms     0 HIGH_CORRELATION_SOURCE_TRAIN          rs_mean                 0.842082
 Exp2_source       rx_absmean     0 HIGH_CORRELATION_SOURCE_TRAIN          rx_mean                 0.702665
```

4–5. Common-direction features and status: **COMMON_AXIS_WEAK**.

```text
    feature_name  median_weight_exp1  median_weight_exp2  sign_stability_exp1  sign_stability_exp2  common_weight_raw  kept_common                        drop_reason  w_common
rs_corrdist_base            0.022774           -0.177035                0.995                1.000           0.000000            0 FAILED_STABILITY_SIGN_OR_MAGNITUDE       0.0
         rs_mean           -0.374263            0.316991                0.995                0.945           0.000000            0 FAILED_STABILITY_SIGN_OR_MAGNITUDE       0.0
         rx_mean            0.453667            0.019990                1.000                0.615           0.000000            0 FAILED_STABILITY_SIGN_OR_MAGNITUDE       0.0
          rx_q05           -0.068916           -0.061967                0.990                0.650           0.000000            0 FAILED_STABILITY_SIGN_OR_MAGNITUDE       0.0
          ry_p2p            0.081815            0.093951                1.000                0.990           0.087673            1                               KEPT       1.0
```

6–9. Target segment diagnostics (P, BD, and terminal branch score):

```text
direction_id target_dataset  P_cycle_spearman_early  BD_cycle_spearman_early  B_cycle_spearman_early  P_cycle_spearman_middle  BD_cycle_spearman_middle  B_cycle_spearman_middle  P_cycle_spearman_late  BD_cycle_spearman_late  B_cycle_spearman_late  P_BD_joint_high_rate  high_BD_low_P_rate  high_BD_positive_B_rate  high_BD_negative_B_rate
Exp1_to_Exp2           Exp2               -0.111654                 0.718981                0.595807                 0.846768                 -0.164410                -0.016509               0.361007                0.223440               0.631741              0.095865                 0.0                 0.100094                      0.0
Exp2_to_Exp1           Exp1                0.770922                 0.875027                0.563565                 0.640770                  0.423427                 0.626755               0.305441                0.506071              -0.281980              0.056969                 0.0                 0.100058                      0.0
```

The terminal branch score is a residual direction relative to Exp1/Exp2 terminal references. It may include condition differences and requires physical validation before any stable–severe interpretation.
10. Physical-validation candidate regions:

```text
direction_id source_dataset target_dataset          candidate_type  candidate_start_cycle  candidate_end_cycle  peak_cycle  duration_windows  peak_P_common    peak_BD  peak_B_terminal  peak_TES  peak_weighted_oos  diagnostic_only  not_an_online_alarm  requires_physical_validation
Exp1_to_Exp2           Exp1           Exp2          high_P_high_BD                12371.0              12490.0     12430.5                21      12.332596 109.542570        11.125957  5.399068                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2          high_P_high_BD                11786.0              11905.0     11845.5                21      10.447440  89.806961        10.481040  8.791394                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2          high_P_high_BD                12871.0              12990.0     12930.5                21       7.695935  69.542323        11.257195  6.558566                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2          high_P_high_BD                10396.0              10445.0     10420.5                 7       6.618742  66.660982         9.030196  9.348783                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2 stable_branch_candidate                  826.0                915.0       885.5                15      -2.957540  22.734811        -3.089606  0.595304                1.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2 stable_branch_candidate                 2181.0               2300.0      2240.5                21      -2.201987  17.527488        -2.498400  0.051097                1.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2 stable_branch_candidate                 1426.0               1495.0      1485.5                11      -3.117008  26.722211        -1.892419  0.366738                1.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2 severe_branch_candidate                 9886.0               9970.0      9910.5                14      -0.736365  55.589350        13.914849  5.871017                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2 severe_branch_candidate                13141.0              13260.0     13200.5                21       3.349683  47.895661        13.345069  5.007316                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2 severe_branch_candidate                13926.0              13995.0     13985.5                11       5.702333  64.463766        13.201090 18.917172                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2 severe_branch_candidate                12616.0              12735.0     12675.5                21      10.364421  84.463911        12.578416  1.127706                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2          rapid_P_growth                10251.0              10330.0     10270.5                13       5.693236  58.415214        10.013142  7.489318                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2          rapid_P_growth                11796.0              11900.0     11855.5                18      10.303457  88.336671        10.240227  7.075784                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2          rapid_P_growth                 6951.0               6995.0      6985.5                 6       5.446555  41.795548         0.755089  0.589921                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2          rapid_P_growth                10876.0              10990.0     10935.5                20       2.651174  34.005797         7.851607  0.286431                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2          rapid_P_growth                 8466.0               8495.0      8485.5                 3       3.852763  28.786804         1.048289  0.743895                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2         rapid_BD_growth                11791.0              11900.0     11850.5                19      10.380423  88.997203        10.301122  7.907674                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2         rapid_BD_growth                 2806.0               2890.0      2865.5                14      -0.414397  60.306756         3.027066  0.250686                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2         rapid_BD_growth                 6956.0               6995.0      6985.5                 5       5.446555  41.795548         0.755089  0.589921                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2         rapid_BD_growth                 8756.0               8825.0      8765.5                11      -1.112264  40.382762         8.629406  5.138916                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2         rapid_BD_growth                  661.0                760.0       700.5                17      -2.821275  22.336251        -1.944734  0.001344                1.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2         rapid_BD_growth                 9816.0               9900.0      9875.5                14       4.473900  60.640670        10.001243  3.981709                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2       branch_transition                 8756.0               8875.0      8815.5                21      -0.950719  37.985692         7.672376  1.397448                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2       branch_transition                 9356.0               9475.0      9415.5                21       1.927267  29.952097         8.913185  1.322243                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2       branch_transition                 2756.0               2875.0      2815.5                21      -0.293192  60.099544         3.217279  0.922889                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2       branch_transition                11756.0              11875.0     11815.5                21       9.704030  82.679519        10.848328  2.677420                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2       branch_transition                 6136.0               6200.0      6150.5                10      -1.729163  20.017456         2.704017  0.601475                1.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2                high_TES                11691.0              11780.0     11750.5                15      10.564922  88.137531        11.168547 74.020349                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2                high_TES                 6896.0               6970.0      6950.5                12       5.473153  42.177594         1.186268 67.297725                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2                high_TES                 2686.0               2755.0      2705.5                11      -0.683465  59.137179         3.988096 57.795726                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2                high_TES                10231.0              10265.0     10245.5                 4       6.092780  58.191977        10.430355 49.939112                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2                high_TES                 8691.0               8760.0      8710.5                11      -1.782932  51.838638        10.312641 48.476430                1.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2                high_TES                  601.0                660.0       645.5                 9      -2.818270  22.625992        -1.658616 26.767363                1.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2                high_TES                10836.0              10860.0     10845.5                 2       2.687667  36.993185         9.113082 25.385022                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2                high_TES                 1796.0               1860.0      1810.5                10      -3.961809  34.200977        -1.703204 21.380562                1.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2                high_TES                 9716.0               9780.0      9735.5                10       5.216370  53.995475         9.657841 20.345652                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2                high_TES                 5696.0               5745.0      5710.5                 7      -1.398375  38.730089         9.580094 19.644196                1.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2                high_TES                 1201.0               1255.0      1245.5                 8      -3.553165  29.671971        -0.845442 19.022533                1.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2                high_TES                13971.0              13995.0     13985.5                 2       5.702333  64.463766        13.201090 18.917172                0.0                1                    1                             1
Exp1_to_Exp2           Exp1           Exp2                high_TES                 3336.0               3360.0      3345.5                 2      -2.908287  45.905948         8.485035 16.939776                1.0                1                    1                             1
```

11–13. Adapter activity:

```text
              adapter_updated  adapter_rollback
direction_id
Exp1_to_Exp2              524               212
Exp2_to_Exp1              155              5789
```

The adapter is constrained to the nuisance subspace and baseline-replay rollback; it is not permitted to modify P/branch source axes or BD baselines.
14–15. Strict delayed-observation forecast metrics and benefit decisions:

```text
direction_id  horizon_cycles      model  prediction_count    MAE_P    RMSE_P  direction_accuracy_P    MAE_BD   RMSE_BD  direction_accuracy_BD    MAE_B    RMSE_B  direction_accuracy_B
Exp1_to_Exp2             100     Frozen              2778 1.569889  2.196076              0.511519  5.811389  9.401945               0.518719 1.375916  1.808730              0.525918
Exp1_to_Exp2             100 Online_RLS              2778 1.962933  3.249497              0.484881 13.584991 26.519245               0.512959 2.353371  4.974826              0.558315
Exp1_to_Exp2             500     Frozen              2698 4.947155  6.052696              0.560044 13.509423 18.446224               0.641586 3.684428  4.425941              0.525204
Exp1_to_Exp2             500 Online_RLS              2698 3.665604  5.885625              0.489622 28.779357 48.979582               0.585248 4.438932  6.830340              0.487769
Exp1_to_Exp2            1000     Frozen              2598 8.783211 11.287389              0.531178 17.981083 23.896332               0.675905 5.285489  6.348356              0.520015
Exp1_to_Exp2            1000 Online_RLS              2598 4.859988  7.035996              0.577752 42.107633 59.176009               0.572748 8.138036 12.099874              0.519630
Exp2_to_Exp1             100     Frozen              9076 0.560620  0.778234              0.475650  5.717191  6.246091               0.527435 0.962316  1.079148              0.435434
Exp2_to_Exp1             100 Online_RLS              9076 0.733462  1.476883              0.519061  1.931668  7.671761               0.542309 0.643988  2.498018              0.522918
Exp2_to_Exp1             500     Frozen              8996 1.374493  1.820744              0.527345 17.972847 19.115639               0.551356 2.873990  3.054066              0.378502
Exp2_to_Exp1             500 Online_RLS              8996 2.298967  5.344573              0.496554  5.464253 12.484808               0.514673 2.527892  9.828405              0.499333
Exp2_to_Exp1            1000     Frozen              8896 2.625578  3.247279              0.513939 14.625002 15.985132               0.575540 4.870821  5.102543              0.371628
Exp2_to_Exp1            1000 Online_RLS              8896 3.655236  7.146280              0.514164  7.933458 14.806222               0.552496 3.820478 10.249339              0.519897
direction_id  beneficial_horizon_count  baseline_replay_drift_ok ONLINE_ADAPTATION_BENEFIT
Exp1_to_Exp2                         2                      True                      PASS
Exp2_to_Exp1                         3                      True                      PASS
```

16. A more complex neural adapter is not justified solely by trajectory shape. It should only be considered if the fixed support, replay, and delayed-prediction diagnostics remain acceptable.
17. P, B, and BD are not wear percentages, absolute wear quantities, failure probabilities, or Stage5 probabilities.
18. Exp1 and Exp2 terminal references are deliberately not forced to be equal: Exp1 late behavior is treated as a stable-wear reference, while Exp2 late behavior is a more severe reference; the model uses a common trunk plus a terminal residual branch.
