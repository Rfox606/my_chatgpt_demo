# Continuous State Monitoring v4.2 report

## Method

v4.2 keeps v4.1's causal target calibration, Guard, label isolation, multi-scale rates, equal rs/rx/ry subspace fusion, and Safe Gate. It evaluates 24 pre-declared baseline/distance/feature-group configurations and reports configuration consensus rather than assigning ordered binary states. The formal consensus begins after 1,000 cycles; configurations enter only after their own calibration ends.

For each continuous state quantity, the output records Q25/Q50/Q75, MAD, and valid configuration count. `multi_scale_rate_divergence` is the renamed former A-state. Change episodes are support-qualified continuous intervals: directed, rate-divergence, and abrupt are composition proportions, not fixed-order labels. Abrupt remains the independent residual/CUSUM track.

## Required answers

1. **Exp1 near 8,000:** 7570.5-7985.5, support=1.000, dominant=abrupt. A high-support interval requires support ≥0.80; result is **PASS**.
2. **Exp2 near 11,695:** 9155.5-13300.5, support=1.000, composition=directed=0.133;rate_divergence=0.486;abrupt=0.654, dominant=abrupt. abrupt+directed predominance is **PASS**.
3. **Removing ry in Exp1:** D consensus Spearman=0.724; full/no-ry episode counts=9/2; episode-peak Jaccard=0.222.
4. **Consensus vs v4.1 single-configuration events:** high-support consensus episode fraction=0.625; v4.1 event-collection stability=0.130; status=PASS.
5. **Online forecast:** Online RLS is better than the best simple baseline in 4/24 output–horizon–protocol comparisons. RLS clipping ratio=0.002; Safe Gate Online selection ratio=0.318.

## Change episodes

```text
   protocol_id target_dataset  start_cycle  end_cycle  peak_cycle  peak_change_score  configuration_support  location_uncertainty  configuration_peak_count                              evidence_composition  directed_composition  rate_divergence_composition  abrupt_composition dominant_evidence  covers_guard_or_stop_boundary
A_Exp1_to_Exp2           Exp2       1210.5     1895.5      1825.5          27.810659               1.000000                265.00                        16 directed=0.000;rate_divergence=0.669;abrupt=0.269              0.000000                     0.669471            0.269231   rate_divergence                           True
A_Exp1_to_Exp2           Exp2       2695.5     4155.5      3090.5          78.362514               1.000000                  5.00                        24 directed=0.120;rate_divergence=0.438;abrupt=0.764              0.119732                     0.438218            0.764368            abrupt                           True
A_Exp1_to_Exp2           Exp2       6360.5     6470.5      6390.5           2.883662               0.625000                  0.00                        24 directed=0.311;rate_divergence=0.064;abrupt=0.303              0.311404                     0.063596            0.302632          directed                          False
A_Exp1_to_Exp2           Exp2       6900.5     7300.5      7285.5          60.971943               1.000000                 10.00                        24 directed=0.100;rate_divergence=0.653;abrupt=0.609              0.100282                     0.653249            0.608757   rate_divergence                           True
A_Exp1_to_Exp2           Exp2       7805.5     8155.5      7840.5          16.733271               0.916667                112.50                        24 directed=0.036;rate_divergence=0.658;abrupt=0.406              0.035714                     0.657738            0.406250   rate_divergence                           True
A_Exp1_to_Exp2           Exp2       8410.5     8840.5      8710.5          25.374972               1.000000                  0.00                        24 directed=0.395;rate_divergence=0.736;abrupt=0.750              0.394737                     0.735746            0.750000            abrupt                          False
A_Exp1_to_Exp2           Exp2       9155.5    13300.5     10585.5          82.995047               1.000000                 30.00                        24 directed=0.133;rate_divergence=0.486;abrupt=0.654              0.132981                     0.486475            0.653646            abrupt                           True
B_Exp2_to_Exp1           Exp1       1105.5     1530.5      1525.5          15.407779               1.000000                  0.00                        16 directed=0.031;rate_divergence=0.740;abrupt=0.240              0.031250                     0.739583            0.239583   rate_divergence                           True
B_Exp2_to_Exp1           Exp1       7570.5     7985.5      7655.5          31.581472               1.000000                 16.25                        24 directed=0.357;rate_divergence=0.298;abrupt=0.699              0.357143                     0.298115            0.698909            abrupt                           True
B_Exp2_to_Exp1           Exp1      19655.5    19755.5     19755.5          32.345596               0.583333                  0.00                        24 directed=0.000;rate_divergence=0.542;abrupt=0.125              0.000000                     0.541667            0.125000   rate_divergence                          False
B_Exp2_to_Exp1           Exp1      21130.5    21250.5     21180.5          25.865027               1.000000                  1.25                        24 directed=0.142;rate_divergence=0.646;abrupt=0.530              0.142045                     0.645833            0.530303   rate_divergence                          False
B_Exp2_to_Exp1           Exp1      27840.5    27960.5     27900.5          28.424311               0.875000                  1.25                        24 directed=0.282;rate_divergence=0.668;abrupt=0.640              0.281667                     0.668333            0.640000   rate_divergence                          False
B_Exp2_to_Exp1           Exp1      34595.5    34895.5     34610.5          18.783823               0.791667                 63.75                        24 directed=0.342;rate_divergence=0.293;abrupt=0.475              0.342320                     0.293301            0.475490            abrupt                           True
B_Exp2_to_Exp1           Exp1      39120.5    39320.5     39120.5          12.830978               0.583333                  1.25                        24 directed=0.292;rate_divergence=0.396;abrupt=0.427              0.291667                     0.395833            0.427083            abrupt                          False
B_Exp2_to_Exp1           Exp1      41365.5    42135.5     41370.5          22.184831               0.666667                 22.50                        24 directed=0.232;rate_divergence=0.294;abrupt=0.442              0.231838                     0.293803            0.442308            abrupt                           True
B_Exp2_to_Exp1           Exp1      45015.5    45220.5     45015.5          21.549981               0.666667                  0.00                        24 directed=0.292;rate_divergence=0.304;abrupt=0.408              0.291667                     0.304167            0.408333            abrupt                           True
```

## ry_p2p audit

```text
dataset feature_name  baseline_location  baseline_scale  monitoring_outlier_count  monitoring_outlier_fraction  guard_outlier_fraction  non_guard_outlier_fraction  guard_dependency_enrichment  stop_boundary_outlier_fraction  ry_subspace_dominance_p95  ry_subspace_dominance_fraction_over_060  single_feature_dominance_flag
   Exp1       ry_p2p          -0.339204        0.407574                      7686                     0.862142                0.857276                     0.86369                     0.992573                        0.862360                   0.763569                                 0.759394                           True
   Exp2       ry_p2p          -1.752253        2.064344                       526                     0.200994                0.180967                     0.20749                     0.872174                        0.201923                   0.477366                                 0.009171                          False
```

## Forecast comparison

```text
   protocol_id           output_name  horizon_cycles best_static_model  best_static_MAE  Online_RLS_MAE  Safe_Gate_MAE  online_to_best_static_mae_ratio  online_truly_better  safe_gate_no_worse_than_best_static_5pct
A_Exp1_to_Exp2               A_state             100      Frozen_Ridge         0.790452        1.010449       0.823114                         1.278317                False                                      True
A_Exp1_to_Exp2               A_state             500      Frozen_Ridge         0.702377        1.026511       0.694608                         1.461482                False                                      True
A_Exp1_to_Exp2               A_state            1000      Frozen_Ridge         0.920993        6.168878       2.017217                         6.698074                False                                     False
A_Exp1_to_Exp2               D_state             100        Zero_Delta         0.539341        0.762249       0.539886                         1.413296                False                                      True
A_Exp1_to_Exp2               D_state             500        Zero_Delta         1.207339        1.808328       1.301477                         1.497779                False                                     False
A_Exp1_to_Exp2               D_state            1000        Zero_Delta         1.655048        4.858360       1.831073                         2.935480                False                                     False
A_Exp1_to_Exp2             V500_norm             100        Zero_Delta         0.156933        0.190339       0.165665                         1.212868                False                                     False
A_Exp1_to_Exp2             V500_norm             500        Zero_Delta         0.243727        0.272382       0.230678                         1.117570                False                                      True
A_Exp1_to_Exp2             V500_norm            1000        Zero_Delta         0.286933        0.853474       0.286933                         2.974472                False                                      True
A_Exp1_to_Exp2 residual_change_score             100        Zero_Delta         0.525552        0.670135       0.555501                         1.275106                False                                     False
A_Exp1_to_Exp2 residual_change_score             500        Zero_Delta         0.816891        0.972168       0.861421                         1.190082                False                                     False
A_Exp1_to_Exp2 residual_change_score            1000        Zero_Delta         0.872372        5.050225       0.872372                         5.789071                False                                      True
B_Exp2_to_Exp1               A_state             100        Zero_Delta         0.258637        0.290357       0.256334                         1.122644                False                                      True
B_Exp2_to_Exp1               A_state             500        Zero_Delta         0.290429        0.358592       0.288719                         1.234697                False                                      True
B_Exp2_to_Exp1               A_state            1000        Zero_Delta         0.320632        0.362078       0.301940                         1.129263                False                                      True
B_Exp2_to_Exp1               D_state             100        Zero_Delta         0.260417        0.296125       0.262800                         1.137118                False                                      True
B_Exp2_to_Exp1               D_state             500        Zero_Delta         0.594823        0.750611       0.607383                         1.261907                False                                      True
B_Exp2_to_Exp1               D_state            1000        Zero_Delta         0.836223        1.224455       0.856166                         1.464268                False                                      True
B_Exp2_to_Exp1             V500_norm             100        Zero_Delta         0.067526        0.073581       0.067956                         1.089673                False                                      True
B_Exp2_to_Exp1             V500_norm             500        Zero_Delta         0.133066        0.125158       0.126735                         0.940571                 True                                      True
B_Exp2_to_Exp1             V500_norm            1000      Frozen_Ridge         0.154453        0.144072       0.150440                         0.932786                 True                                      True
B_Exp2_to_Exp1 residual_change_score             100        Zero_Delta         0.235693        0.235280       0.224149                         0.998247                 True                                      True
B_Exp2_to_Exp1 residual_change_score             500        Zero_Delta         0.382260        0.398230       0.359580                         1.041777                False                                      True
B_Exp2_to_Exp1 residual_change_score            1000        Zero_Delta         0.442327        0.434080       0.426808                         0.981355                 True                                      True
```

## PASS/FAIL record

Implementation diagnostics: **PASS**.

```text
........................................................................ [ 62%]
............................................                             [100%]
============================== warnings summary ===============================
tests/test_csv1_no_stage_leakage.py: 6 warnings
tests/test_csv1_rank_direction.py: 7 warnings
tests/test_csv2_pre_refit_validation.py: 7 warnings
  D:\Program Files\Python313\Lib\site-packages\sklearn\linear_model\_logistic.py:1403: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
116 passed, 20 warnings in 18.09s
```

Failures are retained as observed; no post-hoc feature or threshold adjustment was made.
