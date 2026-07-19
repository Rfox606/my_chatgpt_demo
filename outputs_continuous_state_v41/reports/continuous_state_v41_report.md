# Continuous State Monitoring v4.1 report

## Causal protocol

The first 1,000 cycles are calibration-only. They freeze target robust location/scale, Ledoit–Wolf group covariances, evidence thresholds, and the causal residual reference; no formal state, evidence, or forecast row is emitted before the post-baseline monitoring start. Stage columns are never read by online modules.

`rs_absmean` is excluded. rs, rx, and ry are normalized and scored in separate subspaces; their distances are fused with equal group weights. `baseline_outlier_fraction` uses the frozen target calibration, while `source_support_oos` uses the source experiment support envelope. V100/V500/V1000 replace the short v4 velocity scales. Abrupt evidence is an online EWMA-residual CUSUM, not an acceleration derivative.

## Required answers

1. **Exp1 around 8,000 cycles:** event-collection stability in v4.1 ablations is 0.104 for configurations with comparable near-8,000 event records. This is FAIL under the descriptive matching rule; it is not a wear label.
2. **Exp2 late acceleration/abrupt evidence:** v4.1 late-event fraction=0.267, v4 fraction=0.438; concentrated-more-than-v4=False. Exp2 current onsets: `acceleration_evidence, abrupt_change_evidence`.
3. **Trajectory and complete-event-set stability:** trajectory-stable rate=0.896; complete-event-collection-stable rate=0.130. This is FAIL across baseline, distance, and feature-group ablations, without changing thresholds after inspection.
4. **Online against best simple baseline:** Online RLS is lower-MAE in 4/24 output–horizon–protocol comparisons. Strict all-comparison superiority is **FAIL**. Safe Gate compares Online with the current best of Zero Delta, Local Linear, Kalman, and Frozen Ridge.
5. **Event excess or numeric instability:** event-density status=PASS; numerical-stability status=PASS. Event onset density=0.642/1,000 cycles; finite state/forecast values=True; maximum Online RLS prediction magnitude=30.000.

## Evidence summary

```text
   protocol_id target_dataset            evidence_type  active_window_count  onset_count  first_onset_cycle  last_onset_cycle  active_at_8000
A_Exp1_to_Exp2           Exp2    low_activity_evidence                  345            2             4910.5           13830.5           False
A_Exp1_to_Exp2           Exp2 directed_change_evidence                   21            2             2795.5           11795.5           False
A_Exp1_to_Exp2           Exp2    acceleration_evidence                   54            5             2790.5           10895.5           False
A_Exp1_to_Exp2           Exp2   abrupt_change_evidence                  736           10             2790.5           12705.5           False
B_Exp2_to_Exp1           Exp1    low_activity_evidence                 7130            8             1885.5           42920.5           False
B_Exp2_to_Exp1           Exp1 directed_change_evidence                  103            3            23935.5           34705.5           False
B_Exp2_to_Exp1           Exp1    acceleration_evidence                   10            1            27935.5           27935.5           False
B_Exp2_to_Exp1           Exp1   abrupt_change_evidence                  242            6             7705.5           45230.5            True
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
........................................................................ [ 65%]
......................................                                   [100%]
============================== warnings summary ===============================
tests/test_csv1_no_stage_leakage.py: 6 warnings
tests/test_csv1_rank_direction.py: 7 warnings
tests/test_csv2_pre_refit_validation.py: 7 warnings
  D:\Program Files\Python313\Lib\site-packages\sklearn\linear_model\_logistic.py:1403: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
110 passed, 20 warnings in 15.57s
```

All failures above are retained as observed; no post-hoc acceptance tuning was applied.
