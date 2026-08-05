# Continuous State Monitoring v4 report

## Scope and causal protocol

v4 replaces the ordered platform-lock → platform-exit → serious-event state machine with four independent algorithm-state evidence tracks: `low_activity_evidence`, `directed_change_evidence`, `acceleration_evidence`, and `abrupt_change_evidence`. These are algorithmic evidence only; they are not stability/wear/failure labels or probabilities.

Each target stream freezes robust location, scale, covariance/precision, and all evidence thresholds from its initial non-guard baseline. The online stream reads no Stage columns, predicts before RLS updates, pauses evidence tracks in restart guards, and is checked by prefix replay. D, V20/V50/V100, direction consistency, A, volatility, weighted OOS, every per-feature velocity vector, and rs/rx/ry velocity contributions are exported.

## Required answers

1. **Exp1 near 8,000 cycles:** No v4 serious-wear class or alarm is emitted, so the former class-level false-alarm question does not apply as an output. Active tracks at the nearest 8,000-cycle window: `none`; onset events in 7,800–8,200 cycles: `directed_change_evidence@8180.5`. Any such event is label-free algorithm evidence, not a serious-wear label.
2. **Exp2 without a platform:** 是. At least one onset was observed for the following independent directional/acceleration tracks: `acceleration_evidence`. This result does not depend on platform detection.
3. **Baseline/distance/feature-group stability:** **FAIL**. 56/192 evidence-specific ablation rows meet the pre-declared descriptive stability rule (D Spearman ≥ 0.80 and matching/move ≤500-cycle first onset). Trajectory-stable rate=0.604; event-position-stable rate=0.417. This is an observation, not a threshold-selection step.
4. **Does Online truly beat the best simple baseline?** Strict all-comparison superiority: **FAIL**. Online RLS is lower-MAE than the best of Zero Delta, Local Linear, Kalman, and Frozen Ridge in only 2/24 output–horizon–protocol comparisons. It is therefore not universally superior. Safe Gate selects Online only when its current cycle-window MAE is no worse than the current best static baseline.

## Evidence summary

```text
   protocol_id target_dataset            evidence_type  active_window_count  onset_count  first_onset_cycle  last_onset_cycle  active_at_8000
A_Exp1_to_Exp2           Exp2    low_activity_evidence                  729           20              205.5           11265.5            True
A_Exp1_to_Exp2           Exp2 directed_change_evidence                    0            0                NaN               NaN           False
A_Exp1_to_Exp2           Exp2    acceleration_evidence                  125            4             3775.5           13930.5           False
A_Exp1_to_Exp2           Exp2   abrupt_change_evidence                  177           12             2790.5           13290.5           False
B_Exp2_to_Exp1           Exp1    low_activity_evidence                 3039           42              460.5           42410.5           False
B_Exp2_to_Exp1           Exp1 directed_change_evidence                   13            1             8180.5            8180.5           False
B_Exp2_to_Exp1           Exp1    acceleration_evidence                  915           11            14660.5           44380.5           False
B_Exp2_to_Exp1           Exp1   abrupt_change_evidence                  972           11            14455.5           44385.5           False
```

## Forecast comparison to the current best static baseline

```text
   protocol_id  output_name  horizon_cycles best_static_model  best_static_MAE  Online_RLS_MAE  Safe_Gate_MAE  online_to_best_static_mae_ratio  online_truly_better  safe_gate_no_worse_than_best_static_5pct
A_Exp1_to_Exp2      A_state             100      Frozen_Ridge        17.069766       15.404903      15.582230                         0.902467                 True                                      True
A_Exp1_to_Exp2      A_state             500      Frozen_Ridge        15.596791       21.948385      17.887558                         1.407237                False                                     False
A_Exp1_to_Exp2      A_state            1000        Zero_Delta        18.431849       26.768769      18.360801                         1.452311                False                                      True
A_Exp1_to_Exp2      D_state             100        Zero_Delta         2.650248        4.492711       2.650248                         1.695204                False                                      True
A_Exp1_to_Exp2      D_state             500        Zero_Delta         6.153046       17.279353       6.153046                         2.808260                False                                      True
A_Exp1_to_Exp2      D_state            1000        Zero_Delta         8.634183       61.340387       8.634183                         7.104365                False                                      True
A_Exp1_to_Exp2     V50_norm             100        Zero_Delta         8.172852        8.291147       7.856002                         1.014474                False                                      True
A_Exp1_to_Exp2     V50_norm             500        Zero_Delta         8.783526       20.632343       9.347215                         2.348982                False                                     False
A_Exp1_to_Exp2     V50_norm            1000        Zero_Delta         8.749962       12.591646       8.749962                         1.439051                False                                      True
A_Exp1_to_Exp2 abrupt_score             100      Frozen_Ridge        32.276493       28.751060      29.313916                         0.890774                 True                                      True
A_Exp1_to_Exp2 abrupt_score             500      Frozen_Ridge        31.220437       36.707683      35.792453                         1.175758                False                                     False
A_Exp1_to_Exp2 abrupt_score            1000        Zero_Delta        34.402365       54.379244      34.679452                         1.580683                False                                      True
B_Exp2_to_Exp1      A_state             100        Zero_Delta         4.655828     1242.073787       4.655828                       266.778259                False                                      True
B_Exp2_to_Exp1      A_state             500        Zero_Delta         5.758871     2023.140540       5.758871                       351.308520                False                                      True
B_Exp2_to_Exp1      A_state            1000        Zero_Delta         5.700509      846.097358       5.700509                       148.424869                False                                      True
B_Exp2_to_Exp1      D_state             100        Zero_Delta         1.233305        2.288166       1.233305                         1.855312                False                                      True
B_Exp2_to_Exp1      D_state             500        Zero_Delta         3.246630        5.236347       3.247639                         1.612856                False                                      True
B_Exp2_to_Exp1      D_state            1000        Zero_Delta         5.105426        9.022580       5.654202                         1.767253                False                                     False
B_Exp2_to_Exp1     V50_norm             100        Zero_Delta         2.792332        9.071368       2.792332                         3.248671                False                                      True
B_Exp2_to_Exp1     V50_norm             500        Zero_Delta         3.101860      600.614985       3.101860                       193.630611                False                                      True
B_Exp2_to_Exp1     V50_norm            1000        Zero_Delta         3.209028        5.821991       3.223338                         1.814254                False                                      True
B_Exp2_to_Exp1 abrupt_score             100        Zero_Delta         8.781978     1348.012521       8.781978                       153.497599                False                                      True
B_Exp2_to_Exp1 abrupt_score             500        Zero_Delta        10.736203     2897.031250      10.736203                       269.837602                False                                      True
B_Exp2_to_Exp1 abrupt_score            1000        Zero_Delta        10.373532     3445.870478      10.373532                       332.179081                False                                      True
```

## PASS/FAIL record

Implementation diagnostics: **PASS**.

```text
........................................................................ [ 69%]
................................                                         [100%]
============================== warnings summary ===============================
tests/test_csv1_no_stage_leakage.py: 6 warnings
tests/test_csv1_rank_direction.py: 7 warnings
tests/test_csv2_pre_refit_validation.py: 7 warnings
  D:\Program Files\Python313\Lib\site-packages\sklearn\linear_model\_logistic.py:1403: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
104 passed, 20 warnings in 14.32s
```

Failures, if any, are retained above and were not removed by post-hoc parameter adjustment.
