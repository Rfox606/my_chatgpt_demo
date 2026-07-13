# Continuous State Monitoring v3 — Causal Plateau-to-Severe Online Adaptation

## Required answers

1. Exp1 stable-wear plateau detected: False.
2. Exp1 plateau lock cycle: nan.
3. Exp1 persistent severe false alarms (>500 cycles): 0.
4. Exp2 plateau/low-speed region before instability: plateau=nan, exit=nan.
5. Exp2 plateau exit cycle: nan.
6. Exp2 persistent severe-candidate onset: nan.
7. Exp2 lead/lag to the post-hoc known severe boundary: nan cycles. This label was read only after online outputs were saved.
8. Protocol A leakage check: PASS; Exp2 future and terminal data were excluded from source priors.
9. Target severe-direction updates (causal, post-exit only):

```text
No available records.
```

10. D/V/A/S differences are retained as continuous target-relative state signals, not stages:

```text
   protocol_id target_dataset   D_state  V50_norm   A_state  S_severe_candidate
A_Exp1_to_Exp2           Exp2 36.991409  1.936995 -2.276511                 NaN
B_Exp2_to_Exp1           Exp1 17.341169  0.590123 -0.572239                 NaN
```

11. M0–M5 ablation evidence:

```text
   protocol_id ablation                                         module                   evidence
A_Exp1_to_Exp2       M0 Frozen target-relative state + Frozen forecast       frozen_predictions=6
A_Exp1_to_Exp2       M1                               Plateau detector   plateau_locked_windows=0
A_Exp1_to_Exp2       M2                   Causal plateau-exit detector   exit_confirmed_windows=0
A_Exp1_to_Exp2       M3                        Online severe direction severe_available_windows=0
A_Exp1_to_Exp2       M4                              Robust online RLS   delayed_rls_updates=4842
A_Exp1_to_Exp2       M5                     Safe ensemble and rollback          PASS; resets=4880
B_Exp2_to_Exp1       M0 Frozen target-relative state + Frozen forecast       frozen_predictions=6
B_Exp2_to_Exp1       M1                               Plateau detector   plateau_locked_windows=0
B_Exp2_to_Exp1       M2                   Causal plateau-exit detector   exit_confirmed_windows=0
B_Exp2_to_Exp1       M3                        Online severe direction severe_available_windows=0
B_Exp2_to_Exp1       M4                              Robust online RLS  delayed_rls_updates=32319
B_Exp2_to_Exp1       M5                     Safe ensemble and rollback          PASS; resets=1723
```

12–14. Frozen, robust RLS, and Safe Ensemble 500/1000-cycle errors:

```text
   protocol_id output_name  horizon_cycles             model       MAE      RMSE  direction_accuracy
A_Exp1_to_Exp2     D_state             500      Frozen_Ridge 12.984037 16.177210            0.633432
A_Exp1_to_Exp2     D_state             500 Robust_Online_RLS 20.928914 35.105418            0.594144
A_Exp1_to_Exp2     D_state             500     Safe_Ensemble 13.157660 16.327008            0.620830
A_Exp1_to_Exp2     D_state            1000      Frozen_Ridge 14.174500 17.514132            0.673210
A_Exp1_to_Exp2     D_state            1000 Robust_Online_RLS 28.135205 50.335256            0.644342
A_Exp1_to_Exp2     D_state            1000     Safe_Ensemble 14.174500 17.514132            0.673210
A_Exp1_to_Exp2    V50_norm             500      Frozen_Ridge  2.104691  3.197141            0.719422
A_Exp1_to_Exp2    V50_norm             500 Robust_Online_RLS  2.685188  4.600038            0.704596
A_Exp1_to_Exp2    V50_norm             500     Safe_Ensemble  1.954652  2.879980            0.726093
A_Exp1_to_Exp2    V50_norm            1000      Frozen_Ridge  2.465794  3.756249            0.597383
A_Exp1_to_Exp2    V50_norm            1000 Robust_Online_RLS  2.226199  3.197566            0.698229
A_Exp1_to_Exp2    V50_norm            1000     Safe_Ensemble  2.168466  3.177682            0.697844
B_Exp2_to_Exp1     D_state             500      Frozen_Ridge  7.957461  8.340868            0.450534
B_Exp2_to_Exp1     D_state             500 Robust_Online_RLS  2.415095  4.639155            0.449311
B_Exp2_to_Exp1     D_state             500     Safe_Ensemble  2.388272  4.273168            0.450756
B_Exp2_to_Exp1     D_state            1000      Frozen_Ridge  7.072218  7.674534            0.446268
B_Exp2_to_Exp1     D_state            1000 Robust_Online_RLS  2.408371  4.017493            0.464816
B_Exp2_to_Exp1     D_state            1000     Safe_Ensemble  2.734586  4.497198            0.467626
B_Exp2_to_Exp1    V50_norm             500      Frozen_Ridge  3.082818  3.241852            0.416407
B_Exp2_to_Exp1    V50_norm             500 Robust_Online_RLS  0.721808  1.612018            0.505336
B_Exp2_to_Exp1    V50_norm             500     Safe_Ensemble  0.768443  1.671790            0.496109
B_Exp2_to_Exp1    V50_norm            1000      Frozen_Ridge  3.044354  3.208609            0.433341
B_Exp2_to_Exp1    V50_norm            1000 Robust_Online_RLS  0.835287  1.687251            0.547999
B_Exp2_to_Exp1    V50_norm            1000     Safe_Ensemble  0.856949  1.710875            0.531812
```

15. SAFE_ONLINE_FORECAST_BENEFIT:

```text
   protocol_id  main_output_horizon_count  safe_no_major_mae_harm  safe_no_major_rmse_harm                            improved_output_horizons SAFE_ONLINE_FORECAST_BENEFIT
A_Exp1_to_Exp2                          4                    True                     True                                       V50_norm@1000                         PASS
B_Exp2_to_Exp1                          4                    True                     True D_state@500;D_state@1000;V50_norm@500;V50_norm@1000                         PASS
```

16. These internal state-trend forecasts do not establish more accurate prediction of true physical wear. Independent morphology, debris, and mass-loss evidence remains required.
17. Plateau lock/exit and severe-candidate conclusions require morphology, wear-debris, and mass-loss validation before physical interpretation.
18. Stable wear is a relative low-speed platform, not a final state. Severe wear is a possible later instability/progression after that platform; no Stage1–Stage5 classifier is used.

## Acceptance

```text
implementation=PASS
prefix_causality=PASS
safe_ensemble_rollback=PASS
```
