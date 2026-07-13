# Continuous State Monitoring v3.1 report

## Guard-aware plateau repair

The v3 reachability defect is **PASS**: the state machine measures evidence in valid observed cycles and preserves both success and failure counters in a restart guard.  Guard pause audit: **PASS**.  Plateau-reference freeze audit: **PASS**.

## Required scientific answers

1. Exp1 plateau detected: True; lock cycle: 2725.5; reference interval is recorded in `state_window_scores_v31.csv`.
2. Exp1 false plateau-exit confirmation: True; persistent severe-candidate events: 1.
3. Exp2 plateau/low-speed evidence: True; exit cycle: NOT_FOUND.
4. Exp2 target severe direction established: False; first causal severe-update cycle: NOT_FOUND.
5. Exp2 persistent severe-candidate onset: NOT_FOUND; post-hoc lead/lag to the known severe boundary: NOT_FOUND cycles.
6. Plateau-condition failure counts, Exp1: {'D_condition': 77, 'V50_condition': 273, 'V100_condition': 314, 'volatility_condition': 430}; Exp2: {'D_condition': 74, 'V50_condition': 2000, 'V100_condition': 2379, 'volatility_condition': 1399}.  These identify D, V50, V100, or volatility conditions rather than hiding a failed plateau search.
7. q50/q60/q75 sensitivity (q75 remains the fixed primary test):

```text
   protocol_id  quantile  plateau_lock_detected  plateau_lock_cycle  plateau_exit_detected  plateau_exit_cycle
A_Exp1_to_Exp2      0.50                      0                 NaN                      0                 NaN
A_Exp1_to_Exp2      0.60                      0                 NaN                      0                 NaN
A_Exp1_to_Exp2      0.75                      0                 NaN                      0                 NaN
B_Exp2_to_Exp1      0.50                      1              2965.5                      1              8225.5
B_Exp2_to_Exp1      0.60                      1              2850.5                      1              8225.5
B_Exp2_to_Exp1      0.75                      1              2725.5                      1              8225.5
```

8. Severe-direction feature weights (only causal post-exit updates):

```text
No available records.
```

## Forecast comparison

All six required models are compared below.  Unavailable instability or severe-score heads are preserved as unavailable; they are never zero-filled into training.

```text
   protocol_id        output_name  horizon_cycles             model  prediction_count       MAE      RMSE  direction_accuracy
A_Exp1_to_Exp2            D_state             500        Zero_Delta               253 12.889076 16.432985            0.000000
A_Exp1_to_Exp2            D_state             500      Local_Linear               253 13.237459 17.059420            0.509881
A_Exp1_to_Exp2            D_state             500      Kalman_Trend               253 23.874684 30.182353            0.335968
A_Exp1_to_Exp2            D_state             500      Frozen_Ridge               253 12.793130 15.886204            0.573123
A_Exp1_to_Exp2            D_state             500 Robust_Online_RLS               253 16.961834 20.328871            0.561265
A_Exp1_to_Exp2            D_state             500     Safe_Ensemble               253 12.793130 15.886204            0.573123
A_Exp1_to_Exp2            D_state            1000        Zero_Delta               243 14.611803 18.062909            0.000000
A_Exp1_to_Exp2            D_state            1000      Local_Linear               243 16.046093 20.029560            0.477366
A_Exp1_to_Exp2            D_state            1000      Kalman_Trend               243 40.467946 50.103245            0.271605
A_Exp1_to_Exp2            D_state            1000      Frozen_Ridge               243 13.289615 16.574153            0.695473
A_Exp1_to_Exp2            D_state            1000 Robust_Online_RLS               243 16.378373 21.490835            0.728395
A_Exp1_to_Exp2            D_state            1000     Safe_Ensemble               243 13.289615 16.574153            0.695473
A_Exp1_to_Exp2           V50_norm             500        Zero_Delta               253  2.410525  3.561541            0.000000
A_Exp1_to_Exp2           V50_norm             500      Local_Linear               253  2.436149  3.564789            0.478261
A_Exp1_to_Exp2           V50_norm             500      Kalman_Trend               253  5.502496  7.680008            0.217391
A_Exp1_to_Exp2           V50_norm             500      Frozen_Ridge               253  5.627485  6.900920            0.501976
A_Exp1_to_Exp2           V50_norm             500 Robust_Online_RLS               253  3.115790  4.135188            0.545455
A_Exp1_to_Exp2           V50_norm             500     Safe_Ensemble               253  5.414396  6.795455            0.498024
A_Exp1_to_Exp2           V50_norm            1000        Zero_Delta               243  2.511741  3.615525            0.000000
A_Exp1_to_Exp2           V50_norm            1000      Local_Linear               243  2.559939  3.640095            0.497942
A_Exp1_to_Exp2           V50_norm            1000      Kalman_Trend               243  8.769488 12.022163            0.255144
A_Exp1_to_Exp2           V50_norm            1000      Frozen_Ridge               243  3.844534  4.802369            0.539095
A_Exp1_to_Exp2           V50_norm            1000 Robust_Online_RLS               243  2.735548  3.563579            0.572016
A_Exp1_to_Exp2           V50_norm            1000     Safe_Ensemble               243  3.827732  4.794745            0.555556
B_Exp2_to_Exp1            D_state             500        Zero_Delta               883  0.722806  1.530505            0.168743
B_Exp2_to_Exp1            D_state             500      Local_Linear               883  0.764023  1.527509            0.353341
B_Exp2_to_Exp1            D_state             500      Kalman_Trend               883  1.203660  2.395651            0.375991
B_Exp2_to_Exp1            D_state             500      Frozen_Ridge               883  3.926942  5.188364            0.456399
B_Exp2_to_Exp1            D_state             500 Robust_Online_RLS               883  1.686790  2.490023            0.455266
B_Exp2_to_Exp1            D_state             500     Safe_Ensemble               883  1.997713  2.812488            0.457531
B_Exp2_to_Exp1            D_state            1000        Zero_Delta               873  1.036871  1.991315            0.150057
B_Exp2_to_Exp1            D_state            1000      Local_Linear               873  1.159263  1.989959            0.337915
B_Exp2_to_Exp1            D_state            1000      Kalman_Trend               873  2.172925  4.024706            0.387171
B_Exp2_to_Exp1            D_state            1000      Frozen_Ridge               873  3.019072  3.820012            0.509737
B_Exp2_to_Exp1            D_state            1000 Robust_Online_RLS               873  2.376364  3.514890            0.447881
B_Exp2_to_Exp1            D_state            1000     Safe_Ensemble               873  2.539712  3.497755            0.497136
B_Exp2_to_Exp1 S_severe_candidate             500        Zero_Delta               847  1.567932  3.805925            0.151122
B_Exp2_to_Exp1 S_severe_candidate             500      Local_Linear               847  1.681469  3.851461            0.296340
B_Exp2_to_Exp1 S_severe_candidate             500      Kalman_Trend               847  2.540308  5.909162            0.386068
B_Exp2_to_Exp1 S_severe_candidate             500      Frozen_Ridge               847  3.391879  4.914614            0.422668
B_Exp2_to_Exp1 S_severe_candidate             500 Robust_Online_RLS               847  2.245710  4.470524            0.399055
B_Exp2_to_Exp1 S_severe_candidate             500     Safe_Ensemble               847  2.365303  4.466848            0.396694
B_Exp2_to_Exp1 S_severe_candidate            1000        Zero_Delta               837  2.227153  4.868575            0.123059
B_Exp2_to_Exp1 S_severe_candidate            1000      Local_Linear               837  2.469693  4.950414            0.290323
B_Exp2_to_Exp1 S_severe_candidate            1000      Kalman_Trend               837  4.565203 10.169621            0.438471
B_Exp2_to_Exp1 S_severe_candidate            1000      Frozen_Ridge               837  4.711024  6.640134            0.428913
B_Exp2_to_Exp1 S_severe_candidate            1000 Robust_Online_RLS               837  4.219788  6.504823            0.461171
B_Exp2_to_Exp1 S_severe_candidate            1000     Safe_Ensemble               837  4.420739  6.394457            0.422939
B_Exp2_to_Exp1           V50_norm             500        Zero_Delta               883  0.675017  1.826696            0.160815
B_Exp2_to_Exp1           V50_norm             500      Local_Linear               883  0.712952  1.857039            0.272933
B_Exp2_to_Exp1           V50_norm             500      Kalman_Trend               883  1.353623  3.651490            0.314836
B_Exp2_to_Exp1           V50_norm             500      Frozen_Ridge               883  2.982384  3.303031            0.443941
B_Exp2_to_Exp1           V50_norm             500 Robust_Online_RLS               883  1.096534  2.000310            0.499434
B_Exp2_to_Exp1           V50_norm             500     Safe_Ensemble               883  1.486011  2.342921            0.471121
B_Exp2_to_Exp1           V50_norm            1000        Zero_Delta               873  0.762150  1.973378            0.132875
B_Exp2_to_Exp1           V50_norm            1000      Local_Linear               873  0.847396  2.068959            0.254296
B_Exp2_to_Exp1           V50_norm            1000      Kalman_Trend               873  2.256250  5.999560            0.358534
B_Exp2_to_Exp1           V50_norm            1000      Frozen_Ridge               873  1.724770  2.469765            0.550974
B_Exp2_to_Exp1           V50_norm            1000 Robust_Online_RLS               873  1.378292  2.233547            0.532646
B_Exp2_to_Exp1           V50_norm            1000     Safe_Ensemble               873  1.398957  2.134358            0.552119
B_Exp2_to_Exp1  instability_score             500        Zero_Delta               847  2.776586  7.012941            0.147580
B_Exp2_to_Exp1  instability_score             500      Local_Linear               847  2.881555  7.094442            0.334120
B_Exp2_to_Exp1  instability_score             500      Kalman_Trend               847  4.405765 10.791943            0.404959
B_Exp2_to_Exp1  instability_score             500      Frozen_Ridge               847  8.976282 10.550012            0.416765
B_Exp2_to_Exp1  instability_score             500 Robust_Online_RLS               847  4.545876  9.031909            0.449823
B_Exp2_to_Exp1  instability_score             500     Safe_Ensemble               847  4.995183  8.967122            0.448642
B_Exp2_to_Exp1  instability_score            1000        Zero_Delta               837  3.994379  9.040028            0.119474
B_Exp2_to_Exp1  instability_score            1000      Local_Linear               837  4.211657  9.156853            0.317802
B_Exp2_to_Exp1  instability_score            1000      Kalman_Trend               837  8.061141 18.653121            0.445639
B_Exp2_to_Exp1  instability_score            1000      Frozen_Ridge               837 13.410523 15.337036            0.455197
B_Exp2_to_Exp1  instability_score            1000 Robust_Online_RLS               837  6.521110 11.249203            0.430108
B_Exp2_to_Exp1  instability_score            1000     Safe_Ensemble               837  7.054935 11.242409            0.440860
```

Early/middle/late performance:

```text
   protocol_id        output_name  horizon_cycles segment             model       MAE      RMSE  direction_accuracy
A_Exp1_to_Exp2            D_state             500   early        Zero_Delta 12.862273 15.037702            0.000000
A_Exp1_to_Exp2            D_state             500   early      Local_Linear 13.053959 15.456377            0.541176
A_Exp1_to_Exp2            D_state             500   early      Kalman_Trend 23.608680 27.500485            0.317647
A_Exp1_to_Exp2            D_state             500   early      Frozen_Ridge 11.538652 14.054779            0.552941
A_Exp1_to_Exp2            D_state             500   early Robust_Online_RLS 12.635181 15.322482            0.694118
A_Exp1_to_Exp2            D_state             500   early     Safe_Ensemble 11.538652 14.054779            0.552941
A_Exp1_to_Exp2            D_state             500  middle        Zero_Delta 14.190310 18.950039            0.000000
A_Exp1_to_Exp2            D_state             500  middle      Local_Linear 14.262373 19.330514            0.511905
A_Exp1_to_Exp2            D_state             500  middle      Kalman_Trend 28.859246 37.328284            0.273810
A_Exp1_to_Exp2            D_state             500  middle      Frozen_Ridge 17.242141 20.433356            0.500000
A_Exp1_to_Exp2            D_state             500  middle Robust_Online_RLS 19.598266 22.914137            0.511905
A_Exp1_to_Exp2            D_state             500  middle     Safe_Ensemble 17.242141 20.433356            0.500000
A_Exp1_to_Exp2            D_state             500    late        Zero_Delta 11.614964 15.013836            0.000000
A_Exp1_to_Exp2            D_state             500    late      Local_Linear 12.398229 16.159321            0.476190
A_Exp1_to_Exp2            D_state             500    late      Kalman_Trend 19.159294 24.188583            0.416667
A_Exp1_to_Exp2            D_state             500    late      Frozen_Ridge  9.613531 11.946061            0.666667
A_Exp1_to_Exp2            D_state             500    late Robust_Online_RLS 18.703562 21.956268            0.476190
A_Exp1_to_Exp2            D_state             500    late     Safe_Ensemble  9.613531 11.946061            0.666667
A_Exp1_to_Exp2            D_state            1000   early        Zero_Delta 11.418426 14.291956            0.000000
A_Exp1_to_Exp2            D_state            1000   early      Local_Linear 12.805305 15.842868            0.432099
A_Exp1_to_Exp2            D_state            1000   early      Kalman_Trend 36.802118 42.885560            0.259259
A_Exp1_to_Exp2            D_state            1000   early      Frozen_Ridge 10.092805 12.326128            0.703704
A_Exp1_to_Exp2            D_state            1000   early Robust_Online_RLS 10.515699 12.209974            0.679012
A_Exp1_to_Exp2            D_state            1000   early     Safe_Ensemble 10.092805 12.326128            0.703704
A_Exp1_to_Exp2            D_state            1000  middle        Zero_Delta 16.588865 19.404786            0.000000
A_Exp1_to_Exp2            D_state            1000  middle      Local_Linear 17.671933 20.617815            0.456790
A_Exp1_to_Exp2            D_state            1000  middle      Kalman_Trend 43.973839 58.572533            0.345679
A_Exp1_to_Exp2            D_state            1000  middle      Frozen_Ridge 15.766728 19.228604            0.666667
A_Exp1_to_Exp2            D_state            1000  middle Robust_Online_RLS 14.842474 19.053657            0.740741
A_Exp1_to_Exp2            D_state            1000  middle     Safe_Ensemble 15.766728 19.228604            0.666667
A_Exp1_to_Exp2            D_state            1000    late        Zero_Delta 15.828117 19.949946            0.000000
A_Exp1_to_Exp2            D_state            1000    late      Local_Linear 17.661040 22.966475            0.543210
A_Exp1_to_Exp2            D_state            1000    late      Kalman_Trend 40.627880 47.550948            0.209877
A_Exp1_to_Exp2            D_state            1000    late      Frozen_Ridge 14.009312 17.390659            0.716049
A_Exp1_to_Exp2            D_state            1000    late Robust_Online_RLS 23.776946 29.554063            0.765432
A_Exp1_to_Exp2            D_state            1000    late     Safe_Ensemble 14.009312 17.390659            0.716049
A_Exp1_to_Exp2           V50_norm             500   early        Zero_Delta  1.965529  2.809417            0.000000
A_Exp1_to_Exp2           V50_norm             500   early      Local_Linear  1.966571  2.807671            0.470588
A_Exp1_to_Exp2           V50_norm             500   early      Kalman_Trend  4.451852  6.028330            0.211765
A_Exp1_to_Exp2           V50_norm             500   early      Frozen_Ridge  4.651866  5.553971            0.447059
A_Exp1_to_Exp2           V50_norm             500   early Robust_Online_RLS  3.503342  4.626736            0.447059
A_Exp1_to_Exp2           V50_norm             500   early     Safe_Ensemble  4.651866  5.553971            0.447059
A_Exp1_to_Exp2           V50_norm             500  middle        Zero_Delta  2.038304  3.079477            0.000000
A_Exp1_to_Exp2           V50_norm             500  middle      Local_Linear  2.039708  3.076949            0.500000
A_Exp1_to_Exp2           V50_norm             500  middle      Kalman_Trend  4.431278  6.419871            0.226190
A_Exp1_to_Exp2           V50_norm             500  middle      Frozen_Ridge  6.708179  8.033079            0.535714
A_Exp1_to_Exp2           V50_norm             500  middle Robust_Online_RLS  2.364052  3.027037            0.595238
A_Exp1_to_Exp2           V50_norm             500  middle     Safe_Ensemble  6.708179  8.033079            0.535714
A_Exp1_to_Exp2           V50_norm             500    late        Zero_Delta  3.233038  4.553545            0.000000
A_Exp1_to_Exp2           V50_norm             500    late      Local_Linear  3.307758  4.563987            0.464286
A_Exp1_to_Exp2           V50_norm             500    late      Kalman_Trend  7.636865  9.983065            0.214286
A_Exp1_to_Exp2           V50_norm             500    late      Frozen_Ridge  5.534025  6.905857            0.523810
A_Exp1_to_Exp2           V50_norm             500    late Robust_Online_RLS  3.475362  4.547354            0.595238
A_Exp1_to_Exp2           V50_norm             500    late     Safe_Ensemble  4.892220  6.583328            0.511905
A_Exp1_to_Exp2           V50_norm            1000   early        Zero_Delta  2.028864  2.771362            0.000000
A_Exp1_to_Exp2           V50_norm            1000   early      Local_Linear  2.036167  2.792721            0.481481
A_Exp1_to_Exp2           V50_norm            1000   early      Kalman_Trend  7.316709  9.592439            0.234568
A_Exp1_to_Exp2           V50_norm            1000   early      Frozen_Ridge  3.118035  3.701049            0.617284
A_Exp1_to_Exp2           V50_norm            1000   early Robust_Online_RLS  2.601963  3.279311            0.530864
A_Exp1_to_Exp2           V50_norm            1000   early     Safe_Ensemble  3.118035  3.701049            0.617284
A_Exp1_to_Exp2           V50_norm            1000  middle        Zero_Delta  2.280549  3.319363            0.000000
A_Exp1_to_Exp2           V50_norm            1000  middle      Local_Linear  2.268047  3.308319            0.567901
A_Exp1_to_Exp2           V50_norm            1000  middle      Kalman_Trend  7.286536 10.300268            0.246914
A_Exp1_to_Exp2           V50_norm            1000  middle      Frozen_Ridge  4.180558  5.090938            0.567901
A_Exp1_to_Exp2           V50_norm            1000  middle Robust_Online_RLS  2.219779  2.992344            0.629630
A_Exp1_to_Exp2           V50_norm            1000  middle     Safe_Ensemble  4.180558  5.090938            0.567901
A_Exp1_to_Exp2           V50_norm            1000    late        Zero_Delta  3.225810  4.529620            0.000000
A_Exp1_to_Exp2           V50_norm            1000    late      Local_Linear  3.375601  4.583296            0.444444
A_Exp1_to_Exp2           V50_norm            1000    late      Kalman_Trend 11.705219 15.345579            0.283951
A_Exp1_to_Exp2           V50_norm            1000    late      Frozen_Ridge  4.235008  5.438091            0.432099
A_Exp1_to_Exp2           V50_norm            1000    late Robust_Online_RLS  3.384903  4.288273            0.555556
A_Exp1_to_Exp2           V50_norm            1000    late     Safe_Ensemble  4.184602  5.417870            0.481481
B_Exp2_to_Exp1            D_state             500   early        Zero_Delta  0.819215  1.433705            0.010169
B_Exp2_to_Exp1            D_state             500   early      Local_Linear  0.909167  1.386238            0.515254
B_Exp2_to_Exp1            D_state             500   early      Kalman_Trend  1.251097  1.908387            0.525424
B_Exp2_to_Exp1            D_state             500   early      Frozen_Ridge  2.937744  3.634018            0.596610
B_Exp2_to_Exp1            D_state             500   early Robust_Online_RLS  1.862252  2.669620            0.583051
B_Exp2_to_Exp1            D_state             500   early     Safe_Ensemble  2.792913  3.499272            0.589831
B_Exp2_to_Exp1            D_state             500  middle        Zero_Delta  0.619762  1.242881            0.404762
B_Exp2_to_Exp1            D_state             500  middle      Local_Linear  0.619762  1.242881            0.404762
B_Exp2_to_Exp1            D_state             500  middle      Kalman_Trend  1.134629  2.120640            0.163265
B_Exp2_to_Exp1            D_state             500  middle      Frozen_Ridge  5.689709  7.050615            0.319728
B_Exp2_to_Exp1            D_state             500  middle Robust_Online_RLS  1.884224  2.789827            0.306122
B_Exp2_to_Exp1            D_state             500  middle     Safe_Ensemble  1.884224  2.789827            0.306122
B_Exp2_to_Exp1            D_state             500    late        Zero_Delta  0.729114  1.851498            0.091837
B_Exp2_to_Exp1            D_state             500    late      Local_Linear  0.762647  1.880117            0.139456
B_Exp2_to_Exp1            D_state             500    late      Kalman_Trend  1.225093  3.014217            0.438776
B_Exp2_to_Exp1            D_state             500    late      Frozen_Ridge  3.156736  4.229272            0.452381
B_Exp2_to_Exp1            D_state             500    late Robust_Online_RLS  1.313297  1.920282            0.476190
B_Exp2_to_Exp1            D_state             500    late     Safe_Ensemble  1.313297  1.920282            0.476190
B_Exp2_to_Exp1            D_state            1000   early        Zero_Delta  1.293986  2.212939            0.003436
B_Exp2_to_Exp1            D_state            1000   early      Local_Linear  1.581474  2.137961            0.522337
B_Exp2_to_Exp1            D_state            1000   early      Kalman_Trend  2.395887  3.364401            0.529210
B_Exp2_to_Exp1            D_state            1000   early      Frozen_Ridge  2.381236  2.835931            0.601375
B_Exp2_to_Exp1            D_state            1000   early Robust_Online_RLS  2.245734  3.044804            0.512027
B_Exp2_to_Exp1            D_state            1000   early     Safe_Ensemble  2.381236  2.835931            0.601375
B_Exp2_to_Exp1            D_state            1000  middle        Zero_Delta  0.769716  1.414362            0.371134
B_Exp2_to_Exp1            D_state            1000  middle      Local_Linear  0.769716  1.414362            0.371134
B_Exp2_to_Exp1            D_state            1000  middle      Kalman_Trend  1.867805  3.465576            0.199313
B_Exp2_to_Exp1            D_state            1000  middle      Frozen_Ridge  3.580268  4.357081            0.422680
B_Exp2_to_Exp1            D_state            1000  middle Robust_Online_RLS  3.045459  4.172876            0.384880
B_Exp2_to_Exp1            D_state            1000  middle     Safe_Ensemble  3.400003  4.275599            0.443299
B_Exp2_to_Exp1            D_state            1000    late        Zero_Delta  1.046910  2.235729            0.075601
B_Exp2_to_Exp1            D_state            1000    late      Local_Linear  1.126598  2.304020            0.120275
B_Exp2_to_Exp1            D_state            1000    late      Kalman_Trend  2.255082  5.026466            0.432990
B_Exp2_to_Exp1            D_state            1000    late      Frozen_Ridge  3.095712  4.092776            0.505155
B_Exp2_to_Exp1            D_state            1000    late Robust_Online_RLS  1.837898  3.221742            0.446735
B_Exp2_to_Exp1            D_state            1000    late     Safe_Ensemble  1.837898  3.221742            0.446735
B_Exp2_to_Exp1 S_severe_candidate             500   early        Zero_Delta  0.874705  1.604988            0.053004
B_Exp2_to_Exp1 S_severe_candidate             500   early      Local_Linear  1.110189  1.670452            0.466431
B_Exp2_to_Exp1 S_severe_candidate             500   early      Kalman_Trend  1.682178  2.680993            0.363958
B_Exp2_to_Exp1 S_severe_candidate             500   early      Frozen_Ridge  2.504448  3.029217            0.462898
B_Exp2_to_Exp1 S_severe_candidate             500   early Robust_Online_RLS  2.033711  2.937329            0.459364
B_Exp2_to_Exp1 S_severe_candidate             500   early     Safe_Ensemble  2.391643  2.920544            0.452297
B_Exp2_to_Exp1 S_severe_candidate             500  middle        Zero_Delta  1.350803  3.057864            0.333333
B_Exp2_to_Exp1 S_severe_candidate             500  middle      Local_Linear  1.352678  3.057891            0.312057
B_Exp2_to_Exp1 S_severe_candidate             500  middle      Kalman_Trend  2.247528  5.450049            0.290780
B_Exp2_to_Exp1 S_severe_candidate             500  middle      Frozen_Ridge  3.364503  4.496880            0.336879
B_Exp2_to_Exp1 S_severe_candidate             500  middle Robust_Online_RLS  1.532235  3.013120            0.297872
B_Exp2_to_Exp1 S_severe_candidate             500  middle     Safe_Ensemble  1.532235  3.013120            0.297872
```

The rolling metrics and cumulative-regret tables are saved in their full temporal order.  The final available Safe Ensemble regrets are:

```text
 due_observation_cycle    protocol_id        output_name  horizon_cycles baseline_model  instantaneous_regret  cumulative_regret
               14060.5 A_Exp1_to_Exp2            D_state             100   Kalman_Trend             -0.982103        -234.768997
               14060.5 A_Exp1_to_Exp2            D_state             500   Local_Linear             -4.321683        -112.415241
               14060.5 A_Exp1_to_Exp2            D_state             500   Frozen_Ridge              0.000000           0.000000
               14060.5 A_Exp1_to_Exp2            D_state             500   Kalman_Trend              0.700184       -2803.633319
               14060.5 A_Exp1_to_Exp2            D_state             500     Zero_Delta             -2.510927         -24.274300
               14060.5 A_Exp1_to_Exp2            D_state             100   Local_Linear              1.217302         154.128308
               14060.5 A_Exp1_to_Exp2            D_state             100   Frozen_Ridge              0.000000           0.000000
               14060.5 A_Exp1_to_Exp2           V50_norm            1000   Frozen_Ridge             -0.395448          -4.082873
               14060.5 A_Exp1_to_Exp2           V50_norm             500   Local_Linear              0.153299         753.496482
               14060.5 A_Exp1_to_Exp2           V50_norm            1000   Kalman_Trend             -3.133465       -1200.846666
               14060.5 A_Exp1_to_Exp2            D_state             100     Zero_Delta              0.869802         157.748601
               14060.5 A_Exp1_to_Exp2           V50_norm             500   Frozen_Ridge             -3.337436         -53.911552
               14060.5 A_Exp1_to_Exp2           V50_norm             500   Kalman_Trend             -0.327501         -22.289213
               14060.5 A_Exp1_to_Exp2           V50_norm             100   Local_Linear              1.836678         555.745762
               14060.5 A_Exp1_to_Exp2           V50_norm             100     Zero_Delta              1.811960         556.885883
               14060.5 A_Exp1_to_Exp2           V50_norm             100   Frozen_Ridge             -2.509972        -117.470286
               14060.5 A_Exp1_to_Exp2           V50_norm             500     Zero_Delta              1.131481         759.979493
               14060.5 A_Exp1_to_Exp2           V50_norm             100   Kalman_Trend              1.838057         438.284008
               14060.5 A_Exp1_to_Exp2           V50_norm            1000     Zero_Delta             -0.248280         319.785764
               14060.5 A_Exp1_to_Exp2           V50_norm            1000   Local_Linear              0.485703         308.073712
               14060.5 A_Exp1_to_Exp2            D_state            1000     Zero_Delta              3.777183        -321.291610
               14060.5 A_Exp1_to_Exp2            D_state            1000   Frozen_Ridge              0.000000           0.000000
               14060.5 A_Exp1_to_Exp2            D_state            1000   Kalman_Trend            -10.647538       -6604.334321
               14060.5 A_Exp1_to_Exp2            D_state            1000   Local_Linear              2.832751        -669.824003
               45560.5 B_Exp2_to_Exp1 S_severe_candidate             500   Frozen_Ridge             -1.697347        -869.510184
               45560.5 B_Exp2_to_Exp1  instability_score             500   Frozen_Ridge             -5.470077       -3371.990918
               45560.5 B_Exp2_to_Exp1 S_severe_candidate            1000     Zero_Delta              0.618444        1836.031710
               45560.5 B_Exp2_to_Exp1 S_severe_candidate            1000   Local_Linear              0.618444        1633.025594
               45560.5 B_Exp2_to_Exp1 S_severe_candidate            1000   Kalman_Trend              0.610459        -120.915744
               45560.5 B_Exp2_to_Exp1 S_severe_candidate            1000   Frozen_Ridge             -5.469393        -242.968449
               45560.5 B_Exp2_to_Exp1  instability_score             500   Kalman_Trend             -2.302255         499.237207
               45560.5 B_Exp2_to_Exp1           V50_norm             100     Zero_Delta              0.102470         899.317525
               45560.5 B_Exp2_to_Exp1           V50_norm             100   Local_Linear              0.102470         895.076122
               45560.5 B_Exp2_to_Exp1           V50_norm             100   Frozen_Ridge             -2.030410       -1769.560145
               45560.5 B_Exp2_to_Exp1           V50_norm             500   Local_Linear              0.010103         682.611260
               45560.5 B_Exp2_to_Exp1           V50_norm             500     Zero_Delta              0.010103         716.108078
               45560.5 B_Exp2_to_Exp1           V50_norm             100   Kalman_Trend              0.079388         808.245392
               45560.5 B_Exp2_to_Exp1           V50_norm             500   Frozen_Ridge             -1.894104       -1321.296794
               45560.5 B_Exp2_to_Exp1  instability_score             500   Local_Linear              0.021515        1790.243154
               45560.5 B_Exp2_to_Exp1           V50_norm             500   Kalman_Trend             -0.223151         116.898772
               45560.5 B_Exp2_to_Exp1           V50_norm            1000   Frozen_Ridge             -1.201231        -284.434491
               45560.5 B_Exp2_to_Exp1  instability_score             500     Zero_Delta              0.021515        1879.151446
               45560.5 B_Exp2_to_Exp1           V50_norm            1000   Kalman_Trend             -0.145495        -748.416775
               45560.5 B_Exp2_to_Exp1  instability_score             100   Local_Linear             -0.168084         990.871473
               45560.5 B_Exp2_to_Exp1  instability_score             100   Frozen_Ridge             -2.617352       -2250.784406
               45560.5 B_Exp2_to_Exp1  instability_score            1000   Frozen_Ridge            -13.069753       -5319.627557
               45560.5 B_Exp2_to_Exp1  instability_score             100   Kalman_Trend             -0.020526         902.819912
               45560.5 B_Exp2_to_Exp1  instability_score             100     Zero_Delta             -0.168084        1003.341756
               45560.5 B_Exp2_to_Exp1           V50_norm            1000     Zero_Delta             -0.002020         555.932620
               45560.5 B_Exp2_to_Exp1           V50_norm            1000   Local_Linear             -0.002020         481.512723
               45560.5 B_Exp2_to_Exp1 S_severe_candidate             500     Zero_Delta              0.029037         675.373236
               45560.5 B_Exp2_to_Exp1 S_severe_candidate             500   Local_Linear              0.029037         579.207817
               45560.5 B_Exp2_to_Exp1  instability_score            1000     Zero_Delta             -0.332153        2561.684614
               45560.5 B_Exp2_to_Exp1 S_severe_candidate             100   Frozen_Ridge             -1.282307       -1242.519689
               45560.5 B_Exp2_to_Exp1 S_severe_candidate             500   Kalman_Trend             -1.234124        -148.228916
               45560.5 B_Exp2_to_Exp1 S_severe_candidate             100   Kalman_Trend             -0.071232         470.612628
               45560.5 B_Exp2_to_Exp1            D_state            1000     Zero_Delta              0.055730        1311.980817
               45560.5 B_Exp2_to_Exp1  instability_score            1000   Local_Linear             -0.332153        2379.822970
               45560.5 B_Exp2_to_Exp1            D_state            1000   Local_Linear              0.055730        1205.132589
               45560.5 B_Exp2_to_Exp1            D_state            1000   Kalman_Trend             -0.443944         320.205338
               45560.5 B_Exp2_to_Exp1 S_severe_candidate             100     Zero_Delta             -0.149902         543.147175
               45560.5 B_Exp2_to_Exp1 S_severe_candidate             100   Local_Linear             -0.149902         532.693365
               45560.5 B_Exp2_to_Exp1            D_state            1000   Frozen_Ridge              0.382435        -418.481120
               45560.5 B_Exp2_to_Exp1            D_state             500   Frozen_Ridge             -0.065997       -1703.508797
               45560.5 B_Exp2_to_Exp1            D_state             100   Frozen_Ridge             -0.213772        -553.157705
               45560.5 B_Exp2_to_Exp1  instability_score            1000   Kalman_Trend             -0.431316        -842.194785
               45560.5 B_Exp2_to_Exp1            D_state             500     Zero_Delta              0.482075        1125.742526
               45560.5 B_Exp2_to_Exp1            D_state             500   Local_Linear              0.482075        1089.348346
               45560.5 B_Exp2_to_Exp1            D_state             500   Kalman_Trend              0.642305         701.148784
               45560.5 B_Exp2_to_Exp1            D_state             100   Kalman_Trend              0.022869         630.359484
               45560.5 B_Exp2_to_Exp1            D_state             100   Local_Linear             -0.043383         675.049311
               45560.5 B_Exp2_to_Exp1            D_state             100     Zero_Delta             -0.043383         678.890755
```

Forecast-benefit decision (Protocol A is mandatory; Protocol B is control only):

```text
   protocol_id  main_output_horizon_count  best_static_reference  safe_no_major_mae_harm  safe_no_major_rmse_harm improved_output_horizons SAFE_ONLINE_FORECAST_BENEFIT
A_Exp1_to_Exp2                          4 minimum_of_B0_B1_B2_F0                   False                    False                                                  FAIL
B_Exp2_to_Exp1                          8 minimum_of_B0_B1_B2_F0                   False                    False                                                  FAIL
```

## Safe Ensemble

Independent reset episodes: 6.  The reset audit is **PASS**; a FROZEN episode does not reset repeatedly or extend `freeze_until`.  The state monitoring table has 3360 pre-lock rows; this is an availability context, not a real wear quantity.

## Causality and limits

- Label leakage: **PASS**. Stage labels were read only after online tables were saved for post-hoc physical comparison.
- Prefix causality: **PASS**. Delayed forecast update: **PASS**. Severe-direction causality: **PASS**.
- Cache fingerprint: **PASS**. No v3 output cache was used.
- Stable wear is a relatively slow, low-volatility region displaced from the initial baseline, not a final state. Severe wear is a possible later persistent instability after that plateau. No Stage1–Stage5 classifier is used online.
- These continuous internal signals do not by themselves prove physical wear amount. Independent morphology, debris, and mass-loss/quality-loss validation (ideally a third independent experiment) is still required before claiming real wear prediction.
