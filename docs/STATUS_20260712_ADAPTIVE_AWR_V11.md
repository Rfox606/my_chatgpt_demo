# STATUS 20260712 Adaptive AWR v1.1

Overall acceptance: **FAIL**.

| direction_id | source_dataset | target_dataset | model | watch_logit_threshold | high_logit_threshold | Stage5_AUROC | Stage5_AUPRC | Stage5_Recall_at_high | Stage4to5_Recall_at_watch | Stage1to2_FPR_at_high | Stage1to2_FPR_at_watch | Recall_at_10pct_Stage1to2_FPR | Risk_Stage_Spearman | soft_target_brier | first_WATCH_cycle | first_HIGH_cycle | lead_cycles_relative_to_Stage5 | false_HIGH_episodes_per_1000_cycles | watch_occupancy | high_occupancy | update_episode_count | update_window_count | freeze_episode_count | frozen_window_count | rollback_count | ONLINE_ADAPTATION_NOT_EXERCISED | Stage5_risk_suppression_R2_minus_R5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1_to_Exp2 | Exp1 | Exp2 | R0 | 2.7525 | 3.4821 | 0.9906 | 0.9430 | 0.9958 | 0.7222 | 0.0000 | 0.0334 | 1.0000 | 0.8237 | 0.1527 | 3610.5000 | 9310.5000 | 2385.0000 | 0.0000 | 0.2911 | 0.2233 | 0 | 0 | 0 | 0 | 0 | True | 0.0048 |
| Exp1_to_Exp2 | Exp1 | Exp2 | R1 | 0.6693 | 1.6267 | 0.9938 | 0.9557 | 0.9958 | 0.9667 | 0.0000 | 0.2865 | 1.0000 | 0.8207 | 0.1192 | 3015.5000 | 9595.5000 | 2100.0000 | 0.0000 | 0.4966 | 0.1949 | 5 | 128 | 0 | 0 | 0 | False | 0.0048 |
| Exp1_to_Exp2 | Exp1 | Exp2 | R2 | 0.6693 | 1.6267 | 0.9938 | 0.9557 | 1.0000 | 1.0000 | 0.4561 | 0.5990 | 1.0000 | 0.8207 | 0.2287 | 10.5000 | 2705.5000 | 8990.0000 | 0.6667 | 0.7916 | 0.5964 | 1 | 2 | 0 | 0 | 0 | False | 0.0048 |
| Exp1_to_Exp2 | Exp1 | Exp2 | R3 | 0.6693 | 1.6267 | 0.9808 | 0.8642 | 1.0000 | 1.0000 | 0.3910 | 0.5990 | 1.0000 | 0.8193 | 0.2206 | 10.5000 | 2710.5000 | 8985.0000 | 1.3333 | 0.7725 | 0.5605 | 1 | 28 | 0 | 0 | 0 | False | 0.0048 |
| Exp1_to_Exp2 | Exp1 | Exp2 | R4 | 0.6693 | 1.6267 | 0.9808 | 0.8643 | 1.0000 | 1.0000 | 0.3901 | 0.5990 | 1.0000 | 0.8192 | 0.2205 | 10.5000 | 2710.5000 | 8985.0000 | 1.3333 | 0.7725 | 0.5598 | 1 | 28 | 0 | 0 | 0 | False | 0.0048 |
| Exp1_to_Exp2 | Exp1 | Exp2 | R5 | 0.6693 | 1.6267 | 0.9808 | 0.8643 | 1.0000 | 1.0000 | 0.3901 | 0.5990 | 1.0000 | 0.8193 | 0.2205 | 10.5000 | 2710.5000 | 8985.0000 | 1.3333 | 0.7725 | 0.5598 | 1 | 28 | 4 | 251 | 0 | False | 0.0048 |
| Exp1_to_Exp2 | Exp1 | Exp2 | V1_B4_REF | 1.3863 | 13.1452 | 0.9970 | 0.9888 | 0.2833 | 0.5917 | 0.0000 | 0.0000 | 1.0000 | 0.5440 | 0.0956 | 9640.5000 | 12640.5000 | -945.0000 | 0.0000 | 0.2268 | 0.0483 | 0 | 0 | 0 | 0 | 0 | True | 0.0048 |
| Exp2_to_Exp1 | Exp2 | Exp1 | R0 | 2.8612 | 4.4106 | 0.8443 | 0.6190 | 0.0000 | 0.1932 | 0.0000 | 0.0000 | 0.7185 | 0.7777 | 0.0603 | 34195.5000 | 34530.5000 | 60.0000 | 0.0000 | 0.0753 | 0.0010 | 0 | 0 | 0 | 0 | 0 | True | 0.0000 |
| Exp2_to_Exp1 | Exp2 | Exp1 | R1 | -0.0000 | 7.0080 | 0.8322 | 0.4715 | 0.0000 | 1.0000 | 0.0000 | 0.7778 | 0.3142 | 0.5772 | 0.1456 | 10.5000 | nan | nan | 0.0000 | 0.8971 | 0.0000 | 0 | 0 | 0 | 0 | 0 | True | 0.0000 |
| Exp2_to_Exp1 | Exp2 | Exp1 | R2 | -0.0000 | 7.0080 | 0.8322 | 0.4715 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.3142 | 0.5772 | 0.1456 | 10.5000 | nan | nan | 0.0000 | 1.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 | True | 0.0000 |
| Exp2_to_Exp1 | Exp2 | Exp1 | R3 | -0.0000 | 7.0080 | 0.8254 | 0.4640 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.2760 | 0.5425 | 0.1456 | 10.5000 | nan | nan | 0.0000 | 1.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 | True | 0.0000 |
| Exp2_to_Exp1 | Exp2 | Exp1 | R4 | -0.0000 | 7.0080 | 0.8470 | 0.4915 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.3365 | 0.5793 | 0.1456 | 10.5000 | nan | nan | 0.0000 | 1.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 | True | 0.0000 |
| Exp2_to_Exp1 | Exp2 | Exp1 | R5 | -0.0000 | 7.0080 | 0.8470 | 0.4915 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.3365 | 0.5793 | 0.1456 | 10.5000 | nan | nan | 0.0000 | 1.0000 | 0.0000 | 0 | 0 | 7 | 411 | 0 | True | 0.0000 |
| Exp2_to_Exp1 | Exp2 | Exp1 | V1_B4_REF | 1.3863 | 34.7909 | 0.8647 | 0.7837 | 0.0000 | 0.5052 | 0.0000 | 0.0647 | 0.7767 | 0.4511 | 0.1443 | 7620.5000 | nan | nan | 0.0000 | 0.2484 | 0.0000 | 0 | 0 | 0 | 0 | 0 | True | 0.0000 |

| criterion | observed | status |
| --- | --- | --- |
| R5 Stage5 Recall >= 0.85 in both directions | 0.0000 | FAIL |
| R5 Stage1-2 HIGH FPR <= 0.10 in both directions | 0.3901 | FAIL |
| R5 lead relative to Stage5 >= 0 where detected | 8985.0000 | PASS |
| R5 worst AUROC actual drop <= 0.02 | 0.0000 | PASS |
| R5 worst AUPRC actual drop <= 0.03 | 0.1275 | FAIL |
| Stage5 risk suppression R2 minus R5 <= 0.10 | 0.0048 | PASS |
| clean TES events in restart guard == 0 | 0.0000 | PASS |
| clean freeze triggers in restart guard == 0 | 0.0000 | PASS |
| all non-intercept coefficients <= 5 | 0.5438 | PASS |
| high probability threshold < 0.99 | 0.9991 | FAIL |
| watch logit threshold < high logit threshold | True | PASS |
| source validation Stage5 Recall >= 0.85 | 0.8557 | PASS |
| source validation Stage1-2 HIGH FPR <= 0.10 | 0.0000 | PASS |
| signed_delta_R5_minus_R0_worst_AUROC | 0.0027 | INFO |
| actual_drop_R0_minus_R5_worst_AUROC | 0.0000 | INFO |
| signed_delta_R5_minus_R0_worst_AUPRC | -0.1275 | INFO |
| actual_drop_R0_minus_R5_worst_AUPRC | 0.1275 | INFO |
