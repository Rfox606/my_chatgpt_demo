# Adaptive AWR v1.1 Report

## Outcome

Overall acceptance: **FAIL**. Target stage labels are attached only after label-free sequential scoring.

## Required Questions

1. **Was v1 recall failure mainly a near-one threshold problem?** v1 source probability thresholds: Exp1_to_Exp2=0.999998, Exp2_to_Exp1=1.000000. v1.1 uses logit thresholds; its equivalently transformed high probabilities are Exp1_to_Exp2=0.835713, Exp2_to_Exp1=0.999096. The difficult direction remains saturated, so the change alone does not solve transfer.
2. **Did regularisation reduce saturation?** Yes for coefficients: maximum non-intercept coefficient is `0.5438` (bound 5). Probability-threshold saturation remains a diagnosed failure where the source validation distribution is degenerate.
3. **Did target logit alignment improve absolute transfer?** R2-R1: Exp1_to_Exp2: AUROC +0.0000, AUPRC +0.0000; Exp2_to_Exp1: AUROC +0.0000, AUPRC +0.0000. This is reported as measured, not assumed beneficial.
4. **What is R2 relative to R1?** Exp1_to_Exp2: AUROC +0.0000, AUPRC +0.0000; Exp2_to_Exp1: AUROC +0.0000, AUPRC +0.0000.
5. **Does revised reliability preserve ranking?** R3-R2: Exp1_to_Exp2: AUROC -0.0130, AUPRC -0.0914; Exp2_to_Exp1: AUROC -0.0068, AUPRC -0.0075. The report retains any degradation.
6. **Were 500-cycle pseudo-events removed?** Raw/clean guard counts are below; clean TES events and clean freeze triggers are both zero.
7. **How many independent freeze episodes occurred?** R5 freeze episodes: Exp1_to_Exp2=4.0000, Exp2_to_Exp1=7.0000; frozen windows are separately present in the summary.
8. **Did residual offset update?** R5 update episodes: Exp1_to_Exp2=1.0000, Exp2_to_Exp1=0.0000. A direction with zero is marked `ONLINE_ADAPTATION_NOT_EXERCISED`.
9. **Did updating suppress Stage5 risk?** R5 suppression R2-R5: Exp1_to_Exp2=0.0048, Exp2_to_Exp1=0.0000.
10. **Why does v1.1 fail, if it fails?** Failed acceptance checks below distinguish threshold transfer, early false positives, ranking change, and online-adaptation opportunity. No acceptance threshold has been changed.

## Bidirectional Summary

| direction_id | model | Stage5_AUROC | Stage5_AUPRC | Stage5_Recall_at_high | Stage4to5_Recall_at_watch | Stage1to2_FPR_at_high | lead_cycles_relative_to_Stage5 | update_episode_count | freeze_episode_count | Stage5_risk_suppression_R2_minus_R5 | ONLINE_ADAPTATION_NOT_EXERCISED |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1_to_Exp2 | R0 | 0.9906 | 0.9430 | 0.9958 | 0.7222 | 0.0000 | 2385.0000 | 0 | 0 | 0.0048 | True |
| Exp1_to_Exp2 | R1 | 0.9938 | 0.9557 | 0.9958 | 0.9667 | 0.0000 | 2100.0000 | 5 | 0 | 0.0048 | False |
| Exp1_to_Exp2 | R2 | 0.9938 | 0.9557 | 1.0000 | 1.0000 | 0.4561 | 8990.0000 | 1 | 0 | 0.0048 | False |
| Exp1_to_Exp2 | R3 | 0.9808 | 0.8642 | 1.0000 | 1.0000 | 0.3910 | 8985.0000 | 1 | 0 | 0.0048 | False |
| Exp1_to_Exp2 | R4 | 0.9808 | 0.8643 | 1.0000 | 1.0000 | 0.3901 | 8985.0000 | 1 | 0 | 0.0048 | False |
| Exp1_to_Exp2 | R5 | 0.9808 | 0.8643 | 1.0000 | 1.0000 | 0.3901 | 8985.0000 | 1 | 4 | 0.0048 | False |
| Exp1_to_Exp2 | V1_B4_REF | 0.9970 | 0.9888 | 0.2833 | 0.5917 | 0.0000 | -945.0000 | 0 | 0 | 0.0048 | True |
| Exp2_to_Exp1 | R0 | 0.8443 | 0.6190 | 0.0000 | 0.1932 | 0.0000 | 60.0000 | 0 | 0 | 0.0000 | True |
| Exp2_to_Exp1 | R1 | 0.8322 | 0.4715 | 0.0000 | 1.0000 | 0.0000 | nan | 0 | 0 | 0.0000 | True |
| Exp2_to_Exp1 | R2 | 0.8322 | 0.4715 | 0.0000 | 1.0000 | 0.0000 | nan | 0 | 0 | 0.0000 | True |
| Exp2_to_Exp1 | R3 | 0.8254 | 0.4640 | 0.0000 | 1.0000 | 0.0000 | nan | 0 | 0 | 0.0000 | True |
| Exp2_to_Exp1 | R4 | 0.8470 | 0.4915 | 0.0000 | 1.0000 | 0.0000 | nan | 0 | 0 | 0.0000 | True |
| Exp2_to_Exp1 | R5 | 0.8470 | 0.4915 | 0.0000 | 1.0000 | 0.0000 | nan | 0 | 7 | 0.0000 | True |
| Exp2_to_Exp1 | V1_B4_REF | 0.8647 | 0.7837 | 0.0000 | 0.5052 | 0.0000 | nan | 0 | 0 | 0.0000 | True |

## Regularization Grid

| direction_id | l2 | Stage5_AUROC | Stage5_AUPRC | Risk_Stage_Spearman | soft_target_brier | selection_score | max_abs_nonintercept_beta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1_to_Exp2 | 0.1000 | 0.9987 | 0.9957 | 0.9733 | 0.0437 | 0.9893 | 0.5438 |
| Exp1_to_Exp2 | 0.5000 | 0.9893 | 0.9417 | 0.9673 | 0.1022 | 0.9636 | 0.1746 |
| Exp1_to_Exp2 | 1.0000 | 0.9830 | 0.8708 | 0.9619 | 0.1204 | 0.9407 | 0.0948 |
| Exp1_to_Exp2 | 2.0000 | 0.9782 | 0.8286 | 0.9559 | 0.1316 | 0.9265 | 0.0496 |
| Exp1_to_Exp2 | 5.0000 | 0.9738 | 0.7992 | 0.9443 | 0.1391 | 0.9155 | 0.0205 |
| Exp2_to_Exp1 | 0.1000 | 1.0000 | 1.0000 | 0.4334 | 0.0641 | 0.8761 | 0.0000 |
| Exp2_to_Exp1 | 0.5000 | 1.0000 | 1.0000 | 0.4334 | 0.0641 | 0.8761 | 0.0000 |
| Exp2_to_Exp1 | 1.0000 | 1.0000 | 1.0000 | 0.4334 | 0.0641 | 0.8761 | 0.0000 |
| Exp2_to_Exp1 | 2.0000 | 1.0000 | 1.0000 | 0.4334 | 0.0641 | 0.8761 | 0.0000 |
| Exp2_to_Exp1 | 5.0000 | 1.0000 | 1.0000 | 0.4334 | 0.0641 | 0.8761 | 0.0000 |

## Boundary Guard

| direction_id | model | guard_window_count | raw_TES_events_in_guard | clean_TES_events_in_guard | raw_freeze_triggers_in_guard | clean_freeze_triggers_in_guard |
| --- | --- | --- | --- | --- | --- | --- |
| Exp1_to_Exp2 | R0 | 615 | 0 | 0 | 0 | 0 |
| Exp1_to_Exp2 | R4 | 615 | 0 | 0 | 398 | 0 |
| Exp1_to_Exp2 | R5 | 615 | 0 | 0 | 398 | 0 |
| Exp1_to_Exp2 | V1_B4_REF | 2817 | 0 | 0 | 0 | 0 |
| Exp2_to_Exp1 | R0 | 1999 | 0 | 0 | 0 | 0 |
| Exp2_to_Exp1 | R4 | 1999 | 628 | 0 | 710 | 0 |
| Exp2_to_Exp1 | R5 | 1999 | 628 | 0 | 710 | 0 |
| Exp2_to_Exp1 | V1_B4_REF | 9115 | 0 | 0 | 0 | 0 |

## Acceptance

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
