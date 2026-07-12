# Adaptive AWR v1 Cross-Domain Report

## Outcome

Overall acceptance status: **FAIL**. `Stage5` is used as a late-state proxy for final evaluation only, not as a target-side training or adaptation input.

## Protocol

- Transfers: `Exp1 -> Exp2` and `Exp2 -> Exp1`.
- Shared stable_plus feature directions and logistic risk heads are fitted only from source training windows.
- Target scoring is causal. The inference interface rejects `stage` and `stage_label` columns.
- Target calibration is limited to the initial `500` cycles. Feature baseline centres/IQRs, directions and risk-head coefficients are never updated online.
- B1 is static; B2 enables gated reliability; B3 adds the bounded logit offset; B4 adds forced freezing, checkpoints and rollback.

## Transfer Results

| direction_id | model | Stage5_AUROC | Stage5_AUPRC | Stage5_Recall | Stage1to2_FPR | Recall_at_10pct_Stage1to2_FPR | Risk_Stage_Spearman | detection_lead_cycles_relative_to_Stage5 | Stage5_risk_suppression_B1_minus_B4 | adaptation_safety_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1_to_Exp2 | B0 | 0.9906 | 0.9430 | 0.9958 | 0.0000 | 1.0000 | 0.8237 | 2365.5000 | 0.0000 | False |
| Exp1_to_Exp2 | B1 | 0.9963 | 0.9865 | 0.3083 | 0.0000 | 1.0000 | 0.5522 | -949.5000 | 0.0000 | False |
| Exp1_to_Exp2 | B2 | 0.9976 | 0.9912 | 0.2250 | 0.0000 | 1.0000 | 0.5362 | -999.5000 | 0.0000 | False |
| Exp1_to_Exp2 | B3 | 0.9970 | 0.9888 | 0.2833 | 0.0000 | 1.0000 | 0.5444 | -964.5000 | 0.0000 | False |
| Exp1_to_Exp2 | B4 | 0.9970 | 0.9888 | 0.2833 | 0.0000 | 1.0000 | 0.5440 | -964.5000 | 0.0000 | False |
| Exp2_to_Exp1 | B0 | 0.8443 | 0.6190 | 0.0000 | 0.0000 | 0.7185 | 0.7777 | 20.5000 | -0.0653 | False |
| Exp2_to_Exp1 | B1 | 0.7963 | 0.6824 | 0.0000 | 0.0000 | 0.6253 | 0.3684 |  | -0.0653 | False |
| Exp2_to_Exp1 | B2 | 0.8512 | 0.7672 | 0.0000 | 0.0000 | 0.7572 | 0.4334 |  | -0.0653 | False |
| Exp2_to_Exp1 | B3 | 0.8647 | 0.7837 | 0.0000 | 0.0000 | 0.7767 | 0.4511 |  | -0.0653 | False |
| Exp2_to_Exp1 | B4 | 0.8647 | 0.7837 | 0.0000 | 0.0000 | 0.7767 | 0.4511 |  | -0.0653 | False |

## Source-only Thresholds

| direction_id | source_dataset | target_dataset | B0_source_validation_AWR_p95_threshold | source_AWR_high_threshold | source_BD_high_threshold | source_TES_threshold | source_RS50_threshold | risk_threshold | source_validation_stage5_recall | source_validation_stage5_precision | source_validation_stage1to2_fpr | threshold_selection_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1_to_Exp2 | Exp1 | Exp2 | 3.4821 | 3.4821 | 8.0273 | 9.2828 | 0.0050 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | min_early_fpr_subject_to_stage5_recall_0.85 |
| Exp2_to_Exp1 | Exp2 | Exp1 | 4.4106 | 4.4106 | 4.2026 | 3.0000 | 0.0050 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | min_early_fpr_subject_to_stage5_recall_0.85 |

## Risk-head Optimisation

| direction_id | source_dataset | target_dataset | optimizer_success | objective | beta0 | beta_AWR_adaptive | beta_BDall_xy_v2 | beta_RS50_positive | beta_TES | beta_high_AWR_high_BD_occupancy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1_to_Exp2 | Exp1 | Exp2 | True | 1150.4204 | -4.6912 | 9.9215 | 10.3891 | 0.0000 | 0.0000 | 0.0000 |
| Exp2_to_Exp1 | Exp2 | Exp1 | True | 10.6324 | -26.3836 | 20.0692 | 7.1822 | 3.0378 | 2.2267 | 0.0093 |

## Acceptance Checks

| criterion | observed | status |
| --- | --- | --- |
| B4 worst AUROC drop from B0 <= 0.02 | 0.0204 | PASS |
| B4 worst AUPRC drop from B0 <= 0.03 | 0.1647 | PASS |
| B4 Stage5 Recall is not below B0 in either direction | -0.7125 | FAIL |
| B4 Stage1-2 FPR increase <= 0.05 in either direction | 0.0000 | PASS |
| Stage5 risk suppression <= 0.10 in either direction | 0.0000 | PASS |

## Interpretation

The adaptive output is a **late-state risk score**, not a calibrated failure probability. A failed acceptance item is retained as a result and has not triggered threshold or model retuning.
