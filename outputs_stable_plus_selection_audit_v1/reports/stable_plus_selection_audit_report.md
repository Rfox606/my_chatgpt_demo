# stable_plus Selection Audit Report

## Purpose

This run documents the evidence behind the current stable_plus feature set. It does not retrain AWR, rebuild Stage1-Stage5 classification, or change the research target.

## Main Answer

stable_plus is not an arbitrary hand-picked list. It is a compact subset of the candidate shear-ratio feature pool, audited against direction consistency, cross-experiment target-side ranking, effect size, saturation risk, redundancy, and physical interpretability.

## Candidate Pool

- Candidate features audited: 28.
- stable_plus features: 10.
- non stable_plus reference features: 18.
- Source pool: `outputs_weighted_awrcore_v1/results/window_feature_z_table.csv`.

## Selection Evidence

- Direction consistency checks whether Exp1 and Exp2 show the same early-to-late feature direction.
- Target AUROC/AUPRC are computed by applying the source-dataset direction sign to the opposite dataset and using Stage5 only as a late-state proxy label.
- Spearman_min summarizes the weakest signed monotonic relation between stage and feature across Exp1/Exp2.
- Saturation warnings combine available normalization diagnostics with observed clipped z-values.
- Redundancy notes are read from the fair ablation feature weight table where available.

## stable_plus Feature Audit

| feature_name | channel | feature_family | direction_consistent | worst_direction_AUROC | worst_direction_AUPRC | Spearman_min | saturation_warning | keep_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rs_absmean | rs | absmean | True | 0.8014 | 0.3802 | 0.592 | no: max_saturation=0.000, frac_abs_ge_11.9=0.000 | retained in stable_plus because it comes from the candidate shear-ratio pool; direction consistency=True; worst AUROC=0.801; Spearman_min=0.592; no major clipping flag under the current diagnostics. |
| rs_mean | rs | mean | True | 0.8014 | 0.3802 | 0.592 | no: max_saturation=0.000, frac_abs_ge_11.9=0.000 | retained in stable_plus because it comes from the candidate shear-ratio pool; direction consistency=True; worst AUROC=0.801; Spearman_min=0.592; no major clipping flag under the current diagnostics. |
| rs_q05 | rs | q05 | True | 0.7951 | 0.3072 | 0.5847 | no: max_saturation=0.000, frac_abs_ge_11.9=0.000 | retained in stable_plus because it comes from the candidate shear-ratio pool; direction consistency=True; worst AUROC=0.795; Spearman_min=0.585; no major clipping flag under the current diagnostics. |
| ry_p2p | ry | p2p | True | 0.7837 | 0.4828 | 0.7392 | yes: max_saturation=0.057, frac_abs_ge_11.9=0.048 | retained in stable_plus because it comes from the candidate shear-ratio pool; direction consistency=True; worst AUROC=0.784; Spearman_min=0.739; kept with a physical-validation flag because clipping/saturation is visible. |
| rs_rms | rs | rms | True | 0.7698 | 0.404 | 0.5101 | no: max_saturation=0.000, frac_abs_ge_11.9=0.000 | retained in stable_plus because it comes from the candidate shear-ratio pool; direction consistency=True; worst AUROC=0.770; Spearman_min=0.510; no major clipping flag under the current diagnostics. |
| rs_corrdist_base | rs | corrdist_base | True | 0.7063 | 0.2009 | 0.7352 | yes: max_saturation=0.687, frac_abs_ge_11.9=0.530 | retained in stable_plus because it comes from the candidate shear-ratio pool; direction consistency=True; worst AUROC=0.706; Spearman_min=0.735; kept with a physical-validation flag because clipping/saturation is visible. |
| rx_corrdist_base | rx | corrdist_base | True | 0.7019 | 0.1986 | 0.7586 | yes: max_saturation=0.694, frac_abs_ge_11.9=0.536 | retained in stable_plus because it comes from the candidate shear-ratio pool; direction consistency=True; worst AUROC=0.702; Spearman_min=0.759; kept with a physical-validation flag because clipping/saturation is visible. |
| rx_q05 | rx | q05 | True | 0.7014 | 0.3547 | 0.4301 | no: max_saturation=0.000, frac_abs_ge_11.9=0.000 | retained in stable_plus because it comes from the candidate shear-ratio pool; direction consistency=True; worst AUROC=0.701; Spearman_min=0.430; no major clipping flag under the current diagnostics. |
| rx_mean | rx | mean | True | 0.7 | 0.3226 | 0.4792 | no: max_saturation=0.034, frac_abs_ge_11.9=0.008 | retained in stable_plus because it comes from the candidate shear-ratio pool; direction consistency=True; worst AUROC=0.700; Spearman_min=0.479; no major clipping flag under the current diagnostics. |
| rx_absmean | rx | absmean | True | 0.7 | 0.3225 | 0.4792 | no: max_saturation=0.034, frac_abs_ge_11.9=0.008 | retained in stable_plus because it comes from the candidate shear-ratio pool; direction consistency=True; worst AUROC=0.700; Spearman_min=0.479; no major clipping flag under the current diagnostics. |

## Non stable_plus Reference Features

| feature_name | channel | feature_family | direction_consistent | worst_direction_AUROC | worst_direction_AUPRC | Spearman_min | saturation_warning | keep_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rx_peak_phase | rx | peak_phase | True | 0.6544 | 0.24 | 0.2362 | yes: max_saturation=0.131, frac_abs_ge_11.9=0.105 | not retained in stable_plus: weaker cross-experiment target ranking; clip/saturation risk. |
| ry_rms | ry | rms | True | 0.6538 | 0.2094 | 0.3614 | yes: max_saturation=0.446, frac_abs_ge_11.9=0.353 | not retained in stable_plus: weaker cross-experiment target ranking; clip/saturation risk. |
| ry_absmean | ry | absmean | True | 0.6435 | 0.2044 | 0.3487 | yes: max_saturation=0.093, frac_abs_ge_11.9=0.071 | not retained in stable_plus: weaker cross-experiment target ranking; clip/saturation risk. |
| ry_mean | ry | mean | True | 0.6435 | 0.2044 | 0.3487 | yes: max_saturation=0.092, frac_abs_ge_11.9=0.071 | not retained in stable_plus: weaker cross-experiment target ranking; clip/saturation risk. |
| rx_rms | rx | rms | True | 0.6254 | 0.2769 | 0.3156 | no: max_saturation=0.043, frac_abs_ge_11.9=0.011 | not retained in stable_plus: weaker cross-experiment target ranking. |
| ry_q05 | ry | q05 | True | 0.6095 | 0.1611 | 0.5014 | yes: max_saturation=0.886, frac_abs_ge_11.9=0.679 | not retained in stable_plus: weaker cross-experiment target ranking; clip/saturation risk. |
| ry_std | ry | std | True | 0.606 | 0.5012 | 0.5295 | no: max_saturation=0.000, frac_abs_ge_11.9=0.000 | not retained in stable_plus: weaker cross-experiment target ranking. |
| ry_peak_phase | ry | peak_phase | True | 0.5476 | 0.4973 | -0.0152 | yes: max_saturation=0.432, frac_abs_ge_11.9=0.361 | not retained in stable_plus: weaker cross-experiment target ranking; clip/saturation risk. |
| ry_corrdist_base | ry | corrdist_base | True | 0.4304 | 0.1697 | 0.1528 | yes: max_saturation=0.585, frac_abs_ge_11.9=0.520 | not retained in stable_plus: weaker cross-experiment target ranking; clip/saturation risk. |
| ry_peak_width | ry | peak_width | False | 0.05684 | 0.09308 | -0.1709 | no: max_saturation=0.000, frac_abs_ge_11.9=0.000 | not retained in stable_plus: direction changes between Exp1 and Exp2; weaker cross-experiment target ranking. |
| rx_std | rx | std | False | 0.0503 | 0.09237 | 0.4906 | yes: max_saturation=0.300, frac_abs_ge_11.9=0.230 | not retained in stable_plus: direction changes between Exp1 and Exp2; weaker cross-experiment target ranking; clip/saturation risk. |
| rs_q95 | rs | q95 | False | 0.04475 | 0.09303 | 0.1243 | no: max_saturation=0.000, frac_abs_ge_11.9=0.000 | not retained in stable_plus: direction changes between Exp1 and Exp2; weaker cross-experiment target ranking. |
| rx_q95 | rx | q95 | False | 0.03454 | 0.09183 | 0.4571 | no: max_saturation=0.001, frac_abs_ge_11.9=0.002 | not retained in stable_plus: direction changes between Exp1 and Exp2; weaker cross-experiment target ranking. |
| rx_p2p | rx | p2p | False | 0.03241 | 0.09151 | 0.4216 | yes: max_saturation=0.238, frac_abs_ge_11.9=0.185 | not retained in stable_plus: direction changes between Exp1 and Exp2; weaker cross-experiment target ranking; clip/saturation risk. |
| rx_peak_width | rx | peak_width | False | 0.03204 | 0.1263 | 0.3413 | no: max_saturation=0.000, frac_abs_ge_11.9=0.000 | not retained in stable_plus: direction changes between Exp1 and Exp2; weaker cross-experiment target ranking. |
| rs_std | rs | std | False | 0.02741 | 0.09148 | 0.6443 | yes: max_saturation=0.181, frac_abs_ge_11.9=0.142 | not retained in stable_plus: direction changes between Exp1 and Exp2; weaker cross-experiment target ranking; clip/saturation risk. |
| ry_q95 | ry | q95 | False | 0.008294 | 0.1319 | -0.1647 | yes: max_saturation=0.092, frac_abs_ge_11.9=0.071 | not retained in stable_plus: direction changes between Exp1 and Exp2; weaker cross-experiment target ranking; clip/saturation risk. |
| rs_p2p | rs | p2p | False | 0.001408 | 0.09068 | 0.5992 | no: max_saturation=0.027, frac_abs_ge_11.9=0.023 | not retained in stable_plus: direction changes between Exp1 and Exp2; weaker cross-experiment target ranking. |

## Fair Ablation Context

M1_stable uses explicit direction correction. In the fair ablation, M1_stable is close to M0_stable, which supports the interpretation that internal direction conflict inside stable_plus is limited. M2_stable does not provide incremental gain over M1_stable, so the current equal-weight direction-corrected structure is the more robust interpretation layer.

| model_name | feature_group | formulation | mean_AUROC | worst_AUROC | mean_AUPRC | worst_AUPRC | mean_Spearman | worst_Spearman | mean_ScoreGap | worst_ScoreGap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0_stable | stable_plus | mean_z | 0.9174 | 0.8443 | 0.781 | 0.619 | 0.8007 | 0.7777 | 5.26 | 5.142 |
| M1_stable | stable_plus | direction_mean_z | 0.9174 | 0.8443 | 0.781 | 0.619 | 0.8007 | 0.7777 | 5.26 | 5.142 |
| M2_stable | stable_plus | weighted_direction | 0.9104 | 0.8428 | 0.7376 | 0.5957 | 0.82 | 0.8052 | 5.595 | 5.445 |

## Physical Validation Implications

- Features with corrdist_base or saturation warnings should be prioritized in physical closed-loop validation because large normalized values may combine real waveform-shape change with clipping pressure.
- Resultant shear features (`rs_*`) are useful as integrated shear-state descriptors, but their high agreement should be checked against rx/ry channel-specific evidence.
- Tail/span features such as `rs_q05`, `rx_q05`, and `ry_p2p` should be checked against sensitive-phase contact migration, local wear morphology, and debris evidence.

## Warnings

- Preferred normalization diagnostics missing; used fallback: outputs_weighted_awrcore_v1\diagnostics\feature_normalization_diagnostics.csv
