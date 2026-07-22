# stable_plus Feature Rationale

This note explains why the current stable_plus features are treated as a justified signal layer rather than an arbitrary empirical subset.
Each item is drawn from the candidate shear-ratio feature pool and is checked for cross-experiment direction, target-side ranking, effect gap, saturation risk, redundancy, and physical meaning.

## `rs_corrdist_base`

`rs_corrdist_base` belongs to the `rs` channel and `corrdist_base` feature family. It reflects resultant shear ratio contrast; shape departure from the baseline mean waveform.
It is kept because Exp1/Exp2 directions are increase / increase, direction_consistent=True, worst AUROC=0.706, worst AUPRC=0.201, and Spearman_min=0.735.
Physical-loop priority: Yes; verify whether the large normalized response is physical rather than clipping-driven.

## `rs_mean`

`rs_mean` belongs to the `rs` channel and `mean` feature family. It reflects resultant shear ratio contrast; signed average shear ratio in the sensitive phase.
It is kept because Exp1/Exp2 directions are increase / increase, direction_consistent=True, worst AUROC=0.801, worst AUPRC=0.380, and Spearman_min=0.592.
Physical-loop priority: Moderate; use resultant shear behavior as an integrated cross-channel validation clue.

## `rs_absmean`

`rs_absmean` belongs to the `rs` channel and `absmean` feature family. It reflects resultant shear ratio contrast; average shear magnitude in the sensitive phase.
It is kept because Exp1/Exp2 directions are increase / increase, direction_consistent=True, worst AUROC=0.801, worst AUPRC=0.380, and Spearman_min=0.592.
Physical-loop priority: Moderate; use resultant shear behavior as an integrated cross-channel validation clue.

## `rs_q05`

`rs_q05` belongs to the `rs` channel and `q05` feature family. It reflects resultant shear ratio contrast; lower-tail sensitive-phase shear ratio.
It is kept because Exp1/Exp2 directions are increase / increase, direction_consistent=True, worst AUROC=0.795, worst AUPRC=0.307, and Spearman_min=0.585.
Physical-loop priority: Yes; check sensitive-phase tail/span behavior against local contact and debris evidence.

## `rx_corrdist_base`

`rx_corrdist_base` belongs to the `rx` channel and `corrdist_base` feature family. It reflects Fx/Fz main shear ratio; shape departure from the baseline mean waveform.
It is kept because Exp1/Exp2 directions are increase / increase, direction_consistent=True, worst AUROC=0.702, worst AUPRC=0.199, and Spearman_min=0.759.
Physical-loop priority: Yes; verify whether the large normalized response is physical rather than clipping-driven.

## `rs_rms`

`rs_rms` belongs to the `rs` channel and `rms` feature family. It reflects resultant shear ratio contrast; energy-like shear ratio magnitude.
It is kept because Exp1/Exp2 directions are increase / increase, direction_consistent=True, worst AUROC=0.770, worst AUPRC=0.404, and Spearman_min=0.510.
Physical-loop priority: Moderate; use resultant shear behavior as an integrated cross-channel validation clue.

## `ry_p2p`

`ry_p2p` belongs to the `ry` channel and `p2p` feature family. It reflects Fy/Fz lateral shear ratio; peak-to-peak sensitive-phase span.
It is kept because Exp1/Exp2 directions are increase / increase, direction_consistent=True, worst AUROC=0.784, worst AUPRC=0.483, and Spearman_min=0.739.
Physical-loop priority: Yes; verify whether the large normalized response is physical rather than clipping-driven.

## `rx_mean`

`rx_mean` belongs to the `rx` channel and `mean` feature family. It reflects Fx/Fz main shear ratio; signed average shear ratio in the sensitive phase.
It is kept because Exp1/Exp2 directions are increase / increase, direction_consistent=True, worst AUROC=0.700, worst AUPRC=0.323, and Spearman_min=0.479.
Physical-loop priority: Moderate; validate when the candidate window is selected for physical-loop checking.

## `rx_absmean`

`rx_absmean` belongs to the `rx` channel and `absmean` feature family. It reflects Fx/Fz main shear ratio; average shear magnitude in the sensitive phase.
It is kept because Exp1/Exp2 directions are increase / increase, direction_consistent=True, worst AUROC=0.700, worst AUPRC=0.323, and Spearman_min=0.479.
Physical-loop priority: Moderate; validate when the candidate window is selected for physical-loop checking.

## `rx_q05`

`rx_q05` belongs to the `rx` channel and `q05` feature family. It reflects Fx/Fz main shear ratio; lower-tail sensitive-phase shear ratio.
It is kept because Exp1/Exp2 directions are increase / increase, direction_consistent=True, worst AUROC=0.701, worst AUPRC=0.355, and Spearman_min=0.430.
Physical-loop priority: Yes; check sensitive-phase tail/span behavior against local contact and debris evidence.
