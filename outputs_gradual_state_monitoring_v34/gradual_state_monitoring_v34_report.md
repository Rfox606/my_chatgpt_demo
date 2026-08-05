# V3.4 source weak supervision and target-label-free online adaptation

This report keeps four phases separate:

1. **Source weak-label supervision:** Gaussian soft labels around source-only Stage boundaries trained the shared Adapter + causal TCN.
2. **Target label-free online adaptation:** DirectTransfer, CalibrationOnly, and OnlineAdaptation consumed only declared target force inputs. The first 128 windows were calibration-only.
3. **Target labels for offline evaluation:** target Stage mapping was opened only after all online transition-score and candidate-event files had frozen.
4. **TargetSupervisedUpperBound:** this is a deliberately offline labelled reference, not an online-monitoring result.

The TCN receptive field is 255 windows (required: at least 128). Fixed seeds: [3401, 3402, 3403, 3404, 3405].

## Aggregated comparison

| direction | setting | metric | value_mean | value_std | seed_count |
| --- | --- | --- | --- | --- | --- |
| Exp1_to_Exp2 | CalibrationOnly | Top-8 event recall | 0.65 | 0.13693063937629152 | 5 |
| Exp1_to_Exp2 | CalibrationOnly | boundary_score_percentile | 0.9281331349944217 | 0.03589507487287622 | 5 |
| Exp1_to_Exp2 | CalibrationOnly | proximity_weighted_score | 2.294516412048968 | 0.6865159133427523 | 5 |
| Exp1_to_Exp2 | DirectTransfer | Top-8 event recall | 0.0 | 0.0 | 5 |
| Exp1_to_Exp2 | DirectTransfer | boundary_score_percentile | 0.8213276310896245 | 0.047128843756239325 | 5 |
| Exp1_to_Exp2 | DirectTransfer | proximity_weighted_score | 0.0 | 0.0 | 5 |
| Exp1_to_Exp2 | OnlineAdaptation | Top-8 event recall | 0.3 | 0.27386127875258304 | 5 |
| Exp1_to_Exp2 | OnlineAdaptation | boundary_score_percentile | 0.8745072517664558 | 0.045587852474491916 | 5 |
| Exp1_to_Exp2 | OnlineAdaptation | proximity_weighted_score | 1.2301436837265327 | 0.9166469570826132 | 5 |
| Exp1_to_Exp2 | TargetSupervisedUpperBound | Top-8 event recall | 0.7 | 0.1118033988749895 | 5 |
| Exp1_to_Exp2 | TargetSupervisedUpperBound | boundary_score_percentile | 0.9179806619561175 | 0.029329216579777113 | 5 |
| Exp1_to_Exp2 | TargetSupervisedUpperBound | proximity_weighted_score | 1.652041720314673 | 0.6021131092295725 | 5 |
| Exp2_to_Exp1 | CalibrationOnly | Top-8 event recall | 0.7 | 0.11180339887498948 | 5 |
| Exp2_to_Exp1 | CalibrationOnly | boundary_score_percentile | 0.9720763324802493 | 0.0057217288910523705 | 5 |
| Exp2_to_Exp1 | CalibrationOnly | proximity_weighted_score | 18.851071047008453 | 10.985512286281455 | 5 |
| Exp2_to_Exp1 | DirectTransfer | Top-8 event recall | 0.25 | 0.0 | 5 |
| Exp2_to_Exp1 | DirectTransfer | boundary_score_percentile | 0.9801212863024368 | 0.005390679823057149 | 5 |
| Exp2_to_Exp1 | DirectTransfer | proximity_weighted_score | 30641.967146678064 | 12295.530831242751 | 5 |
| Exp2_to_Exp1 | OnlineAdaptation | Top-8 event recall | 1.0 | 0.0 | 5 |
| Exp2_to_Exp1 | OnlineAdaptation | boundary_score_percentile | 0.9732279959942138 | 0.009925333259737842 | 5 |
| Exp2_to_Exp1 | OnlineAdaptation | proximity_weighted_score | 12.047004849562208 | 2.490285023541667 | 5 |
| Exp2_to_Exp1 | TargetSupervisedUpperBound | Top-8 event recall | 0.7 | 0.11180339887498948 | 5 |
| Exp2_to_Exp1 | TargetSupervisedUpperBound | boundary_score_percentile | 0.9718760431734728 | 0.009688460485547118 | 5 |
| Exp2_to_Exp1 | TargetSupervisedUpperBound | proximity_weighted_score | 14.293450914842708 | 4.2982441968908685 | 5 |

## Adaptation behaviour

| direction | setting | adapter_update_fraction | adapter_frozen_fraction | candidate_event_count | stage_internal_event_count |
| --- | --- | --- | --- | --- | --- |
| Exp1_to_Exp2 | DirectTransfer | 0.0 | 1.0 | 1 | 1 |
| Exp1_to_Exp2 | CalibrationOnly | 0.0 | 0.7973224246931945 | 15 | 10 |
| Exp1_to_Exp2 | OnlineAdaptation | 0.5894384529564894 | 0.4105615470435106 | 24 | 16 |
| Exp1_to_Exp2 | TargetSupervisedUpperBound | 0.0 | 0.5894384529564894 | 18 | 13 |
| Exp1_to_Exp2 | DirectTransfer | 0.0 | 1.0 | 1 | 1 |
| Exp1_to_Exp2 | CalibrationOnly | 0.0 | 0.6162142060245445 | 19 | 13 |
| Exp1_to_Exp2 | OnlineAdaptation | 0.6987727779843809 | 0.3012272220156192 | 14 | 10 |
| Exp1_to_Exp2 | TargetSupervisedUpperBound | 0.0 | 0.577538118259576 | 18 | 11 |
| Exp1_to_Exp2 | DirectTransfer | 0.0 | 1.0 | 1 | 1 |
| Exp1_to_Exp2 | CalibrationOnly | 0.0 | 0.8594272963927111 | 10 | 7 |
| Exp1_to_Exp2 | OnlineAdaptation | 0.4131647452584604 | 0.5868352547415396 | 21 | 13 |
| Exp1_to_Exp2 | TargetSupervisedUpperBound | 0.0 | 0.8077352175529937 | 13 | 9 |
| Exp1_to_Exp2 | DirectTransfer | 0.0 | 1.0 | 1 | 1 |
| Exp1_to_Exp2 | CalibrationOnly | 0.0 | 0.5228709557456304 | 19 | 8 |
| Exp1_to_Exp2 | OnlineAdaptation | 0.7337300111565638 | 0.2662699888434362 | 14 | 11 |
| Exp1_to_Exp2 | TargetSupervisedUpperBound | 0.0 | 0.5102268501301599 | 13 | 5 |
| Exp1_to_Exp2 | DirectTransfer | 0.0 | 1.0 | 1 | 1 |
| Exp1_to_Exp2 | CalibrationOnly | 0.0 | 0.5206396429899591 | 20 | 11 |
| Exp1_to_Exp2 | OnlineAdaptation | 0.7761249535143175 | 0.2238750464856824 | 16 | 10 |
| Exp1_to_Exp2 | TargetSupervisedUpperBound | 0.0 | 0.523986612123466 | 19 | 9 |
| Exp2_to_Exp1 | DirectTransfer | 0.0 | 1.0 | 1 | 0 |
| Exp2_to_Exp1 | CalibrationOnly | 0.0 | 0.7108044953822188 | 34 | 30 |
| Exp2_to_Exp1 | OnlineAdaptation | 0.34694558807165904 | 0.6530544119283409 | 54 | 48 |
| Exp2_to_Exp1 | TargetSupervisedUpperBound | 0.0 | 0.6549460331590075 | 36 | 29 |
| Exp2_to_Exp1 | DirectTransfer | 0.0 | 1.0 | 1 | 0 |
| Exp2_to_Exp1 | CalibrationOnly | 0.0 | 0.6867697785690442 | 36 | 30 |
| Exp2_to_Exp1 | OnlineAdaptation | 0.40024479804161567 | 0.5997552019583844 | 56 | 49 |
| Exp2_to_Exp1 | TargetSupervisedUpperBound | 0.0 | 0.675197507510849 | 29 | 23 |
| Exp2_to_Exp1 | DirectTransfer | 0.0 | 1.0 | 1 | 0 |
| Exp2_to_Exp1 | CalibrationOnly | 0.0 | 0.7181484366306887 | 24 | 19 |
| Exp2_to_Exp1 | OnlineAdaptation | 0.3382663847780127 | 0.6617336152219874 | 53 | 46 |
| Exp2_to_Exp1 | TargetSupervisedUpperBound | 0.0 | 0.6915544675642595 | 30 | 24 |
| Exp2_to_Exp1 | DirectTransfer | 0.0 | 1.0 | 1 | 0 |
| Exp2_to_Exp1 | CalibrationOnly | 0.0 | 0.711694670079003 | 43 | 37 |
| Exp2_to_Exp1 | OnlineAdaptation | 0.4329587181484366 | 0.5670412818515633 | 38 | 33 |
| Exp2_to_Exp1 | TargetSupervisedUpperBound | 0.0 | 0.7414042505841771 | 38 | 33 |
| Exp2_to_Exp1 | DirectTransfer | 0.0 | 1.0 | 1 | 0 |
| Exp2_to_Exp1 | CalibrationOnly | 0.0 | 0.8960721041504395 | 10 | 6 |
| Exp2_to_Exp1 | OnlineAdaptation | 0.4325136308000445 | 0.5674863691999555 | 32 | 27 |
| Exp2_to_Exp1 | TargetSupervisedUpperBound | 0.0 | 0.7256036497162568 | 42 | 34 |

## Required interpretation questions

The comparison table is intentionally descriptive: it does not issue a global PASS/FAIL. Compare OnlineAdaptation with DirectTransfer and CalibrationOnly for Top-8 recall, proximity-weighted score, and boundary-score percentile; inspect candidate counts and frozen fractions before claiming a gain. Extra events are retained as `unmatched_candidate_event`, and their post-hoc Stage locations are reported rather than automatically called false alarms.

## Label provenance

```json
{'source_stage': 'weak supervision only', 'target_online': 'no labels', 'target_evaluation': 'post hoc', 'upper_bound': 'offline only'}
```
