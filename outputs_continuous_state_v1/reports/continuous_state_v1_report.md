# Continuous State Monitoring v1

Implementation acceptance: **PASS**. Scientific acceptance: **FAIL**.

## Required answers

1. Source validation pair AUC and selected C:

```text
direction_id  source_pair_auc  selected_C
Exp1_to_Exp2           0.9031      0.5000
Exp2_to_Exp1           0.8822      5.0000
```

2. The final ten feature weights are listed below. They are normalised L1 rank weights, not calibrated risks.

```text
direction_id     feature_name  normalized_weight  rank
Exp1_to_Exp2           rs_rms           -0.27371     1
Exp1_to_Exp2           rx_q05           -0.22370     2
Exp1_to_Exp2           rs_q05            0.16397     3
Exp1_to_Exp2          rx_mean            0.10528     4
Exp1_to_Exp2       rx_absmean            0.10281     5
Exp1_to_Exp2          rs_mean            0.05449     6
Exp1_to_Exp2       rs_absmean            0.05449     7
Exp1_to_Exp2           ry_p2p            0.01524     8
Exp1_to_Exp2 rs_corrdist_base            0.00356     9
Exp1_to_Exp2 rx_corrdist_base            0.00272    10
Exp2_to_Exp1           rs_rms            0.46312     1
Exp2_to_Exp1          rs_mean           -0.24867     2
Exp2_to_Exp1       rs_absmean           -0.24867     3
Exp2_to_Exp1           rs_q05            0.03253     4
Exp2_to_Exp1 rx_corrdist_base            0.00282     5
Exp2_to_Exp1 rs_corrdist_base           -0.00175     6
Exp2_to_Exp1           rx_q05            0.00121     7
Exp2_to_Exp1           ry_p2p            0.00084     8
Exp2_to_Exp1          rx_mean           -0.00020     9
Exp2_to_Exp1       rx_absmean           -0.00020    10
```

3. Features with the same learned direction in both transfer directions: rs_q05, rx_corrdist_base, ry_p2p.
4. Target long-gap pair concordance is:

```text
direction_id  target_long_gap_concordance  spearman_AWR_cycle  spearman_BD_cycle
Exp1_to_Exp2                       0.8287              0.7398             0.6148
Exp2_to_Exp1                       0.3749             -0.7291             0.6623
```

The fixed target criterion is 0.55.
5. Target initial-baseline stability is:

```text
direction_id  baseline_AWR_rel_median  baseline_AWR_rel_IQR  baseline_BD_median  baseline_BD_p95
Exp1_to_Exp2                   0.0000                0.3119              1.9758           3.0349
Exp2_to_Exp1                   0.0000                0.0074              1.3505           3.0885
```

The baseline AWR relative median is an implementation check and is expected to be approximately zero.
6. Mean target out-of-support fractions are greatest for:

```text
feature_name  oos_fraction
      rx_q05        0.7953
  rx_absmean        0.7064
     rx_mean        0.7064
```

High out-of-support fractions reduce confidence in source-prior transfer; they do not establish a high-wear conclusion.
7. The AWR/BD disagreement and increase candidates are the rows listed below; their cycles identify the follow-up inspection locations.

```text
direction_id       candidate_type  center_cycle  AWR_rel       BD  oos_fraction
Exp1_to_Exp2     high_AWR_high_BD    12960.5000   2.1878 106.4130        0.3000
Exp1_to_Exp2 largest_AWR_increase     3050.5000   0.8298  48.1102        0.4000
Exp1_to_Exp2 largest_AWR_increase     1550.5000   0.2950  22.7707        0.2000
Exp1_to_Exp2 largest_AWR_increase    10550.5000   1.3337  61.8004        0.4000
Exp1_to_Exp2 largest_AWR_increase     6895.5000  -0.0003  16.5007        0.0000
Exp1_to_Exp2 largest_AWR_increase     8695.5000   0.6683  26.0440        0.2000
Exp1_to_Exp2 largest_AWR_increase     9305.5000   1.2449  51.8632        0.3000
Exp1_to_Exp2 largest_AWR_increase    11710.5000   1.6129 104.7228        0.3000
Exp1_to_Exp2 largest_AWR_increase     8095.5000   0.1637  18.8400        0.2000
Exp1_to_Exp2 largest_AWR_increase     9895.5000   1.2653  63.5268        0.3000
Exp1_to_Exp2 largest_AWR_increase    12605.5000   2.0482 127.1382        0.3000
Exp1_to_Exp2 largest_AWR_increase     5695.5000   0.6580  39.6194        0.4000
Exp1_to_Exp2 largest_AWR_increase     3595.5000   0.8223  46.3730        0.4000
Exp1_to_Exp2 largest_AWR_increase      895.5000  -0.2998  25.4965        0.2000
Exp1_to_Exp2 largest_AWR_increase     5095.5000   0.4496  46.2168        0.4000
Exp1_to_Exp2 largest_AWR_increase     2095.5000  -0.2958  34.4355        0.1000
Exp1_to_Exp2 largest_AWR_increase     4195.5000   0.5417  38.0313        0.4000
Exp1_to_Exp2 largest_AWR_increase    13905.5000   1.6219  67.5918        0.3000
Exp1_to_Exp2 largest_AWR_increase    13220.5000   1.7040  63.8813        0.3000
Exp1_to_Exp2 largest_AWR_increase      290.5000   0.0368   1.7933        0.2000
Exp1_to_Exp2 largest_AWR_increase    11095.5000   0.8184  41.0855        0.3000
Exp1_to_Exp2  largest_BD_increase     2700.5000  -0.2432  54.0970        0.0000
Exp1_to_Exp2  largest_BD_increase    11710.5000   1.6129 104.7228        0.3000
Exp1_to_Exp2  largest_BD_increase     8710.5000   1.3644  63.3710        0.4000
Exp1_to_Exp2  largest_BD_increase    10210.5000   1.5998  73.9529        0.4000
Exp1_to_Exp2  largest_BD_increase     6910.5000   0.4754  41.4081        0.2000
Exp1_to_Exp2  largest_BD_increase     9310.5000   1.5147  60.2683        0.3000
Exp1_to_Exp2  largest_BD_increase      610.5000  -0.1052  26.6237        0.3000
Exp1_to_Exp2  largest_BD_increase     1795.5000   0.0298  25.2025        0.2000
Exp1_to_Exp2  largest_BD_increase     5710.5000   1.1563  54.7560        0.4000
```

8. The highest-priority physical checks are candidates marked `high_AWR_high_BD`, `high_AWR_low_BD`, `low_AWR_high_BD`, and the two increase types. These are diagnostic-only offline selections, not online alarms.
9. Source directional evidence below the fixed criterion: []; target long-gap time consistency below the fixed criterion: ['Exp2_to_Exp1'].
10. If this result is not scientifically accepted, the fixed diagnostics distinguish source-direction instability, cross-domain reversal in target concordance, and support loss. The current outcome is: Source directional evidence below the fixed criterion: []; target long-gap time consistency below the fixed criterion: ['Exp2_to_Exp1'].
11. AWR is a continuous, source-learned temporal ranking score. It is **not** a Stage5 probability, a wear percentage, or a failure probability. BD is only distance from that experiment's initial feature-state baseline.
12. No Stage1–Stage5 labels were used for training, C selection, scoring, candidate selection, or output generation in this experiment.

## Interpretation boundary

The two primary outputs are AWR_rel and BD. They are intentionally complementary: a high AWR can coexist with low initial-state distance, and a high BD can occur without a high learned temporal-direction score. Any causal or physical wear conclusion requires the planned surface-morphology, debris, or experimental-observation correspondence.
