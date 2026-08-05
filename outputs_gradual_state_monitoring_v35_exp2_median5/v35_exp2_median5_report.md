# V3.5 Exp2 Median5 diagnostic report

## 1. Did Median5 reduce local fluctuation while retaining long-term trends?

Yes. Across the six features, the mean reduction in median absolute adjacent-window change was 48.1% and the mean reduction in mean absolute adjacent-window change was 16.7%. The minimum Raw-versus-Median5 Spearman correlation was 0.9999, so the long-term ordering/trend was retained.

| Feature | Median step reduction (%) | Mean step reduction (%) | Spearman correlation |
|---|---:|---:|---:|
| rx_mean | 31.9 | 19.2 | 0.9999 |
| rx_q05 | 30.6 | 19.1 | 0.9999 |
| ry_mean | 56.7 | 8.0 | 1.0000 |
| ry_q05 | 65.3 | 14.1 | 0.9999 |
| ry_p2p | 64.4 | 25.8 | 0.9999 |
| rs_rms | 39.6 | 13.9 | 1.0000 |

## 2. Did Exp1 -> Exp2 Top-8, +/-250, and Event F1 improve?

| Setting | Raw Top-8 | Median5 Top-8 | Raw +/-250 | Median5 +/-250 | Raw Event F1 | Median5 Event F1 |
|---|---:|---:|---:|---:|---:|---:|
| CalibrationOnly | 0.35 | 0.35 | 0.65 | 0.65 | 0.315 | 0.323 |
| OnlineAdaptation | 0.15 | 0.10 | 0.15 | 0.10 | 0.157 | 0.100 |
No overall Exp1 -> Exp2 recognition improvement is supported. Median5 changed Top-8 recall by +0.00 for CalibrationOnly and -0.05 for OnlineAdaptation: CalibrationOnly retained Top-8 and +/-250 with only a small F1 increase, whereas OnlineAdaptation declined on all three metrics.

## 3. Were the later 2->3, 3->4, and 4->5 boundaries recovered?

| Boundary | Setting | Raw Top-8 | Median5 Top-8 | Raw +/-250 | Median5 +/-250 | Raw mean rank | Median5 mean rank |
|---|---|---:|---:|---:|---:|---:|---:|
| 2->3 | CalibrationOnly | 0.00 | 0.00 | 0.00 | 0.00 | nan | nan |
| 2->3 | OnlineAdaptation | 0.00 | 0.00 | 0.00 | 0.00 | nan | nan |
| 3->4 | CalibrationOnly | 0.40 | 0.40 | 1.00 | 1.00 | 8.00 | 8.20 |
| 3->4 | OnlineAdaptation | 0.00 | 0.00 | 0.00 | 0.00 | nan | nan |
| 4->5 | CalibrationOnly | 1.00 | 1.00 | 1.00 | 1.00 | 3.80 | 3.80 |
| 4->5 | OnlineAdaptation | 0.00 | 0.00 | 0.00 | 0.00 | nan | nan |
No later boundary showed an improvement in either Top-8 or +/-250 hit rate under Median5.

## 4. Did the relative relationship between CalibrationOnly and OnlineAdaptation change?

| Exp2 condition | Mode | Top-8 | +/-250 | Event F1 | Candidate events |
|---|---|---:|---:|---:|---:|
| Raw | CalibrationOnly | 0.35 | 0.65 | 0.315 | 12.4 |
| Raw | OnlineAdaptation | 0.15 | 0.15 | 0.157 | 3.0 |
| Median5 | CalibrationOnly | 0.35 | 0.65 | 0.323 | 12.0 |
| Median5 | OnlineAdaptation | 0.10 | 0.10 | 0.100 | 2.6 |
The relative ordering did not reverse: CalibrationOnly remained stronger than OnlineAdaptation for Top-8, +/-250, and Event F1 in both Raw and Median5. Under Median5 the gap widened slightly because OnlineAdaptation declined while CalibrationOnly retained its boundary hit rates.

## 5. Did Exp2-Median5 -> Exp1 degrade materially?

| Setting | Raw Top-8 | Median5 Top-8 | Raw +/-250 | Median5 +/-250 | Raw Event F1 | Median5 Event F1 |
|---|---:|---:|---:|---:|---:|---:|
| CalibrationOnly | 0.90 | 0.90 | 0.70 | 0.70 | 0.251 | 0.251 |
| OnlineAdaptation | 0.55 | 0.55 | 0.55 | 0.55 | 0.611 | 0.611 |
No material degradation by the predeclared 0.10 change screen was observed.

## 6. Is any improvement true boundary recovery, rather than only fewer candidates?

For Exp1 -> Exp2 OnlineAdaptation, candidate events changed from 3.0 (Raw) to 2.6 (Median5), while Event F1 changed from 0.157 to 0.100.
The result is not claimed as true boundary recovery unless the boundary table shows a hit improvement together with an Event F1 improvement; candidate-count reduction alone is not treated as an improvement.
