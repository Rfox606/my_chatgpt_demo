# V3.5.1 event-level residual gate report

## Main results

| direction | setting | events | Top-8 | ±250 | F1 | update fraction |
|---|---:|---:|---:|---:|---:|---:|
| Exp1_to_Exp2 | V35_HardAND_Current | 3.0 | 0.15 | 0.15 | 0.157 | 0.797 |
| Exp1_to_Exp2 | V351_EventGate_CurrentAdapt | 3.4 | 0.20 | 0.00 | 0.000 | 0.782 |
| Exp1_to_Exp2 | V351_EventGate_WarningFreeze | 3.4 | 0.15 | 0.10 | 0.089 | 0.655 |
| Exp1_to_Exp2 | V351_EventGate_WarningFreeze_Update5 | 4.2 | 0.40 | 0.10 | 0.080 | 0.104 |
| Exp1_to_Exp2 | V351_ResidualRerank_WarningFreeze_Update5 | 7.8 | 0.50 | 0.15 | 0.090 | 0.104 |
| Exp2_to_Exp1 | V35_HardAND_Current | 3.0 | 0.55 | 0.55 | 0.611 | 0.932 |
| Exp2_to_Exp1 | V351_EventGate_CurrentAdapt | 3.2 | 0.50 | 0.30 | 0.324 | 0.924 |
| Exp2_to_Exp1 | V351_EventGate_WarningFreeze | 3.4 | 0.55 | 0.35 | 0.371 | 0.883 |
| Exp2_to_Exp1 | V351_EventGate_WarningFreeze_Update5 | 1.0 | 0.15 | 0.00 | 0.000 | 0.189 |
| Exp2_to_Exp1 | V351_ResidualRerank_WarningFreeze_Update5 | 1.0 | 0.15 | 0.00 | 0.000 | 0.189 |

## Required conclusions

- **Exp1_to_Exp2, EventGate versus hard-AND:** Top-8 0.20 versus 0.15; EventGate is better on this direction, so it is not an across-direction replacement unless both directions improve.
- **Exp1_to_Exp2, warning freeze and Update5:** WarningFreeze Top-8=0.15; Update5 Top-8=0.40, with update fraction 0.104 versus 0.655 at interval 1.
- **Exp2_to_Exp1, EventGate versus hard-AND:** Top-8 0.50 versus 0.55; EventGate is worse on this direction, so it is not an across-direction replacement unless both directions improve.
- **Exp2_to_Exp1, warning freeze and Update5:** WarningFreeze Top-8=0.55; Update5 Top-8=0.15, with update fraction 0.189 versus 0.883 at interval 1.
- **Does warning freeze restore Exp1-to-Exp2 mid/late boundaries?** No overall: Update5 raises the direction-level Top-8 to 0.40, but the failed-stage table still contains the late Exp1-to-Exp2 boundaries. Remaining failed boundaries (WarningFreeze Update5): Exp1_to_Exp2 1→2 Top-8=0.80; Exp1_to_Exp2 2→3 Top-8=0.20; Exp1_to_Exp2 3→4 Top-8=0.40; Exp1_to_Exp2 4→5 Top-8=0.20; Exp2_to_Exp1 1→2 Top-8=0.60; Exp2_to_Exp1 2→3 Top-8=0.00; Exp2_to_Exp1 3→4 Top-8=0.00; Exp2_to_Exp1 4→5 Top-8=0.00.
- **Can Exp2-to-Exp1 improve Top-8 with 3–6 events?** No: Update5 has 1.0 events and Top-8=0.15; it misses the 3–6-event target and loses the recall retained by interval-1 settings.
- **Residual rerank versus event gate:** summed two-direction Top-8 is 0.65 versus 0.55; it retains more true boundaries at the declared cut-off, while changing Exp1-to-Exp2 event count from 4.2 to 7.8.
- **TCN+attention decision:** Do not proceed yet. First isolate why Update5 prevents Exp2-to-Exp1 events and why EventGate loses calibrated boundary hits; attention capacity would otherwise confound the unresolved gating/adaptation failure.

Protocol: no target Stage, stopping position, total length, fixed cycle location, or future target data was read by online scoring/adaptation. Stage appears only after frozen online outputs for offline evaluation.
