from __future__ import annotations

import pandas as pd


def _text(frame: pd.DataFrame, columns: list[str] | None = None, limit: int = 20) -> str:
    if frame.empty:
        return "No available records."
    shown = frame.loc[:, columns] if columns is not None else frame
    return "\n".join(line.rstrip() for line in shown.head(limit).to_string(index=False).splitlines())


def make_report(physical: pd.DataFrame, severe_updates: pd.DataFrame, states: pd.DataFrame, ablation: pd.DataFrame, metrics: pd.DataFrame, benefit: pd.DataFrame, diagnostics: dict[str, dict[str, object]]) -> str:
    exp1 = physical.loc[physical.target_dataset.eq("Exp1")]
    exp2 = physical.loc[physical.target_dataset.eq("Exp2")]
    exp1_row = exp1.iloc[0] if not exp1.empty else pd.Series(dtype=object)
    exp2_row = exp2.iloc[0] if not exp2.empty else pd.Series(dtype=object)
    freeze = metrics.loc[metrics.horizon_cycles.isin([500, 1000])]
    report = f"""# Continuous State Monitoring v3 — Causal Plateau-to-Severe Online Adaptation

## Required answers

1. Exp1 stable-wear plateau detected: {bool(not exp1.empty and pd.notna(exp1_row.get('detected_plateau_cycle')))}.
2. Exp1 plateau lock cycle: {exp1_row.get('detected_plateau_cycle', 'NOT_FOUND')}.
3. Exp1 persistent severe false alarms (>500 cycles): {exp1_row.get('persistent_severe_alarm_count', 'N/A')}.
4. Exp2 plateau/low-speed region before instability: plateau={exp2_row.get('detected_plateau_cycle', 'NOT_FOUND')}, exit={exp2_row.get('detected_plateau_exit_cycle', 'NOT_FOUND')}.
5. Exp2 plateau exit cycle: {exp2_row.get('detected_plateau_exit_cycle', 'NOT_FOUND')}.
6. Exp2 persistent severe-candidate onset: {exp2_row.get('detected_severe_onset_cycle', 'NOT_FOUND')}.
7. Exp2 lead/lag to the post-hoc known severe boundary: {exp2_row.get('lead_lag_to_known_severe_boundary', 'N/A')} cycles. This label was read only after online outputs were saved.
8. Protocol A leakage check: {diagnostics['target_future_leakage_check']['status']}; Exp2 future and terminal data were excluded from source priors.
9. Target severe-direction updates (causal, post-exit only):

```text
{_text(severe_updates, ['protocol_id','target_dataset','feature_name','weight','cycle','severe_direction_cosine_previous'], 30)}
```

10. D/V/A/S differences are retained as continuous target-relative state signals, not stages:

```text
{_text(states.groupby(['protocol_id','target_dataset'], as_index=False)[['D_state','V50_norm','A_state','S_severe_candidate']].mean())}
```

11. M0–M5 ablation evidence:

```text
{_text(ablation)}
```

12–14. Frozen, robust RLS, and Safe Ensemble 500/1000-cycle errors:

```text
{_text(freeze, ['protocol_id','output_name','horizon_cycles','model','MAE','RMSE','direction_accuracy'], 80)}
```

15. SAFE_ONLINE_FORECAST_BENEFIT:

```text
{_text(benefit)}
```

16. These internal state-trend forecasts do not establish more accurate prediction of true physical wear. Independent morphology, debris, and mass-loss evidence remains required.
17. Plateau lock/exit and severe-candidate conclusions require morphology, wear-debris, and mass-loss validation before physical interpretation.
18. Stable wear is a relative low-speed platform, not a final state. Severe wear is a possible later instability/progression after that platform; no Stage1–Stage5 classifier is used.

## Acceptance

```text
implementation={diagnostics['implementation_acceptance']['status']}
prefix_causality={diagnostics['prefix_causality_check']['status']}
safe_ensemble_rollback={diagnostics['safe_ensemble_rollback_check']['status']}
```
"""
    return "\n".join(line.rstrip() for line in report.splitlines()) + "\n"
