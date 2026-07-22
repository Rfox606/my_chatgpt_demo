from __future__ import annotations

import numpy as np
import pandas as pd


def _table(frame: pd.DataFrame, columns: list[str] | None = None, limit: int = 40) -> str:
    if frame.empty:
        return "No available records."
    shown = frame.loc[:, columns] if columns is not None else frame
    return shown.head(limit).to_string(index=False)


def _first(frame: pd.DataFrame, column: str) -> object:
    value = frame[column].iloc[0] if not frame.empty and column in frame else np.nan
    return "NOT_FOUND" if pd.isna(value) else value


def make_report(
    states: pd.DataFrame,
    physical: pd.DataFrame,
    severe_updates: pd.DataFrame,
    metrics: pd.DataFrame,
    segments: pd.DataFrame,
    rolling: pd.DataFrame,
    regret: pd.DataFrame,
    sensitivity: pd.DataFrame,
    episodes: pd.DataFrame,
    diagnostics: dict[str, dict[str, object]],
    benefit: pd.DataFrame,
) -> str:
    exp1 = physical.loc[physical.target_dataset.eq("Exp1")]
    exp2 = physical.loc[physical.target_dataset.eq("Exp2")]
    exp1_states = states.loc[states.dataset.eq("Exp1")]
    exp2_states = states.loc[states.dataset.eq("Exp2")]
    exp1_conditions = {name: int((exp1_states[name] == 0).sum()) for name in ("D_condition", "V50_condition", "V100_condition", "volatility_condition")} if not exp1_states.empty else {}
    exp2_conditions = {name: int((exp2_states[name] == 0).sum()) for name in ("D_condition", "V50_condition", "V100_condition", "volatility_condition")} if not exp2_states.empty else {}
    resets = int(len(episodes))
    frozen_rows = int((states.plateau_locked == 0).sum())  # State availability context, not physical wear.
    report = f"""# Continuous State Monitoring v3.1 report

## Guard-aware plateau repair

The v3 reachability defect is **{diagnostics['plateau_reachability_check']['status']}**: the state machine measures evidence in valid observed cycles and preserves both success and failure counters in a restart guard.  Guard pause audit: **{diagnostics['guard_pause_check']['status']}**.  Plateau-reference freeze audit: **{diagnostics['plateau_reference_freeze_check']['status']}**.

## Required scientific answers

1. Exp1 plateau detected: {not exp1.empty and pd.notna(_first(exp1, 'detected_plateau_cycle'))}; lock cycle: {_first(exp1, 'detected_plateau_cycle')}; reference interval is recorded in `state_window_scores_v31.csv`.
2. Exp1 false plateau-exit confirmation: {not exp1.empty and pd.notna(_first(exp1, 'detected_plateau_exit_cycle'))}; persistent severe-candidate events: {_first(exp1, 'persistent_severe_alarm_count')}.
3. Exp2 plateau/low-speed evidence: {not exp2.empty and pd.notna(_first(exp2, 'detected_plateau_cycle'))}; exit cycle: {_first(exp2, 'detected_plateau_exit_cycle')}.
4. Exp2 target severe direction established: {not severe_updates.loc[severe_updates.target_dataset.eq('Exp2')].empty}; first causal severe-update cycle: {_first(severe_updates.loc[severe_updates.target_dataset.eq('Exp2')], 'cycle')}.
5. Exp2 persistent severe-candidate onset: {_first(exp2, 'detected_severe_onset_cycle')}; post-hoc lead/lag to the known severe boundary: {_first(exp2, 'lead_lag_cycles')} cycles.
6. Plateau-condition failure counts, Exp1: {exp1_conditions}; Exp2: {exp2_conditions}.  These identify D, V50, V100, or volatility conditions rather than hiding a failed plateau search.
7. q50/q60/q75 sensitivity (q75 remains the fixed primary test):

```text
{_table(sensitivity, ['protocol_id','quantile','plateau_lock_detected','plateau_lock_cycle','plateau_exit_detected','plateau_exit_cycle'])}
```

8. Severe-direction feature weights (only causal post-exit updates):

```text
{_table(severe_updates, ['protocol_id','target_dataset','feature_name','weight','cycle','severe_direction_cosine_previous'], 30)}
```

## Forecast comparison

All six required models are compared below.  Unavailable instability or severe-score heads are preserved as unavailable; they are never zero-filled into training.

```text
{_table(metrics.loc[metrics.horizon_cycles.isin([500, 1000])], ['protocol_id','output_name','horizon_cycles','model','prediction_count','MAE','RMSE','direction_accuracy'], 120)}
```

Early/middle/late performance:

```text
{_table(segments.loc[segments.horizon_cycles.isin([500, 1000])], ['protocol_id','output_name','horizon_cycles','segment','model','MAE','RMSE','direction_accuracy'], 120)}
```

The rolling metrics and cumulative-regret tables are saved in their full temporal order.  The final available Safe Ensemble regrets are:

```text
{_table(regret.sort_values('due_observation_cycle').groupby(['protocol_id','output_name','horizon_cycles','baseline_model'], as_index=False).tail(1), None, 80)}
```

Forecast-benefit decision (Protocol A is mandatory; Protocol B is control only):

```text
{_table(benefit)}
```

## Safe Ensemble

Independent reset episodes: {resets}.  The reset audit is **{diagnostics['safe_ensemble_reset_check']['status']}**; a FROZEN episode does not reset repeatedly or extend `freeze_until`.  The state monitoring table has {frozen_rows} pre-lock rows; this is an availability context, not a real wear quantity.

## Causality and limits

- Label leakage: **{diagnostics['label_leakage_check']['status']}**. Stage labels were read only after online tables were saved for post-hoc physical comparison.
- Prefix causality: **{diagnostics['prefix_causality_check']['status']}**. Delayed forecast update: **{diagnostics['delayed_forecast_check']['status']}**. Severe-direction causality: **{diagnostics['severe_direction_causality_check']['status']}**.
- Cache fingerprint: **{diagnostics['cache_fingerprint_check']['status']}**. No v3 output cache was used.
- Stable wear is a relatively slow, low-volatility region displaced from the initial baseline, not a final state. Severe wear is a possible later persistent instability after that plateau. No Stage1–Stage5 classifier is used online.
- These continuous internal signals do not by themselves prove physical wear amount. Independent morphology, debris, and mass-loss/quality-loss validation (ideally a third independent experiment) is still required before claiming real wear prediction.
"""
    return "\n".join(line.rstrip() for line in report.splitlines()) + "\n"
