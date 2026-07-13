from __future__ import annotations

import pandas as pd


def make_report(validation: pd.DataFrame, pruning: pd.DataFrame, common_status: str, common: pd.DataFrame, diagnostics: pd.DataFrame, adapter: pd.DataFrame, metrics: pd.DataFrame, benefits: pd.DataFrame, candidates: pd.DataFrame) -> str:
    report = f"""# Continuous State Monitoring v2

## Required answers

1. Corrected source pre-refit validation AUC:

```text
{validation[['direction_id','source_validation_auc_pre_refit','source_validation_auc_after_refit_replay']].to_string(index=False)}
```

2. Restart guard now uses full window/guard-region intersection at 50, 100, and 150 cycles; the main analysis uses 100 cycles. Thus overlap candidates near each 500-cycle boundary are excluded rather than selected.
3. Repeated/highly correlated feature audit:

```text
{pruning[pruning.kept.eq(0)].to_string(index=False)}
```

4–5. Common-direction features and status: **{common_status}**.

```text
{common.to_string(index=False)}
```

6–9. Target segment diagnostics (P, BD, and terminal branch score):

```text
{diagnostics.to_string(index=False)}
```

The terminal branch score is a residual direction relative to Exp1/Exp2 terminal references. It may include condition differences and requires physical validation before any stable–severe interpretation.
10. Physical-validation candidate regions:

```text
{candidates.head(40).to_string(index=False) if not candidates.empty else 'No candidate regions.'}
```

11–13. Adapter activity:

```text
{adapter.groupby('direction_id')[['adapter_updated','adapter_rollback']].sum().to_string() if not adapter.empty else 'No target adapter records.'}
```

The adapter is constrained to the nuisance subspace and baseline-replay rollback; it is not permitted to modify P/branch source axes or BD baselines.
14–15. Strict delayed-observation forecast metrics and benefit decisions:

```text
{metrics.to_string(index=False)}
{benefits.to_string(index=False)}
```

16. A more complex neural adapter is not justified solely by trajectory shape. It should only be considered if the fixed support, replay, and delayed-prediction diagnostics remain acceptable.
17. P, B, and BD are not wear percentages, absolute wear quantities, failure probabilities, or Stage5 probabilities.
18. Exp1 and Exp2 terminal references are deliberately not forced to be equal: Exp1 late behavior is treated as a stable-wear reference, while Exp2 late behavior is a more severe reference; the model uses a common trunk plus a terminal residual branch.
"""
    return "\n".join(line.rstrip() for line in report.splitlines()) + "\n"
