from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import GradualStateMonitoringV34Config


def write_report(path: Path, config: GradualStateMonitoringV34Config, methods: pd.DataFrame,
                 adaptation: pd.DataFrame, provenance: dict[str, object]) -> Path:
    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        existing = [column for column in columns if column in frame.columns]
        if not len(frame):
            return "No rows generated."
        selected = frame.loc[:, existing]
        # Avoid making the report depend on pandas' optional ``tabulate`` extra.
        header = "| " + " | ".join(existing) + " |"
        rule = "| " + " | ".join("---" for _ in existing) + " |"
        body = ["| " + " | ".join(str(value) for value in row) + " |" for row in selected.itertuples(index=False, name=None)]
        return "\n".join([header, rule, *body])

    conclusions = methods[methods.metric.isin(["Top-8 event recall", "proximity_weighted_score", "boundary_score_percentile"])]
    text = f"""# V3.4 source weak supervision and target-label-free online adaptation

This report keeps four phases separate:

1. **Source weak-label supervision:** Gaussian soft labels around source-only Stage boundaries trained the shared Adapter + causal TCN.
2. **Target label-free online adaptation:** DirectTransfer, CalibrationOnly, and OnlineAdaptation consumed only declared target force inputs. The first {config.calibration_windows} windows were calibration-only.
3. **Target labels for offline evaluation:** target Stage mapping was opened only after all online transition-score and candidate-event files had frozen.
4. **TargetSupervisedUpperBound:** this is a deliberately offline labelled reference, not an online-monitoring result.

The TCN receptive field is {config.receptive_field} windows (required: at least 128). Fixed seeds: {list(config.random_seeds)}.

## Aggregated comparison

{table(conclusions, ["direction", "setting", "metric", "value_mean", "value_std", "seed_count"])}

## Adaptation behaviour

{table(adaptation, ["direction", "setting", "adapter_update_fraction", "adapter_frozen_fraction", "candidate_event_count", "stage_internal_event_count"])}

## Required interpretation questions

The comparison table is intentionally descriptive: it does not issue a global PASS/FAIL. Compare OnlineAdaptation with DirectTransfer and CalibrationOnly for Top-8 recall, proximity-weighted score, and boundary-score percentile; inspect candidate counts and frozen fractions before claiming a gain. Extra events are retained as `unmatched_candidate_event`, and their post-hoc Stage locations are reported rather than automatically called false alarms.

## Label provenance

```json
{provenance}
```
"""
    output = path / "gradual_state_monitoring_v34_report.md"
    output.write_text(text, encoding="utf-8")
    return output
