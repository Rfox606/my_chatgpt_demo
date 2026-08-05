from __future__ import annotations

import numpy as np
import pandas as pd

from gradual_state_monitoring_v33.data import MAIN_FEATURES

from .config import GradualStateMonitoringV332Config


ONLINE_IDENTIFIER_COLUMNS = ("dataset", "window_index", "center_cycle")


def validate_label_free_online_frame(frame: pd.DataFrame, config: GradualStateMonitoringV332Config) -> pd.DataFrame:
    """Return the exact label-free online boundary and reject prohibited columns.

    This is deliberately called before every V3.3.2 forecasting/state replay.  The
    evaluator has a separate module and receives only already-frozen online output.
    """
    lowered = {str(column).lower() for column in frame.columns}
    leaked = sorted(lowered.intersection(config.label_prohibited_online_fields))
    if leaked:
        raise AssertionError(f"V3.3.2 online path rejects label/outcome columns: {leaked}")
    required = set(ONLINE_IDENTIFIER_COLUMNS).union(MAIN_FEATURES)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"V3.3.2 online input misses required columns: {missing}")
    output = frame.loc[:, [*ONLINE_IDENTIFIER_COLUMNS, *MAIN_FEATURES]].copy()
    if not np.isfinite(output.loc[:, MAIN_FEATURES].to_numpy(float)).all():
        raise ValueError("V3.3.2 refuses non-finite online force features")
    return output.sort_values(["dataset", "window_index"], kind="stable").reset_index(drop=True)
