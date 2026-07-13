import pandas as pd

from continuous_state_v3.config import ContinuousStateV3Config
from continuous_state_v3.guards import add_restart_guard


def test_guard_uses_window_interval_not_just_center():
    frame = pd.DataFrame({"dataset": ["Exp1"], "window_id": [1], "window_index": [1], "start_cycle": [495.], "end_cycle": [520.], "center_cycle": [507.], "baseline_window": [0], "rs_mean": [0.]})
    guarded = add_restart_guard(frame, ContinuousStateV3Config())
    assert int(guarded.is_restart_guard.iloc[0]) == 1
