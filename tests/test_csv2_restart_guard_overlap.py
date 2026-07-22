import pandas as pd

from continuous_state_v2.config import ContinuousStateV2Config
from continuous_state_v2.guards import add_restart_guard


def test_full_window_interval_overlap_is_guarded_not_just_center():
    frame = pd.DataFrame({"start_cycle": [490., 595.], "end_cycle": [510., 610.], "center_cycle": [500., 602.5]})
    guarded = add_restart_guard(frame, 100, ContinuousStateV2Config())
    assert guarded.is_restart_guard.tolist() == [1, 1]
    assert guarded.crosses_stop_boundary.tolist() == [1, 0]
    assert guarded.intersects_post_stop_guard.tolist() == [1, 1]
