import numpy as np
import pandas as pd

from continuous_state_v3.config import ContinuousStateV3Config
from continuous_state_v3.state_engine import build_target_context


def test_target_baseline_uses_initial_non_guard_windows_only():
    n = 12
    frame = pd.DataFrame({"dataset": ["Exp1"] * n, "window_id": range(n), "window_index": range(n), "start_cycle": np.arange(n) * 50. + 1., "end_cycle": np.arange(n) * 50. + 20., "center_cycle": np.arange(n) * 50. + 10., "baseline_window": [1] * 10 + [0, 0], "is_restart_guard": [0] * n, "f": [0., 2.] * 5 + [100., 200.]})
    context, _ = build_target_context(frame, ("f",), frame, {"f": 1.}, ContinuousStateV3Config())
    assert context.median0[0] == 1.
