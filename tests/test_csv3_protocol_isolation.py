import pandas as pd
import pytest

from continuous_state_v3.config import ContinuousStateV3Config
from continuous_state_v3.source_prior import build_source_model


def test_source_model_rejects_stage_label_before_any_prior_is_built():
    frame = pd.DataFrame({"dataset": ["Exp1"], "window_id": [0], "window_index": [0], "start_cycle": [1.], "end_cycle": [20.], "center_cycle": [10.], "baseline_window": [1], "is_restart_guard": [0], "f": [0.], "stage": [1]})
    with pytest.raises(AssertionError):
        build_source_model(frame, ("f",), {"f": 1.}, "A", False, ContinuousStateV3Config())
