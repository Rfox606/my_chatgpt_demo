import numpy as np
import pandas as pd

from adaptive_awr_v11.causal_metrics import CausalMetricTrackerV11, boundary_guard_metadata, build_metric_references
from adaptive_awr_v11.config import AdaptiveAWRV11Config


def test_boundary_guard_cleans_tes_but_leaves_raw_signal_visible() -> None:
    config = AdaptiveAWRV11Config()
    frame = pd.DataFrame({"window_index": [0], "start_cycle": [495.0], "end_cycle": [510.0], "center_cycle": [502.5]})
    guard = boundary_guard_metadata(frame, config)
    assert guard.loc[0, "is_restart_guard"] == 1
    features = pd.DataFrame({feature: np.zeros(20) for feature in config.stable_plus_features})
    refs = build_metric_references(np.zeros(20), np.zeros(20), np.zeros(20), features, config)
    tracker = CausalMetricTrackerV11(refs, config, 1.0, 1.0)
    for _ in range(3):
        tracker.step(0.0, 0.0, 0.0, False)
    metrics = tracker.step(5.0, 5.0, 5.0, True)
    assert metrics["TES_raw"] > 0.0
    assert metrics["TES_clean"] == 0.0
