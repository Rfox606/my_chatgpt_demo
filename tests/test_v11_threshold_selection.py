import numpy as np
import pandas as pd

from adaptive_awr_v11.config import AdaptiveAWRV11Config
from adaptive_awr_v11.risk_head import RISK_FEATURES, RobustScaler, SoftRiskHead, select_logit_thresholds


def test_logit_thresholds_are_ordered_and_separated() -> None:
    stage = np.repeat([1, 2, 3, 4, 5], 20)
    values = np.linspace(-3.0, 3.0, len(stage))
    frame = pd.DataFrame({feature: values for feature in RISK_FEATURES})
    frame["stage"] = stage
    frame["is_restart_guard"] = 0
    head = SoftRiskHead(RobustScaler(np.zeros(5), np.ones(5)), np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), 1.0, True, "synthetic", 0.0)
    thresholds = select_logit_thresholds(frame, head, AdaptiveAWRV11Config())
    assert thresholds["watch_logit_threshold"] < thresholds["high_logit_threshold"]
    assert thresholds["high_logit_threshold"] - thresholds["watch_logit_threshold"] >= 0.25
