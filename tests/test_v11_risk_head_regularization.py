import numpy as np
import pandas as pd

from adaptive_awr_v11.config import AdaptiveAWRV11Config
from adaptive_awr_v11.risk_head import RISK_FEATURES, fit_regularization_grid


def test_weighted_mean_soft_label_grid_and_bounds() -> None:
    config = AdaptiveAWRV11Config()
    stages = np.repeat(np.arange(1, 6), 30)
    frame = pd.DataFrame({feature: np.linspace(0, 1, len(stages)) + 0.02 * index for index, feature in enumerate(RISK_FEATURES)})
    frame["stage"] = stages
    frame["is_restart_guard"] = 0
    train = np.arange(len(frame)) % 3 != 0
    validation = ~train
    head, grid, validation_metrics = fit_regularization_grid(frame, train, validation, config)
    assert len(grid) == len(config.l2_grid)
    assert (grid["max_abs_nonintercept_beta"] <= 5.0).all()
    assert validation_metrics["stage4_training_included"]
    assert head.objective < 10.0
