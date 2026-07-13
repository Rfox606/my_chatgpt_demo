from dataclasses import replace

import numpy as np
import pandas as pd

from continuous_state_v3.config import ContinuousStateV3Config
from continuous_state_v3.forecast import ScalarRLS, run_online_forecasts, train_frozen_models


def _states(n: int = 180) -> pd.DataFrame:
    cycle = np.arange(n) * 10. + 10.
    d = np.linspace(0., 4., n)
    return pd.DataFrame({"dataset": ["Exp1"] * n, "window_id": range(n), "window_index": range(n), "start_cycle": cycle - 9., "end_cycle": cycle + 9., "center_cycle": cycle, "baseline_window": [0] * n, "is_restart_guard": [0] * n, "D_state": d, "V20_norm": np.gradient(d), "V50_norm": np.gradient(d), "V100_norm": np.gradient(d), "direction_consistency": [1.] * n, "A_state": [0.] * n, "state_volatility_20": [.01] * n, "state_volatility_50": [.01] * n, "instability_score": d / 4., "S_severe_candidate": d / 3., "weighted_oos": [0.] * n, "severe_direction_available": [1] * n})


def test_forecast_updates_only_after_delayed_observation():
    config = replace(ContinuousStateV3Config(), forecast_horizons_cycles=(100, 500), ensemble_window=10)
    states = _states(); frozen = train_frozen_models(states, config)
    predictions, _, _ = run_online_forecasts(states, frozen, "A", config)
    due = predictions.loc[predictions.observation_available.eq(1)]
    assert not due.empty
    assert set(due.online_model_updated_after_observation.unique()).issubset({0, 1})
    assert predictions.ensemble_alpha.between(0., 1.).all()


def test_safe_reset_restores_frozen_rls_parameters():
    model = ScalarRLS(np.array([2., 3.]), np.eye(2), np.array([2., 3.]), .999)
    model.theta[:] = 9.
    model.reset()
    assert np.allclose(model.theta, model.frozen_theta)
