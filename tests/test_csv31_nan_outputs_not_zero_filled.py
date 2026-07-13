from dataclasses import replace

import numpy as np

from continuous_state_v31.forecast import run_online_forecasts, train_frozen_models
from tests.csv31_test_utils import forecast_states, state_config


def test_unavailable_severe_head_is_nan_and_not_zero_filled() -> None:
    config = replace(state_config(), forecast_horizons_cycles=(20,), forecast_history_windows=10, ensemble_window=10)
    source = forecast_states(); target = forecast_states()
    predictions, *_ = run_online_forecasts(target, train_frozen_models(source, config), "A_test", config)
    unavailable = predictions.loc[(predictions.output_name == "S_severe_candidate") & (predictions.prediction_origin_cycle < 10.5 + 60 * 5)]
    assert not unavailable.empty
    assert unavailable.prediction_available.eq(0).all()
    assert np.isnan(unavailable.safe_prediction).all()
