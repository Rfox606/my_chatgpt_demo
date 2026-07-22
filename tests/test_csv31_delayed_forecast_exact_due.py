from dataclasses import replace

from continuous_state_v31.forecast import run_online_forecasts, train_frozen_models
from tests.csv31_test_utils import forecast_states, state_config


def test_rls_updates_only_after_due_observation_arrives() -> None:
    config = replace(state_config(), forecast_horizons_cycles=(20,), forecast_history_windows=10, ensemble_window=10)
    source = forecast_states(); target = forecast_states()
    predictions, *_ = run_online_forecasts(target, train_frozen_models(source, config), "A_test", config)
    updates = predictions.loc[predictions.online_model_updated_after_observation.eq(1)]
    assert not updates.empty
    assert (updates.rls_update_cycle >= updates.target_due_cycle).all()
    assert (updates.due_observation_cycle >= updates.target_due_cycle).all()
