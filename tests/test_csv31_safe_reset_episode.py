import pandas as pd

from continuous_state_v31.evaluation import safe_ensemble_reset_check


def test_each_reset_episode_has_one_transition() -> None:
    predictions = pd.DataFrame({"ensemble_state": ["FROZEN", "FROZEN"], "reset_transition": ["", ""],
                                "protocol_id": ["A", "A"], "output_name": ["D", "D"], "horizon_cycles": [500, 500],
                                "reset_episode_id": [1, 1], "freeze_until_cycle": [1500.0, 1500.0]})
    log = pd.DataFrame({"reset_transition": ["ACTIVE_TO_FROZEN"], "reset_episode_id": [1]})
    result = safe_ensemble_reset_check(predictions, log)
    assert result["status"] == "PASS"
    assert result["independent_reset_episode_count"] == 1
