import pandas as pd

from continuous_state_v31.evaluation import safe_ensemble_reset_check


def test_frozen_episode_cannot_extend_freeze_until() -> None:
    predictions = pd.DataFrame({"ensemble_state": ["FROZEN", "FROZEN"], "reset_transition": ["", ""],
                                "protocol_id": ["A", "A"], "output_name": ["D", "D"], "horizon_cycles": [500, 500],
                                "reset_episode_id": [1, 1], "freeze_until_cycle": [1500.0, 1510.0]})
    assert safe_ensemble_reset_check(predictions, pd.DataFrame())["status"] == "FAIL"
