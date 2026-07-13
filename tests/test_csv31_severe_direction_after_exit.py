import numpy as np

from tests.csv31_test_utils import run_plateau, state_config


def test_target_severe_direction_updates_only_after_exit_confirmation() -> None:
    (states, _, exits, updates, metadata), _, _, _ = run_plateau(state_config(weighted_oos_max=1.0, target_clip=1e12), shifted_after=1650, end_cycle=4500)
    assert not exits.empty
    assert not updates.empty
    assert np.isfinite(metadata["exit_cycle"])
    assert (updates.cycle >= updates.exit_confirmation_cycle).all()
    assert (updates.used_max_cycle <= updates.cycle).all()
    assert states.loc[states.center_cycle < metadata["exit_cycle"], "severe_direction_update"].sum() == 0
