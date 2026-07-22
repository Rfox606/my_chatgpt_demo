from tests.csv31_test_utils import run_plateau


def test_plateau_reference_is_frozen_after_lock() -> None:
    (states, *_), _, _, _ = run_plateau()
    locked = states.loc[states.plateau_locked.eq(1)]
    assert not locked.empty
    assert locked.plateau_centroid_signature.nunique() == 1
    assert locked.plateau_covariance_signature.nunique() == 1
    assert locked.plateau_reference_start_cycle.nunique() == 1
