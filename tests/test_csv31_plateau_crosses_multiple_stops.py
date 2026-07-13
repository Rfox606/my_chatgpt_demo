from tests.csv31_test_utils import run_plateau


def test_plateau_lock_crosses_two_restart_guards() -> None:
    (_, events, *_), _, _, _ = run_plateau()
    assert not events.empty
    assert int(events.iloc[0].guards_crossed_before_lock) >= 2
    assert float(events.iloc[0].plateau_reference_valid_cycles) >= 500
