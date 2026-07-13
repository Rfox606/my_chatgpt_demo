from tests.csv31_test_utils import run_plateau


def test_guard_pauses_plateau_evidence_without_reset() -> None:
    (states, *_), _, _, _ = run_plateau()
    guarded = states.loc[states.is_restart_guard.eq(1)].reset_index(drop=True)
    assert not guarded.empty
    for _, row in guarded.iterrows():
        assert row.evidence_increment_cycles == 0
        assert row.plateau_reset_event == 0
