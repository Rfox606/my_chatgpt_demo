from continuous_state_v31.state_engine import EvidenceAccumulator


def test_short_failure_run_retains_plateau_candidate_then_resets_at_150_valid_cycles() -> None:
    accumulator = EvidenceAccumulator(300, 500, 150, "SEARCHING", "CANDIDATE", "LOCKED")
    accumulator.step(False, True, 300)
    for _ in range(29):
        reset, _, _ = accumulator.step(False, False, 5)
        assert not reset
    assert accumulator.valid_cycles == 300
    reset, reason, _ = accumulator.step(False, False, 5)
    assert reset and reason == "FAILURE_VALID_CYCLES_REACHED"
    assert accumulator.valid_cycles == 0
