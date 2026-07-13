from continuous_state_v31.state_engine import EvidenceAccumulator


def test_exit_confirmation_accumulates_across_multiple_guards() -> None:
    exit_evidence = EvidenceAccumulator(300, 500, 150, "PLATEAU", "EXIT_CANDIDATE", "EXIT_CONFIRMED")
    # Valid evidence spans the 1000 and 1500 restart guards with stop interval 500, guard 100 and stride 5.
    for cycle in range(950, 1705, 5):
        guard = 1000 <= cycle < 1100 or 1500 <= cycle < 1600
        exit_evidence.step(guard, True, 5)
    assert exit_evidence.confirmed
    assert exit_evidence.valid_cycles >= 500
