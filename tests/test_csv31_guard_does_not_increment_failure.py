from tests.csv31_test_utils import plateau_prior, run_plateau, state_config


def test_guard_does_not_increment_failure_counter() -> None:
    prior = plateau_prior()
    # A high D threshold makes all normal windows failures; guard rows must still pause it.
    prior = type(prior)(1000.0, prior.v50_threshold, prior.v100_threshold, prior.volatility_threshold, prior.quantile)
    (states, *_), _, _, _ = run_plateau(state_config())
    # Re-run helper result is plateau-positive; directly validate the generic state-machine rule.
    from continuous_state_v31.state_engine import EvidenceAccumulator
    accumulator = EvidenceAccumulator(300, 500, 150, "SEARCHING", "CANDIDATE", "LOCKED")
    accumulator.step(False, False, 100)
    before = accumulator.failure_cycles
    accumulator.step(True, False, 5)
    assert accumulator.failure_cycles == before
