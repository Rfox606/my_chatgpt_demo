from tests.or_v21_test_utils import engine,stream

def test_initial_stage_is_inferred_from_signal_not_hardcoded_to_one():
    frame,encoded=stream([6]*20,posterior_stage=3);discovery=engine();discovery.run(frame,encoded)
    assert discovery.initial_audit["initial_target_stage"]==3
