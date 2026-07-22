import numpy as np
from tests.or_v21_test_utils import engine,stream

def test_snapshot_restores_replay_state_exactly():
    frame,encoded=stream([0]*8+[4]*8);first=engine();first.run(frame,encoded);payload=first.snapshot();second=engine();second.restore(payload)
    assert second.highest_stage==first.highest_stage and np.array_equal(second.states[1].level_proto,first.states[1].level_proto)
