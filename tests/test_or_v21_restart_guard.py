from tests.or_v21_test_utils import engine,stream
def test_restart_guard_blocks_candidate_updates():
    frame,encoded=stream([0]*5+[4]*10,guards=[0]*5+[1]*10);_,proto,_,_=engine().run(frame,encoded)
    assert not (proto.stage==2).any()
