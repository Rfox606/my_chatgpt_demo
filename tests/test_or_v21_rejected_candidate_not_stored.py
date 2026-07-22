from tests.or_v21_test_utils import engine,stream
def test_rejected_candidate_never_enters_state_memory():
    frame,encoded=stream([0]*8+[4]*3+[0]*8);_,proto,_,memory=engine().run(frame,encoded)
    assert (memory.candidate_id=="STABLE").all() and not (proto.stage==2).any()
