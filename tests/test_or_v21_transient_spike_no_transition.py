from tests.or_v21_test_utils import engine,stream
def test_short_spike_is_rejected_without_new_state():
    frame,encoded=stream([0]*8+[4]*3+[0]*8);_,_,events,_=engine().run(frame,encoded)
    assert not (events.event=="STATE_DISCOVERED").any() and (events.event=="CANDIDATE_REJECTED").any()
