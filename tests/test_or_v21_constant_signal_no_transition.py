from tests.or_v21_test_utils import engine,stream
def test_constant_signal_does_not_discover_state():
    frame,encoded=stream([0]*40);_,_,events,_=engine().run(frame,encoded)
    assert not (events.event=="STATE_DISCOVERED").any()
