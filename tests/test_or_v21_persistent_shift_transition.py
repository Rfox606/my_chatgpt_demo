from tests.or_v21_test_utils import engine,stream
def test_persistent_shift_discovers_next_state_after_shift():
    frame,encoded=stream([0]*8+[4]*10);_,_,events,_=engine().run(frame,encoded);found=events[events.event=="STATE_DISCOVERED"]
    assert len(found)==1 and found.new_stage.iloc[0]==2 and found.window_index.iloc[0]>=8
