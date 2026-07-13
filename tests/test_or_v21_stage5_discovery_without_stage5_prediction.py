from tests.or_v21_test_utils import engine,stream
def test_stage5_can_be_discovered_without_source_stage5_prediction():
    frame,encoded=stream([0]*6+[4]*6+[8]*6+[12]*6+[16]*6,posterior_stage=3);_,proto,events,_=engine().run(frame,encoded)
    discovered=events[events.event=="STATE_DISCOVERED"]
    assert len(discovered)>=2 and 5 in proto.stage.to_list() and (discovered.source_predicted_stage!=5).all()
