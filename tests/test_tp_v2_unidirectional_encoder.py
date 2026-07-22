from temporal_prototype_v2.model import TemporalPrototypeNet


def test_encoder_is_strictly_unidirectional():
    model = TemporalPrototypeNet()
    model.assert_unidirectional()
    assert model.gru.bidirectional is False
