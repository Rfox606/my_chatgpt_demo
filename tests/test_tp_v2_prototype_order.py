import numpy as np

from temporal_prototype_v2.online import OnlineRunner
from tests.tp_v2_test_utils import config, source_bundle


def test_crossing_prototype_update_is_rejected():
    cfg = config()
    runner = OnlineRunner(source_bundle(cfg), cfg, "B3_DYNAMIC_PROTOTYPE")
    assert runner._update_prototype(1, np.r_[100.0, np.zeros(15)], 1.0, 0) is False
    assert any(event["event"] == "PROTOTYPE_ORDER_REJECT" for event in runner.events)
