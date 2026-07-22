from temporal_prototype_v2.online import OnlineRunner
from tests.tp_v2_test_utils import config, source_bundle, target_frame


def test_prototype_is_not_updated_when_gate_rejects():
    cfg = config(confidence_threshold=1.1)
    runner = OnlineRunner(source_bundle(cfg), cfg, "B3_DYNAMIC_PROTOTYPE")
    runner.run(target_frame(cfg, 4))
    assert runner.prototype_updates == 0
    assert runner.support.sum() == 0
