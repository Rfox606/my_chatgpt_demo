import numpy as np

from temporal_prototype_v2.online import OnlineRunner
from tests.tp_v2_test_utils import config, source_bundle


def test_rollback_uses_only_unsupervised_quality_signals():
    cfg = config()
    runner = OnlineRunner(source_bundle(cfg), cfg, "B6_FULL_ADAPTATION")
    runner._checkpoint(0)
    runner.checkpoint["quality"] = (1.0, .1)
    runner.prototypes[0, 0] = 100.0
    runner.quality_history = [(0.0, 2.0)] * 100
    runner._rollback_check(100)
    assert runner.rollback_count == 1
