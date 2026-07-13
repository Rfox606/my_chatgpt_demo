import numpy as np

from temporal_prototype_v2.online import OnlineRunner
from tests.tp_v2_test_utils import config, source_bundle


def test_snapshot_payload_restores_model_and_prototypes():
    cfg = config()
    first = OnlineRunner(source_bundle(cfg), cfg, "B6_FULL_ADAPTATION")
    payload = first._snapshot_payload()
    second = OnlineRunner(source_bundle(cfg), cfg, "B6_FULL_ADAPTATION")
    second.prototypes += 3
    second.restore_adaptation(payload)
    assert np.array_equal(second.prototypes, first.prototypes)
