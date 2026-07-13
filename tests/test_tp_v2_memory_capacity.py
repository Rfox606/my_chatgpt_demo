import numpy as np

from temporal_prototype_v2.online import OnlineRunner
from tests.tp_v2_test_utils import config, source_bundle


def test_memory_is_bounded_per_state():
    cfg = config(memory_per_state=3)
    runner = OnlineRunner(source_bundle(cfg), cfg, "B4_TEACHER_MEMORY")
    for index in range(10):
        runner._store_memory({"state": 1, "confidence": index / 10, "embedding": np.zeros(16), "window_index": index})
    assert len(runner.memory[0]) == 3
