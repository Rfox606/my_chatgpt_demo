import numpy as np

from temporal_prototype_v2.online import OnlineRunner
from tests.tp_v2_test_utils import config, source_bundle, target_frame


def test_teacher_is_ema_updated_after_a_student_update():
    cfg = config(confidence_threshold=0.0, posterior_margin_threshold=0.0, entropy_threshold=10.0)
    runner = OnlineRunner(source_bundle(cfg), cfg, "B6_FULL_ADAPTATION")
    sequence = target_frame(cfg, 1)[list(cfg.input_features)].to_numpy(np.float32)
    sequence = np.repeat(sequence[None, :, :], cfg.sequence_length, axis=1)[0]
    item = {"state": 1, "embedding": np.zeros(16), "teacher_ordinal": np.repeat([[.9, .05, .03, .01, .01]], 1, axis=0)[0],
            "student_ordinal": np.repeat([[.9, .05, .03, .01, .01]], 1, axis=0)[0], "posterior": np.repeat([[.9, .05, .03, .01, .01]], 1, axis=0)[0],
            "health": 0., "confidence": .99, "entropy": .1, "window_index": 0, "center_cycle": 1., "restart": False, "TES": 0., "sequence": sequence}
    runner.memory[0] = [item, {**item, "window_index": 60}]
    runner.accepted_total = 2
    before = [p.detach().clone() for p in runner.teacher.parameters()]
    runner._online_update(60)
    assert any(not np.array_equal(a.detach().numpy(), b.detach().numpy()) for a, b in zip(before, runner.teacher.parameters()))
