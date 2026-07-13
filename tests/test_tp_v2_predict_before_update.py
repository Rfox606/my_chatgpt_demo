import numpy as np

from temporal_prototype_v2.online import OnlineRunner
from tests.tp_v2_test_utils import config, source_bundle, target_frame


def test_current_prediction_does_not_depend_on_current_update():
    cfg = config(confidence_threshold=0.0, posterior_margin_threshold=0.0, entropy_threshold=10.0, min_memory_to_update=99)
    target = target_frame(cfg, 1)
    source = source_bundle(cfg)
    with_update, _, _, _ = OnlineRunner(source, cfg, "B6_FULL_ADAPTATION").run(target, permit_updates=True)
    without_update, _, _, _ = OnlineRunner(source, cfg, "B6_FULL_ADAPTATION").run(target, permit_updates=False)
    columns = [f"stage_posterior_{i}" for i in range(1, 6)]
    assert np.array_equal(with_update[columns].to_numpy(), without_update[columns].to_numpy())
