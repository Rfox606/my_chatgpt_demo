import numpy as np

from temporal_prototype_v2.online import OnlineRunner
from tests.tp_v2_test_utils import config, source_bundle, target_frame


def test_prefix_predictions_are_identical_when_run_independently():
    cfg = config()
    target = target_frame(cfg, 9)
    source = source_bundle(cfg)
    full, _, _, _ = OnlineRunner(source, cfg, "B2_STATIC_HMM").run(target)
    prefix, _, _, _ = OnlineRunner(source, cfg, "B2_STATIC_HMM").run(target.iloc[:5].copy())
    cols = [f"stage_posterior_{i}" for i in range(1, 6)] + ["final_health_score"]
    assert np.array_equal(full.loc[:4, cols].to_numpy(), prefix[cols].to_numpy())
