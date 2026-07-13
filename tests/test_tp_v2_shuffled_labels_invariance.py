import numpy as np

from temporal_prototype_v2.data import unlabeled_target
from temporal_prototype_v2.online import OnlineRunner
from tests.tp_v2_test_utils import config, source_bundle, target_frame


def test_shuffling_posthoc_labels_cannot_change_online_outputs():
    cfg = config()
    raw = target_frame(cfg, 6)
    source = source_bundle(cfg)
    a, _, _, _ = OnlineRunner(source, cfg, "B2_STATIC_HMM").run(unlabeled_target(raw.assign(stage=np.arange(6) % 5 + 1)))
    b, _, _, _ = OnlineRunner(source, cfg, "B2_STATIC_HMM").run(unlabeled_target(raw.assign(stage=np.arange(6)[::-1] % 5 + 1)))
    cols = [f"stage_posterior_{i}" for i in range(1, 6)]
    assert np.array_equal(a[cols].to_numpy(), b[cols].to_numpy())
