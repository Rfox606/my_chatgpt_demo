import pytest

from temporal_prototype_v2.data import reject_target_labels
from temporal_prototype_v2.online import OnlineRunner
from tests.tp_v2_test_utils import config, source_bundle, target_frame


def test_target_label_leakage_is_rejected_at_online_entry():
    cfg = config()
    target = target_frame(cfg).assign(stage=1, stage_label="Stage 1")
    with pytest.raises(AssertionError):
        OnlineRunner(source_bundle(cfg), cfg, "B0_STATIC_SOURCE").run(target)


def test_generic_label_guard_rejects_all_forbidden_names():
    with pytest.raises(AssertionError):
        reject_target_labels(target_frame().assign(Stage1to5=1))
