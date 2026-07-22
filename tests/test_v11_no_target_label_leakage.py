import pandas as pd
import pytest

from adaptive_awr_v11.config import AdaptiveAWRV11Config
from run_adaptive_cross_domain_awr_v11 import assert_target_unlabeled, target_setup


@pytest.mark.parametrize("column", ["stage", "stage_label", "Stage1to5"])
def test_target_label_columns_are_rejected(column: str) -> None:
    with pytest.raises(AssertionError):
        assert_target_unlabeled(pd.DataFrame({"window_index": [0], column: [1]}))


def test_target_alignment_setup_rejects_target_stage() -> None:
    with pytest.raises(AssertionError):
        target_setup(pd.DataFrame({"window_index": [0], "stage": [1]}), {}, AdaptiveAWRV11Config())
