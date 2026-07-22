import pandas as pd
import pytest

from continuous_state_v31.data import assert_label_free
from continuous_state_v31.evaluation import label_leakage_check


def test_stage_columns_are_rejected_at_model_boundaries() -> None:
    with pytest.raises(AssertionError):
        assert_label_free(pd.DataFrame({"stage": [1]}))
    result = label_leakage_check(pd.DataFrame({"stage": [1]}), pd.DataFrame())
    assert result["status"] == "FAIL"
