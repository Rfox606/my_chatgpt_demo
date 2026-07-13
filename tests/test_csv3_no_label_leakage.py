import pandas as pd
import pytest

from continuous_state_v3.data import assert_label_free


def test_stage_columns_are_rejected_at_v3_model_boundary():
    with pytest.raises(AssertionError):
        assert_label_free(pd.DataFrame({"stage": [1], "rs_mean": [0.]}))
