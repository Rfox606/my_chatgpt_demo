import pandas as pd
import pytest

from continuous_state_v2.data import assert_label_free


def test_v2_rejects_historical_labels_at_model_boundary():
    with pytest.raises(AssertionError):
        assert_label_free(pd.DataFrame({"stage": [1], "x": [0.]}))
