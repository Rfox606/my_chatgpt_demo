import pandas as pd
import pytest
from run_adaptive_cross_domain_awr_v12 import assert_unlabeled
@pytest.mark.parametrize("c",["stage","stage_label","Stage1to5"])
def test_rejects_labels(c):
 with pytest.raises(AssertionError):assert_unlabeled(pd.DataFrame({c:[1]}))
