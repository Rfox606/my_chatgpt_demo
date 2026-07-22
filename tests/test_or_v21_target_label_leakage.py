import pytest
from ordered_regime_v21.data import reject_target_labels
from tests.or_v21_test_utils import stream
def test_target_labels_are_rejected():
    frame,_=stream([0,0])
    with pytest.raises(AssertionError): reject_target_labels(frame.assign(stage=1))
