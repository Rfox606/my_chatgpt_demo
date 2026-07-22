import numpy as np

from continuous_state_v2.common_axis import causal_rolling_median


def test_causal_smoothing_prefix_is_invariant_to_future_values():
    x=np.array([0.,2.,1.,3.,100.])
    assert np.allclose(causal_rolling_median(x,5)[:4],causal_rolling_median(x[:4],5))
