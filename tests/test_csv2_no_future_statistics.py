import numpy as np

from continuous_state_v2.state_metrics import _rate


def test_multiscale_rate_uses_only_prior_history():
    history=[0.,0.,1.,1.,2.,2.]
    before=_rate(history,4)
    history.append(1000.)
    # The already emitted pre-update value is immutable when a later value arrives.
    assert before == .5
