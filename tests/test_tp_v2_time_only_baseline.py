import numpy as np

from temporal_prototype_v2.online import time_only_hmm, transition_matrix


def test_time_only_hmm_receives_no_signal_and_is_deterministic():
    transition = transition_matrix()
    assert np.array_equal(time_only_hmm(12, transition), time_only_hmm(12, transition))
