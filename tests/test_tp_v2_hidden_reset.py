import numpy as np

from temporal_prototype_v2.data import causal_sequences


def test_restart_mask_resets_sequence_history():
    first = np.zeros((6, 17), dtype=np.float32)
    second = first.copy(); second[:3] = 99
    restart = np.array([0, 0, 0, 1, 0, 0], dtype=bool)
    assert np.array_equal(causal_sequences(first, restart, 5)[3:], causal_sequences(second, restart, 5)[3:])
