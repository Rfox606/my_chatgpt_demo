import numpy as np

from temporal_prototype_v2.online import transition_matrix


def test_transition_matrix_only_permits_neighbours_and_rows_sum_to_one():
    matrix = transition_matrix()
    assert np.allclose(matrix.sum(axis=1), 1)
    assert matrix[0, 2:].sum() == 0
    assert matrix[4, :3].sum() == 0
