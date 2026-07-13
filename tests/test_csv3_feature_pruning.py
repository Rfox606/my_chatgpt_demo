import numpy as np
import pandas as pd

from continuous_state_v3.config import ContinuousStateV3Config, FEATURES
from continuous_state_v3.feature_pruning import direction_free_auc, prune_features


def test_direction_free_auc_treats_negative_direction_as_informative():
    values = np.array([4., 3., 2., 1.]); earlier = np.array([0, 1]); later = np.array([2, 3])
    assert direction_free_auc(values, earlier, later) == 1.


def test_exact_duplicate_is_always_removed_from_source_only_prune():
    rows = 300; data = {"dataset": ["Exp1"] * rows, "window_id": range(rows), "window_index": range(rows), "start_cycle": np.arange(rows) * 10., "end_cycle": np.arange(rows) * 10. + 20., "center_cycle": np.arange(rows) * 10. + 10., "baseline_window": [0] * rows, "is_restart_guard": [0] * rows}
    for index, feature in enumerate(FEATURES): data[feature] = np.arange(rows, dtype=float) * (index + 1) + index
    _, audit = prune_features(pd.DataFrame(data), "A", ContinuousStateV3Config(max_pairs_per_gap_bin=50))
    assert audit.loc[audit.feature_name.eq("rs_absmean"), "kept"].iloc[0] == 0
