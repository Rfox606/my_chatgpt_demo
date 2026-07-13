from __future__ import annotations

import numpy as np
import pandas as pd

from continuous_state_v1.candidate_selection import select_physical_validation_candidates
from continuous_state_v1.config import ContinuousStateV1Config


def test_candidate_windows_keep_required_spacing_per_type() -> None:
    config = ContinuousStateV1Config(candidate_top_k_per_type=20, candidate_min_spacing_cycles=500)
    n = 80
    centers = np.arange(n, dtype=float) * 100
    frame = pd.DataFrame(
        {
            "direction_id": "synthetic", "source_dataset": "source", "target_dataset": "target",
            "window_index": np.arange(n), "start_cycle": centers - 5, "end_cycle": centers + 5,
            "center_cycle": centers, "AWR_raw": np.arange(n), "AWR_rel": np.arange(n),
            "AWR_scaled": np.arange(n), "BD": np.arange(n), "BD_diag": np.arange(n),
            "oos_fraction": np.linspace(0, 1, n), "is_restart_guard": 0,
            **{f"contrib_{feature}": np.arange(n, dtype=float) for feature in config.stable_plus_features},
        }
    )
    candidates = select_physical_validation_candidates(frame, config)
    for _, group in candidates.groupby("candidate_type"):
        gaps = np.diff(np.sort(group.center_cycle.to_numpy(float)))
        assert (gaps >= 500).all()
