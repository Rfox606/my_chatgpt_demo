import numpy as np
import pandas as pd

from continuous_state_v2.config import ContinuousStateV2Config
from continuous_state_v2.source_rank_heads import train_source_head


def test_selection_validation_is_kept_separate_from_deployment_replay():
    c = ContinuousStateV2Config(max_pairs_per_gap_bin=100)
    n = 140; cycle = np.arange(n) * 100. + 10
    f = pd.DataFrame({"dataset":"s", "window_id":range(n), "window_index":range(n), "start_cycle":cycle-5, "end_cycle":cycle+5, "center_cycle":cycle, "baseline_window":0, "is_restart_guard":0, "x":cycle / cycle.max()})
    head, summary, _, _ = train_source_head(f, ("x",), "synthetic", c)
    assert "source_validation_auc_pre_refit" in summary
    assert "source_validation_auc_after_refit_replay" in summary
    assert head.pre_refit_metrics["source_validation_auc_pre_refit"] > .8
