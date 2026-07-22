import numpy as np
import pandas as pd

from continuous_state_v2.config import ContinuousStateV2Config, STABLE_PLUS_FEATURES
from continuous_state_v2.feature_pruning import prune_features


def test_explicit_duplicate_is_removed_without_target_data():
    c=ContinuousStateV2Config(max_pairs_per_gap_bin=50); n=100; cycle=np.arange(n)*100.+10
    frame=pd.DataFrame({"window_id":range(n),"window_index":range(n),"start_cycle":cycle-5,"end_cycle":cycle+5,"center_cycle":cycle,"is_restart_guard":0,**{f:cycle/(i+1) for i,f in enumerate(STABLE_PLUS_FEATURES)}})
    kept,audit=prune_features(frame,"s",c)
    assert "rs_absmean" not in kept
    assert audit.loc[audit.feature_name.eq("rs_absmean"),"drop_reason"].iloc[0] == "EXACT_DUPLICATE_OF_rs_mean"
