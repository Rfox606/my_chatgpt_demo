import numpy as np,pandas as pd
from adaptive_awr_v12.config import AdaptiveAWRV12Config
from adaptive_awr_v12.feature_support import support,candidate_sets
def test_sparse_occupancy_not_in_sets():
 x=np.arange(100,dtype=float);f=pd.DataFrame({"AWR_adaptive":x,"BDall_xy_v2":x,"RS50_positive":x,"TES":x,"high_AWR_high_BD_occupancy":np.r_[np.zeros(98),1,1]});t=support(f,AdaptiveAWRV12Config());assert "high_AWR_high_BD_occupancy" not in ";".join(candidate_sets(t).features)
