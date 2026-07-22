import pandas as pd
from adaptive_awr_v12.config import AdaptiveAWRV12Config
from adaptive_awr_v12.feature_support import support
def test_zero_variance_occupancy_is_ineligible():
 f=pd.DataFrame({"AWR_adaptive":range(30),"BDall_xy_v2":range(30),"RS50_positive":range(30),"TES":range(30),"high_AWR_high_BD_occupancy":[0]*30})
 t=support(f,AdaptiveAWRV12Config()).set_index("feature_name")
 assert not t.loc["high_AWR_high_BD_occupancy","eligible"]
