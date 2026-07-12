import numpy as np,pandas as pd
from adaptive_awr_v12.config import AdaptiveAWRV12Config
from adaptive_awr_v12.risk_head import fit,select_thresholds
def test_quantile_thresholds_ordered():
 x=np.linspace(-3,3,100);f=pd.DataFrame({"AWR_adaptive":x,"stage":np.repeat([1,2,3,4,5],20)});h=fit(f,["AWR_adaptive"],.05,AdaptiveAWRV12Config());q,_=select_thresholds(f,h,AdaptiveAWRV12Config());assert q["watch_logit_threshold"]<q["high_logit_threshold"]
