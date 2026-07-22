import numpy as np,pandas as pd
from adaptive_awr_v12.config import AdaptiveAWRV12Config
from adaptive_awr_v12.risk_head import fit
def test_iqr_bound_limits_effective_beta():
 x=np.linspace(0,1e-5,60);f=pd.DataFrame({"AWR_adaptive":x,"stage":np.tile([1,2,3,4,5],12)});h=fit(f,["AWR_adaptive"],.05,AdaptiveAWRV12Config());assert h.beta[1]/h.width[0]<=5+1e-9
