import numpy as np
import pandas as pd

from continuous_state_v2.config import ContinuousStateV2Config
from continuous_state_v2.online_nuisance_adapter import run_target_online
from continuous_state_v2.state_metrics import make_state_space


def test_online_rows_preserve_pre_update_outputs():
    c=ContinuousStateV2Config(); n=120; cycle=np.arange(n)*5.+10
    frame=pd.DataFrame({"dataset":"t","window_id":range(n),"window_index":range(n),"start_cycle":cycle-2,"end_cycle":cycle+2,"center_cycle":cycle,"baseline_window":1,"is_restart_guard":0,"a":np.linspace(0,1,n),"b":np.linspace(0,.1,n)})
    space=make_state_space(frame,("a","b"),("a",),np.array([1.]),np.array([0.,1.]),c)
    support=pd.DataFrame({"feature_name":["a","b"],"p01":[-99.,-99.],"p99":[99.,99.]})
    scored,_=run_target_online(frame,space,("a","b"),support,("a",),np.array([1.]),("a",),np.array([1.]),c)
    assert np.allclose(scored.P_common,scored.pre_update_P_common)
    assert np.allclose(scored.BD,scored.pre_update_BD)
