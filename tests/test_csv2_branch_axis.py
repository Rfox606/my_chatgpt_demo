import numpy as np
import pandas as pd

from continuous_state_v2.branch_axis import build_branch_axis
from continuous_state_v2.config import ContinuousStateV2Config


def test_branch_axis_points_toward_exp2_terminal_residual():
    n=100; x=np.linspace(0,1,n)
    base=lambda terminal: pd.DataFrame({"center_cycle":np.arange(n),"is_restart_guard":0,"a":x,"b":terminal*x})
    w, table=build_branch_axis(base(-1),base(1),("a","b"),("a",),np.array([1.]),ContinuousStateV2Config())
    assert w[1] > 0
    assert table.loc[table.feature_name.eq("b"),"w_branch"].iloc[0] > 0
