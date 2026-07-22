import numpy as np,pandas as pd
from adaptive_awr_v12.causal_metrics import refs,Tracker
from adaptive_awr_v12.config import AdaptiveAWRV12Config
def test_tracker_prefix_is_identical():
 c=AdaptiveAWRV12Config();r=refs(np.zeros(20),np.zeros(20),np.zeros(20),c)
 def run(x):
  t=Tracker(r,1,1,c);return [t.step(float(v),float(v),0)["TES"] for v in x]
 a=np.arange(120.);b=a.copy();b[80:]+=100;assert run(a)[:80]==run(b)[:80]
