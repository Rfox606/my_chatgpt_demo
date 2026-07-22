from __future__ import annotations
import numpy as np
import pandas as pd
from .config import AdaptiveAWRV12Config

RISK_FEATURES = ("AWR_adaptive", "BDall_xy_v2", "RS50_positive", "TES", "high_AWR_high_BD_occupancy")
SPARSE = {"RS50_positive", "TES", "high_AWR_high_BD_occupancy"}

def support(frame: pd.DataFrame, c: AdaptiveAWRV12Config) -> pd.DataFrame:
    rows=[]
    for f in RISK_FEATURES:
        x=frame[f].to_numpy(float); finite=x[np.isfinite(x)]; n=max(len(x),1)
        frac=len(finite)/n; unique=len(np.unique(np.round(finite,12))); width=float(np.percentile(finite,99)-np.percentile(finite,1)) if len(finite) else 0.; spread=float(np.percentile(finite,75)-np.percentile(finite,25)) if len(finite) else 0.; pos=float(np.mean(finite>0)) if len(finite) else 0.; nonzero=float(np.mean(finite!=0)) if len(finite) else 0.; trans=int(np.count_nonzero(np.diff((finite>0).astype(int)))) if len(finite)>1 else 0
        bad=[]
        if frac<.99: bad.append("finite_fraction")
        if unique<20: bad.append("unique_count")
        if width<=1e-6: bad.append("robust_range")
        if spread<=1e-6: bad.append("IQR")
        if f in SPARSE and pos<.02: bad.append("positive_fraction")
        if f=="high_AWR_high_BD_occupancy":
            if pos<.05: bad.append("occupancy_positive_fraction")
            if trans<5: bad.append("occupancy_transition_count")
            if spread<.02: bad.append("occupancy_IQR")
        rows.append({"feature_name":f,"finite_fraction":frac,"unique_count":unique,"median":float(np.median(finite)) if len(finite) else np.nan,"IQR":spread,"robust_range_p99_minus_p01":width,"nonzero_fraction":nonzero,"positive_fraction":pos,"transition_count":trans,"eligible":not bad,"ineligible_reason":";".join(bad)})
    return pd.DataFrame(rows)

def candidate_sets(table: pd.DataFrame) -> pd.DataFrame:
    okay=set(table.loc[table.eligible,"feature_name"]); definitions=[("F1",["AWR_adaptive"]),("F2",["AWR_adaptive","BDall_xy_v2"]),("F3",["AWR_adaptive","BDall_xy_v2","RS50_positive"]),("F4",["AWR_adaptive","BDall_xy_v2","TES"]),("F5",["AWR_adaptive","BDall_xy_v2","RS50_positive","TES"]),("F6",[f for f in RISK_FEATURES if f in okay])]
    seen=set();rows=[]
    for name,features in definitions:
        features=[f for f in features if f in okay]
        if "AWR_adaptive" not in features or tuple(features) in seen: continue
        seen.add(tuple(features));rows.append({"candidate_set":name,"features":";".join(features),"feature_count":len(features)})
    return pd.DataFrame(rows)
