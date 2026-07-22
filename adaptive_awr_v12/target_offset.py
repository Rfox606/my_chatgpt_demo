from __future__ import annotations
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class Offset:
    mode:str; source_early_median:float; source_early_iqr:float; target_calibration_median:float; target_calibration_iqr:float; raw_shift:float; bounded_shift:float; shrinkage:float; applied_shift:float; calibration_status:str
    def apply(self,z): return np.asarray(z,float)+self.applied_shift
def make(mode,source,target):
    sm=float(np.median(source)) if len(source) else 0.;si=float(np.percentile(source,75)-np.percentile(source,25)) if len(source) else 0.;tm=float(np.median(target)) if len(target) else 0.;ti=float(np.percentile(target,75)-np.percentile(target,25)) if len(target) else 0.
    if mode=="T0": return Offset(mode,sm,si,tm,ti,0,0,0,0,"NO_CALIBRATION")
    if len(target)<30:return Offset(mode,sm,si,tm,ti,0,0,0,0,"DISABLED_TOO_FEW_WINDOWS")
    if len(np.unique(np.round(source,12)))<20:return Offset(mode,sm,si,tm,ti,0,0,0,0,"DISABLED_DEGENERATE_SOURCE_REFERENCE")
    raw=sm-tm;bound=float(np.clip(raw,-1,1));shrink=1. if mode=="T1" else len(target)/(len(target)+200.);return Offset(mode,sm,si,tm,ti,raw,bound,shrink,shrink*bound,"APPLIED")
