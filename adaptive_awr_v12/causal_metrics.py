from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
from .config import AdaptiveAWRV12Config

def finite(x: Iterable[float]) -> np.ndarray:
    a=np.asarray(list(x),dtype=float).reshape(-1); return a[np.isfinite(a)]
def iqr(x: Iterable[float], eps=1e-9) -> float:
    a=finite(x); return float(max(np.percentile(a,75)-np.percentile(a,25),eps)) if len(a) else eps
def mad(x: Iterable[float], eps=1e-9) -> float:
    a=finite(x); return float(max(np.median(np.abs(a-np.median(a)))*1.4826,eps)) if len(a) else eps
def zplus(v: float, ref: Iterable[float], eps=1e-9) -> float:
    a=finite(ref); return float(max((v-np.median(a))/iqr(a,eps),0)) if len(a) and np.isfinite(v) else 0.0
def sigmoid(x: float|np.ndarray) -> float|np.ndarray: return 1/(1+np.exp(-np.clip(x,-35,35)))
def guard(frame: pd.DataFrame, cycles: int, config: AdaptiveAWRV12Config) -> pd.DataFrame:
    rows=[]
    for r in frame.itertuples(index=False):
        start,end,center=float(r.start_cycle),float(r.end_cycle),float(r.center_cycle); crossed=int(np.floor(start/config.known_stop_interval_cycles)!=np.floor(end/config.known_stop_interval_cycles)); b=round(center/config.known_stop_interval_cycles)*config.known_stop_interval_cycles; since=center-b
        rows.append({"window_index":int(r.window_index),"is_restart_guard":int(crossed or 0<=since<=cycles),"nearest_stop_boundary":b,"cycles_since_stop_boundary":since,"restart_associated_event":0})
    return pd.DataFrame(rows)
@dataclass
class Refs:
    vol: np.ndarray; bdjump: np.ndarray; shapejump: np.ndarray; awr_p95: float; bd_p95: float
def refs(awr: Sequence[float], bd: Sequence[float], shape: Sequence[float], c: AdaptiveAWRV12Config) -> Refs:
    a=np.asarray(awr,float); b=np.asarray(bd,float); s=np.asarray(shape,float); vol=[]; bj=[]; sj=[]
    for k in range(len(a)):
        lo=max(0,k-c.reliability_window+1); vol.append(mad(a[lo:k+1],c.eps)); bj.append(max(b[k]-np.median(b[lo:k+1]),0)); sj.append(max(s[k]-np.median(s[lo:k+1]),0))
    return Refs(np.asarray(vol),np.asarray(bj),np.asarray(sj),float(np.percentile(a,95)),float(np.percentile(b,95)))
@dataclass
class Tracker:
    ref: Refs; source_awr_high: float; source_bd_high: float; c: AdaptiveAWRV12Config; ah:list[float]=field(default_factory=list); bh:list[float]=field(default_factory=list); sh:list[float]=field(default_factory=list); occ:list[int]=field(default_factory=list)
    def step(self, awr:float, bd:float, shape:float)->dict[str,float]:
        self.ah.append(awr);self.bh.append(bd);self.sh.append(shape); w=self.c.reliability_window; lo=max(0,len(self.ah)-w); v=mad(self.ah[lo:],self.c.eps); bj=max(bd-np.median(self.bh[lo:]),0); sj=max(shape-np.median(self.sh[lo:]),0)
        rs=np.nan
        if len(self.ah)>=100: rs=(np.median(self.ah[-50:])-np.median(self.ah[-100:-50]))/50
        tes=.4*zplus(v,self.ref.vol,self.c.eps)+.4*zplus(bj,self.ref.bdjump,self.c.eps)+.2*zplus(sj,self.ref.shapejump,self.c.eps); pair=int(awr>self.source_awr_high and bd>self.source_bd_high); self.occ.append(pair)
        return {"AWR_adaptive":awr,"BDall_xy_v2":bd,"RS50_positive":max(rs,0) if np.isfinite(rs) else 0.,"RS50":rs,"TES":tes,"BD_jump":bj,"high_AWR_high_BD_occupancy":float(np.mean(self.occ[-self.c.occupancy_window:]))}
