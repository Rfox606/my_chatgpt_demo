from __future__ import annotations
import numpy as np
import pandas as pd
from ordered_regime_v21.config import OrderedRegimeConfig
from ordered_regime_v21.discovery import DiscoveryParameters, OrderedRegimeDiscovery
from ordered_regime_v21.source_bundle import SourceArtifacts
def config(**changes):
    base=OrderedRegimeConfig(candidate_gap_tolerance=1,prototype_recompute_every=2,memory_per_state=4,post_restart_cooldown_windows=0)
    return base.__class__(**{**base.__dict__,**changes})
def artifact(cfg=None):
    level=np.zeros((5,16));level[:,0]=np.arange(5)*3;traj=np.zeros((5,48));traj[:,0]=np.arange(5)*3
    return SourceArtifacts("Synthetic",None,None,level,traj,np.ones(5),np.ones(5),[np.array([.05,.1,.15]) for _ in range(5)],[np.array([.05,.1,.15]) for _ in range(5)],99.,np.eye(16)[0],1.,1.,{})
def stream(levels,posterior_stage=1,guards=None):
    n=len(levels);guards=np.zeros(n,dtype=int) if guards is None else np.asarray(guards,dtype=int);frame=pd.DataFrame({"window_index":np.arange(n),"center_cycle":np.arange(n)*5+10,"TES":np.zeros(n),"is_restart_guard":guards,"restart_mask":np.zeros(n,dtype=int)})
    emb=np.zeros((n,16));emb[:,0]=levels;traj=np.zeros((n,48));traj[:,0]=levels;p=np.full((n,5),.025);p[:,posterior_stage-1]=.9
    return frame,{"embedding":emb,"trajectory":traj,"stage_probs":p,"health":np.zeros(n),"missing_fraction":np.zeros(n),"health_axis_projection":emb[:,0],"health_axis_velocity20":np.zeros(n),"health_axis_velocity100":np.zeros(n),"trajectory_norm20":np.zeros(n),"trajectory_norm100":np.zeros(n)}
def engine(cfg=None,params=None):
    cfg=cfg or config();return OrderedRegimeDiscovery(artifact(cfg),cfg,params or DiscoveryParameters(2.,5,0),True,True)
