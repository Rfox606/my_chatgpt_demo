from __future__ import annotations
import numpy as np,pandas as pd
from .risk_head import auc,ap,spear
from .config import SOFT_TARGET
def stable_cycle(frame,high,n=10):
 run=0
 for r,on in zip(frame.itertuples(index=False),high):
  run=run+1 if on else 0
  if run>=n:return float(r.center_cycle)
 return np.nan
def evaluate(frame,direction,source,target,model,watch,high):
 f=frame.sort_values("window_index").reset_index(drop=True);s=f.stage.to_numpy(int);z=f.calibrated_logit.to_numpy(float);h=z>=high;w=z>=watch;early=np.isin(s,[1,2]);stage5=s==5;stage45=s>=4; firsth=float(f.loc[h,"center_cycle"].iloc[0]) if h.any() else np.nan;firstw=float(f.loc[w,"center_cycle"].iloc[0]) if w.any() else np.nan;stage5cycle=float(f.loc[stage5,"center_cycle"].iloc[0]) if stage5.any() else np.nan;valid=float(np.mean(h[early]))<=.1 if early.any() else False; lead=stage5cycle-stable_cycle(f,h) if valid and np.isfinite(stable_cycle(f,h)) else np.nan; status="VALID" if valid else "INVALID_DUE_TO_EARLY_FALSE_ALARM";guard=f.is_restart_guard.astype(bool).to_numpy();soft=s.astype(float);soft=np.array([SOFT_TARGET[int(x)] for x in s]);p=1/(1+np.exp(-np.clip(z,-35,35)));episodes=int(np.sum(h & np.r_[True,~h[:-1]]));span=max(float(f.end_cycle.max()-f.start_cycle.min()+1),1)
 return {"direction_id":direction,"source_dataset":source,"target_dataset":target,"model":model,"watch_logit_threshold":watch,"high_logit_threshold":high,"Stage5_AUROC":auc(stage5,z),"Stage5_AUPRC":ap(stage5,z),"Stage5_Recall_at_high":float(np.mean(h[stage5])),"Stage4to5_Recall_at_watch":float(np.mean(w[stage45])),"Stage1to2_FPR_at_high":float(np.mean(h[early])),"Stage1to2_FPR_at_watch":float(np.mean(w[early])),"Recall_at_10pct_Stage1to2_FPR":float(np.mean(z[stage5]>=np.percentile(z[early],90))),"Risk_Stage_Spearman":spear(z,s),"soft_target_brier":float(np.mean((p-soft)**2)),"first_WATCH_cycle":firstw,"first_HIGH_cycle":firsth,"stable_HIGH_cycle":stable_cycle(f,h),"lead_cycles_relative_to_Stage5":lead,"lead_status":status,"false_HIGH_episodes_per_1000_cycles":episodes*1000/span,"guard_HIGH_FPR":float(np.mean(h[guard&early])) if np.any(guard&early) else np.nan,"non_guard_HIGH_FPR":float(np.mean(h[(~guard)&early])) if np.any((~guard)&early) else np.nan,"watch_occupancy":float(np.mean(w)),"high_occupancy":float(np.mean(h))}
def stage_summary(frame,scope):
 rows=[]
 for keys,g in frame.groupby(["direction_id","model","stage"],dropna=False):
  z=g.calibrated_logit.to_numpy(float);h=z>=g.high_logit_threshold.iloc[0];w=z>=g.watch_logit_threshold.iloc[0];rows.append({"scope":scope,"direction_id":keys[0],"model":keys[1],"stage":keys[2],"count":len(g),"median_logit":float(np.median(z)),"IQR_logit":float(np.percentile(z,75)-np.percentile(z,25)),"p05":float(np.percentile(z,5)),"p25":float(np.percentile(z,25)),"p75":float(np.percentile(z,75)),"p95":float(np.percentile(z,95)),"median_probability":float(np.median(1/(1+np.exp(-np.clip(z,-35,35))))),"high_rate":float(np.mean(h)),"watch_rate":float(np.mean(w))})
 return pd.DataFrame(rows)
