from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .config import AdaptiveAWRV12Config, SOFT_TARGET, SAMPLE_WEIGHT
from .causal_metrics import iqr, sigmoid

def split(stages, gap):
    s=np.asarray(stages,int); train=np.zeros(len(s),bool); val=np.zeros(len(s),bool)
    for stage in np.unique(s):
        index=np.flatnonzero(s==stage); cut=int(.7*len(index)); train[index[:max(0,cut-gap)]]=True; val[index[min(len(index),cut+gap):]]=True
    return train,val
def auc(y,x):
    y=np.asarray(y,int);x=np.asarray(x,float);p=y.sum();n=len(y)-p;return float((pd.Series(x).rank().to_numpy()[y==1].sum()-p*(p+1)/2)/(p*n)) if p and n else np.nan
def ap(y,x):
    o=np.asarray(y,int)[np.argsort(-np.asarray(x,float))];return float((np.cumsum(o)/(np.arange(len(o))+1))[o==1].mean()) if o.sum() else np.nan
def spear(x,s): return float(pd.Series(x).rank().corr(pd.Series(s).rank(),method="pearson"))
@dataclass
class Head:
    features:list[str]; median:np.ndarray; width:np.ndarray; beta:np.ndarray; l2:float; success:bool; objective:float
    def logit(self,frame):
        X=frame[self.features].to_numpy(float);X=np.where(np.isfinite(X),X,self.median);return self.beta[0]+((X-self.median)/self.width)@self.beta[1:]
    def audit(self): return {"features":";".join(self.features),"l2":self.l2,"beta0":self.beta[0],**{f"beta_{f}":b for f,b in zip(self.features,self.beta[1:])},**{f"effective_beta_{f}":b/w for f,b,w in zip(self.features,self.beta[1:],self.width)}}
def fit(train,features,l2,c):
    med=np.array([np.median(train[f]) for f in features]);width=np.array([iqr(train[f],c.eps) for f in features]);X=(train[features].to_numpy(float)-med)/width;y=train.stage.map(SOFT_TARGET).to_numpy(float);w=train.stage.map(SAMPLE_WEIGHT).to_numpy(float);ws=w.sum();upper=np.minimum(c.max_standardized_beta,c.max_effective_beta*width)
    def fun(theta):
        p=sigmoid(theta[0]+X@theta[1:]);b=-(y*np.log(p+c.eps)+(1-y)*np.log(1-p+c.eps));loss=(w*b).sum()/ws+l2*np.sum(theta[1:]**2);r=w*(p-y)/ws;return loss,np.r_[r.sum(),X.T@r+2*l2*theta[1:]]
    result=minimize(lambda t:fun(t)[0],np.zeros(len(features)+1),jac=lambda t:fun(t)[1],method="L-BFGS-B",bounds=[c.beta0_bounds]+[(0,float(v)) for v in upper]);return Head(features,med,width,result.x,l2,bool(result.success),float(result.fun))
def metrics(val,h):
    z=h.logit(val);s=val.stage.to_numpy(int);soft=val.stage.map(SOFT_TARGET).to_numpy(float);early=np.isin(s,[1,2]);q=np.percentile(z[early],90) if early.any() else np.inf;sp=spear(z,s);br=float(np.mean((sigmoid(z)-soft)**2));return {"Stage5_AUROC":auc(s==5,z),"Stage5_AUPRC":ap(s==5,z),"Risk_Stage_Spearman":sp,"soft_target_brier":br,"Stage4to5_recall_at_10pct_early_fpr":float(np.mean(z[s>=4]>=q)),"max_effective_beta":float(np.max(h.beta[1:]/h.width)),"distinct_logits":len(np.unique(np.round(z,12)))}
def select_thresholds(val,h,c):
    z=h.logit(val);s=val.stage.to_numpy(int); grid=np.unique(np.quantile(z,np.linspace(.5,.995,c.threshold_quantiles)));rows=[]
    for t in grid:
        high=z>=t;early=np.isin(s,[1,2]);rows.append({"logit_threshold":float(t),"Stage5_recall":float(np.mean(high[s==5])),"Stage5_precision":float(np.mean(s[high]==5)) if high.any() else 0.,"Stage45_recall":float(np.mean(high[s>=4])),"Stage1to2_fpr":float(np.mean(high[early]))})
    curve=pd.DataFrame(rows); feasible=curve[(curve.Stage5_recall>=.85)&(curve.Stage1to2_fpr<=.1)]; high=feasible.sort_values(["Stage1to2_fpr","Stage5_precision","logit_threshold"],ascending=[True,False,False]).iloc[0] if len(feasible) else curve.assign(score=.55*curve.Stage5_recall+.25*curve.Stage5_precision-.4*curve.Stage1to2_fpr).sort_values("score",ascending=False).iloc[0]; watch=curve[(curve.Stage45_recall>=.8)&(curve.Stage1to2_fpr<=.2)&(curve.logit_threshold<high.logit_threshold)].sort_values(["Stage1to2_fpr","Stage45_recall"],ascending=[True,False]); watch=watch.iloc[0] if len(watch) else curve[curve.logit_threshold<high.logit_threshold].iloc[0]
    return {"watch_logit_threshold":float(watch.logit_threshold),"high_logit_threshold":float(high.logit_threshold),"watch_probability":float(sigmoid(watch.logit_threshold)),"high_probability":float(sigmoid(high.logit_threshold)),"high_Stage5_recall":float(high.Stage5_recall),"high_Stage1to2_fpr":float(high.Stage1to2_fpr)},curve
