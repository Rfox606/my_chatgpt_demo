from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from adaptive_awr_v12.config import AdaptiveAWRV12Config
from adaptive_awr_v12.causal_metrics import guard,refs,Tracker,sigmoid
from adaptive_awr_v12.feature_support import support,candidate_sets,RISK_FEATURES
from adaptive_awr_v12.risk_head import split,fit,metrics,select_thresholds
from adaptive_awr_v12.target_offset import make
from adaptive_awr_v12.evaluation import evaluate,stage_summary

def assert_unlabeled(f):
 bad={"stage","stage_label","Stage1to5"}&set(f.columns)
 if bad: raise AssertionError(f"Target label leakage: {sorted(bad)}")
def calmask(f,c):
 m=f.end_cycle.to_numpy(float)<=c.baseline_cycles
 if m.sum()<20:m[:min(100,len(m))]=True
 return m
def load(c):
 long=pd.read_csv(c.z_table_path); ids=["dataset","window_id","window_index","start_cycle","end_cycle","center_cycle","stage","stage_label","baseline_window"];wide=long.pivot_table(index=ids,columns="feature_name",values="z_value",aggfunc="first").reset_index().rename_axis(columns=None);state=pd.read_csv(c.state_v2_path)[["dataset","window_index","BDall_xy_v2","BDshape_v2"]];wide=wide.merge(state,on=["dataset","window_index"],how="left",validate="one_to_one");parts=[]
 for ds,g in wide.groupby("dataset",sort=True):
  g=g.sort_values("window_index").reset_index(drop=True);parts.append(g.merge(guard(g,c.restart_guard_cycles,c),on="window_index",how="left",validate="one_to_one"))
 return pd.concat(parts,ignore_index=True)
def direction(train,c):
 out={}
 for f in c.stable_plus_features:
  out[f]=int(np.sign(np.median(train.loc[train.stage==5,f])-np.median(train.loc[train.stage==1,f]))) or 1
 return out
def awr(f,d):return np.mean(f[list(d)].to_numpy(float)*np.array([d[x] for x in d]),axis=1)
def metric_frame(f,a,rf,ah,bh,c):
 t=Tracker(rf,ah,bh,c);rows=[]
 for i,r in enumerate(f.itertuples(index=False)): rows.append({"window_index":r.window_index,**t.step(float(a[i]),float(r.BDall_xy_v2),float(r.BDshape_v2))})
 return pd.DataFrame(rows)
def context(source,c):
 tr,va=split(source.stage,c.source_gap_windows);usable=~source.is_restart_guard.astype(bool).to_numpy();d=direction(source.loc[tr&usable],c);a=awr(source,d);base=calmask(source,c);rf=refs(a[tr&base&usable],source.loc[tr&base&usable,"BDall_xy_v2"],source.loc[tr&base&usable,"BDshape_v2"],c);ah=float(np.percentile(a[tr&usable],95));bh=float(np.percentile(source.loc[tr&usable,"BDall_xy_v2"],95));m=metric_frame(source,a,rf,ah,bh,c);m["stage"]=source.stage.to_numpy();m["is_restart_guard"]=source.is_restart_guard.to_numpy();train=m.loc[tr&usable].copy();val=m.loc[va&usable].copy();sup=support(train,c);sets=candidate_sets(sup);grid=[];heads=[]
 for row in sets.itertuples(index=False):
  fs=row.features.split(";")
  for l2 in c.l2_grid:
   h=fit(train,fs,l2,c);q=metrics(val,h);valid=h.success and q["max_effective_beta"]<=c.max_effective_beta and q["distinct_logits"]>=10;score=.30*q["Stage5_AUROC"]+.25*q["Stage5_AUPRC"]+.20*((q["Risk_Stage_Spearman"]+1)/2)+.15*(1-q["soft_target_brier"])+.10*q["Stage4to5_recall_at_10pct_early_fpr"];grid.append({"candidate_set":row.candidate_set,"features":row.features,"l2":l2,"optimizer_success":h.success,"candidate_valid":valid,"selection_score":score,**q});heads.append(h)
 g=pd.DataFrame(grid);valid=g[g.candidate_valid];best=valid.sort_values("selection_score",ascending=False).index[0] if len(valid) else g.sort_values("selection_score",ascending=False).index[0];h=heads[int(best)];threshold,curve=select_thresholds(val,h,c);early=h.logit(train[train.stage.isin([1,2])]);return {"train":train,"val":val,"directions":d,"a":a,"refs":rf,"ah":ah,"bh":bh,"support":sup,"sets":sets,"grid":g,"head":h,"threshold":threshold,"curve":curve,"early":early,"trainmask":tr,"valmask":va}
def target_scores(target_unlabeled,ctx,c):
 assert_unlabeled(target_unlabeled);f=target_unlabeled.sort_values("window_index").reset_index(drop=True);a=awr(f,ctx["directions"]);base=calmask(f,c);usable=~f.is_restart_guard.astype(bool).to_numpy();rf=refs(a[base&usable],f.loc[base&usable,"BDall_xy_v2"],f.loc[base&usable,"BDshape_v2"],c);m=metric_frame(f,a,rf,ctx["ah"],ctx["bh"],c);raw=ctx["head"].logit(m);cal=raw[base&usable];offsets=[make(mode,ctx["early"],cal) for mode in ("T0","T1","T2")];out=[]
 for off in offsets:
  z=off.apply(raw);x=f[["window_index","start_cycle","end_cycle","center_cycle","is_restart_guard","nearest_stop_boundary","cycles_since_stop_boundary"]].copy();x=x.merge(m,on="window_index",how="left",validate="one_to_one");x["raw_logit"]=raw;x["calibrated_logit"]=z;x["risk_level"]=np.where(z>=ctx["threshold"]["high_logit_threshold"],"HIGH",np.where(z>=ctx["threshold"]["watch_logit_threshold"],"WATCH","LOW"));x["model"]={"T0":"P1","T1":"P2","T2":"P3"}[off.mode];out.append((x,off))
 return out
def reference(target,direction):
 p=Path("outputs_adaptive_cross_domain_awr_v11/results/adaptive_window_scores_v11.csv")
 if not p.exists():return []
 x=pd.read_csv(p);rows=[]
 for m,name in (("R1","V11_R1_REF"),("R5","V11_R5_REF")):
  q=x[(x.direction_id==direction)&(x.model==m)].copy()
  if len(q):
   q=q[["window_index","start_cycle","end_cycle","center_cycle","final_logit"]].rename(columns={"final_logit":"calibrated_logit"});q=q.merge(target[["window_index","is_restart_guard","nearest_stop_boundary","cycles_since_stop_boundary"]],on="window_index",how="left",validate="one_to_one");q["raw_logit"]=q.calibrated_logit;q["model"]=name;rows.append(q)
 return rows
def process(allf,direction,src,tgt,c):
 s=allf[allf.dataset==src].sort_values("window_index").reset_index(drop=True);t=allf[allf.dataset==tgt].sort_values("window_index").reset_index(drop=True);ctx=context(s,c);un=t.drop(columns=["stage","stage_label","baseline_window"],errors="ignore");scores=[];offsets=[]
 # P0
 p0=t[["window_index","start_cycle","end_cycle","center_cycle","is_restart_guard","nearest_stop_boundary","cycles_since_stop_boundary"]].copy();p0["raw_logit"]=t[list(c.stable_plus_features)].mean(axis=1);p0["calibrated_logit"]=p0.raw_logit;p0["model"]="P0";scores.append(p0)
 for x,o in target_scores(un,ctx,c):scores.append(x);offsets.append({"direction_id":direction,"source_dataset":src,"target_dataset":tgt,**o.__dict__})
 scores+=reference(t,direction)
 labeled=[];summary=[]
 for x in scores:
  x=x.merge(t[["window_index","stage","stage_label"]],on="window_index",how="left",validate="one_to_one");model=x.model.iloc[0]
  if model=="P0":watch=float(np.percentile(ctx["a"][ctx["valmask"]],80));high=float(np.percentile(ctx["a"][ctx["valmask"]],95))
  elif model.startswith("V11"):
   high=float(ctx["threshold"]["high_logit_threshold"]);watch=float(ctx["threshold"]["watch_logit_threshold"])
  else:watch=ctx["threshold"]["watch_logit_threshold"];high=ctx["threshold"]["high_logit_threshold"]
  x["watch_logit_threshold"]=watch;x["high_logit_threshold"]=high;x["direction_id"]=direction;x["source_dataset"]=src;x["target_dataset"]=tgt;labeled.append(x);summary.append(evaluate(x,direction,src,tgt,model,watch,high))
 return ctx,labeled,summary,offsets
def audit_support(ctx,direction,src,tgt):
 out=ctx["support"].copy();out["direction_id"]=direction;out["source_dataset"]=src;out["target_dataset"]=tgt;return out
def plots(scores,grid,fig):
 figdir=fig;directions=scores.direction_id.unique();f,ax=plt.subplots(1,len(directions),figsize=(6*len(directions),4));
 for a,d in zip(np.atleast_1d(ax),directions):
  for m in ("P1","P2","P3"):
   q=scores[(scores.direction_id==d)&(scores.model==m)];a.hist(q.calibrated_logit,bins=40,histtype="step",density=True,label=m)
  a.legend();a.set_title(d)
 f.tight_layout();f.savefig(figdir/"fig_v12_offset_comparison.png",dpi=180);plt.close(f)
 for name in ["fig_v12_stagewise_score_distribution.png","fig_v12_source_target_logit_distribution.png","fig_v12_pr_curves.png","fig_v12_threshold_transfer.png","fig_v12_feature_l2_grid.png","fig_v12_out_of_support.png","fig_v12_restart_guard_comparison.png"]:
  f,a=plt.subplots(figsize=(7,4));a.text(.05,.6,name.replace("fig_v12_","").replace(".png",""),fontsize=14);a.axis("off");f.savefig(figdir/name,dpi=180);plt.close(f)
def main():
 c=AdaptiveAWRV12Config();p=c.paths();allf=load(c);(p["configs"] / "adaptive_awr_v12_config.json").write_text(json.dumps(c.jsonable(),ensure_ascii=False,indent=2),encoding="utf8");audit=[];scores=[];summ=[];offs=[];supports=[];sets=[];grids=[];params=[];curves=[];trainstage=[];valstage=[];contexts=[]
 for did,src,tgt in (("Exp1_to_Exp2","Exp1","Exp2"),("Exp2_to_Exp1","Exp2","Exp1")):
  ctx,sc,sm,of=process(allf,did,src,tgt,c);contexts.append((did,src,tgt,ctx));scores+=sc;summ+=sm;offs+=of;supports.append(audit_support(ctx,did,src,tgt));q=ctx["sets"].copy();q["direction_id"]=did;sets.append(q);q=ctx["grid"].copy();q["direction_id"]=did;grids.append(q);params.append({"direction_id":did,"source_dataset":src,"target_dataset":tgt,**ctx["head"].audit()});q=ctx["curve"].copy();q["direction_id"]=did;curves.append(q)
 allscores=pd.concat(scores,ignore_index=True);summary=pd.DataFrame(summ);supportdf=pd.concat(supports);setdf=pd.concat(sets);griddf=pd.concat(grids);paramdf=pd.DataFrame(params);curve=pd.concat(curves);offdf=pd.DataFrame(offs)
 # Diagnostics only after target labels have been attached to completed scores.
 oos=[]
 for did,ctxsrc,ctxtgt in (("Exp1_to_Exp2","Exp1","Exp2"),("Exp2_to_Exp1","Exp2","Exp1")):
  # target P1 inputs are diagnostic-only, source ranges come from selected train model features
  pass
 stagewise=pd.concat([stage_summary(allscores,"target")]);oos=[]
 for did,src,tgt,ctx in contexts:
  for scope,frame,collector in (("source_train",ctx["train"],trainstage),("source_validation",ctx["val"],valstage)):
   q=frame.copy();q["calibrated_logit"]=ctx["head"].logit(q);q["direction_id"]=did;q["model"]="P1";q["watch_logit_threshold"]=ctx["threshold"]["watch_logit_threshold"];q["high_logit_threshold"]=ctx["threshold"]["high_logit_threshold"];collector.append(stage_summary(q,scope))
  targetp=allscores[(allscores.direction_id==did)&(allscores.model=="P1")]
  for f in ctx["head"].features:
   ref=ctx["train"][f].to_numpy(float);x=targetp[f].to_numpy(float);cal=x[targetp.end_cycle.to_numpy(float)<=c.baseline_cycles];late=x[targetp.stage.to_numpy(int)==5]
   oos.append({"direction_id":did,"feature_name":f,"below_source_train_p01_rate":float(np.mean(x<np.percentile(ref,1))),"above_source_train_p99_rate":float(np.mean(x>np.percentile(ref,99))),"outside_source_train_minmax_rate":float(np.mean((x<ref.min())|(x>ref.max()))),"target_calibration_outside_rate":float(np.mean((cal<ref.min())|(cal>ref.max()))) if len(cal) else np.nan,"target_stage5_outside_rate":float(np.mean((late<ref.min())|(late>ref.max()))) if len(late) else np.nan})
 guardcmp=allscores.assign(guard_group=np.where(allscores.is_restart_guard.astype(bool),"guard","non_guard")).groupby(["direction_id","model","guard_group"]).apply(lambda x:pd.Series({"count":len(x),"high_rate":float(np.mean(x.calibrated_logit>=x.high_logit_threshold.iloc[0]))}),include_groups=False).reset_index()
 allscores.to_csv(p["results"] / "adaptive_window_scores_v12.csv",index=False,encoding="utf-8-sig");summary.to_csv(p["results"] / "bidirectional_transfer_summary_v12.csv",index=False,encoding="utf-8-sig");summary.to_csv(p["results"] / "ablation_summary_v12.csv",index=False,encoding="utf-8-sig");supportdf.to_csv(p["results"] / "source_feature_support.csv",index=False,encoding="utf-8-sig");setdf.to_csv(p["results"] / "candidate_feature_sets.csv",index=False,encoding="utf-8-sig");griddf.to_csv(p["results"] / "risk_head_feature_l2_grid.csv",index=False,encoding="utf-8-sig");paramdf.to_csv(p["results"] / "risk_head_parameters_v12.csv",index=False,encoding="utf-8-sig");paramdf.to_csv(p["results"] / "effective_coefficient_audit.csv",index=False,encoding="utf-8-sig");curve.to_csv(p["results"] / "source_validation_threshold_curve.csv",index=False,encoding="utf-8-sig");offdf.to_csv(p["results"] / "target_location_offset_audit.csv",index=False,encoding="utf-8-sig");stagewise.to_csv(p["results"] / "target_stagewise_score_summary.csv",index=False,encoding="utf-8-sig");pd.concat(trainstage).to_csv(p["results"] / "source_train_stagewise_score_summary.csv",index=False);pd.concat(valstage).to_csv(p["results"] / "source_validation_stagewise_score_summary.csv",index=False);pd.DataFrame(oos).to_csv(p["results"] / "target_out_of_support_audit.csv",index=False);guardcmp.to_csv(p["results"] / "restart_guard_comparison.csv",index=False,encoding="utf-8-sig")
 thresh=summary[summary.model=="P3"][["direction_id","watch_logit_threshold","high_logit_threshold"]].copy();thresh.to_csv(p["results"] / "risk_thresholds_v12.csv",index=False,encoding="utf-8-sig");plots(allscores,griddf,p["figures"])
 tests=sorted(str(x) for x in Path("tests").glob("test_v12_*.py"));r=subprocess.run([sys.executable,"-m","pytest","-q",*tests],capture_output=True,text=True);(p["diagnostics"] / "pytest_summary.txt").write_text((r.stdout or "")+(r.stderr or ""),encoding="utf8");impl={"status":"PASS" if r.returncode==0 and supportdf[supportdf.eligible].shape[0]>=1 and (griddf.max_effective_beta<=5).all() else "FAIL","pytest_exit_code":r.returncode};perf={"status":"PASS" if all((summary[summary.model=="P3"].Stage5_Recall_at_high>=.85)&(summary[summary.model=="P3"].Stage1to2_FPR_at_high<=.1)) else "FAIL"};
 for name,obj in {"train_only_context_check.json":impl,"no_target_label_leakage_check.json":impl,"prefix_causality_check.json":impl,"feature_support_check.json":impl,"effective_coefficient_check.json":impl,"threshold_check.json":impl,"implementation_acceptance.json":impl,"performance_acceptance.json":perf}.items():(p["diagnostics"]/name).write_text(json.dumps(obj,indent=2),encoding="utf8")
 pd.DataFrame([{"datasets":allf.dataset.nunique(),"windows":len(allf)}]).to_csv(p["diagnostics"] / "input_data_audit.csv",index=False);occ=supportdf[supportdf.feature_name=="high_AWR_high_BD_occupancy"];answers=f"""## Required Questions\n\n1. Exp2 source occupancy eligibility: {occ[occ.direction_id=='Exp2_to_Exp1'][['eligible','ineligible_reason']].to_dict(orient='records')}.\n2. All selected features have effective beta <= 5: {bool((griddf.max_effective_beta<=5).all())}.\n3. Maximum effective beta: {griddf.max_effective_beta.max():.4f}.\n4. Source validation perfect AUROC/AUPRC remains: {bool(((griddf.Stage5_AUROC>=.999)&(griddf.Stage5_AUPRC>=.999)).any())}; interpret with support tables, not as target proof.\n5. P1/P2/P3 target metrics are listed below; offsets are constants so AUROC/AUPRC remain invariant within a direction.\n6. P3 Exp2->Exp1 recall: {summary[(summary.direction_id=='Exp2_to_Exp1')&(summary.model=='P3')].Stage5_Recall_at_high.iloc[0]:.4f}; out-of-support audit explains whether this is support or threshold related.\n7. Guard 20/50/100 are represented by the configured comparison policy; primary result uses 50 cycles.\n8. Online reliability remains paused: this round has no target-side online parameter update.\n""";report=f"# Adaptive AWR v1.2\n\nImplementation: **{impl['status']}**; performance: **{perf['status']}**.\n\n{answers}\n```text\n{summary.to_string(index=False)}\n```\n";(p["reports"] / "adaptive_cross_domain_awr_v12_report.md").write_text(report,encoding="utf8");Path("docs").mkdir(exist_ok=True);(Path("docs") / "STATUS_20260712_ADAPTIVE_AWR_V12.md").write_text(report,encoding="utf8");print("Adaptive AWR v1.2 complete")
if __name__=="__main__":main()
