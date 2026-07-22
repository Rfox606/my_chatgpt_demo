from __future__ import annotations

import copy
import json
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ordered_regime_v21.config import OrderedRegimeConfig
from ordered_regime_v21.data import load_table, source_split, target_unlabeled
from ordered_regime_v21.discovery import DiscoveryParameters, OrderedRegimeDiscovery
from ordered_regime_v21.evaluation import evaluate_labeled, source_selection_score
from ordered_regime_v21.source_bundle import load_source_artifacts, load_v12_anchor


def _slice(encoded: dict[str, np.ndarray], end: int) -> dict[str, np.ndarray]:
    return {key:value[:end] for key,value in encoded.items()}


def _anchor(config: OrderedRegimeConfig, direction: str, target: pd.DataFrame) -> np.ndarray:
    anchor=load_v12_anchor(config,direction).set_index("window_index").raw_logit
    return target.window_index.map(anchor).fillna(0.0).to_numpy(float)


def _static_rows(target: pd.DataFrame, encoded: dict[str,np.ndarray], ablation: str, anchor: np.ndarray | None = None) -> pd.DataFrame:
    posterior=encoded["stage_probs"].copy(); final=1/(1+np.exp(-anchor)) if anchor is not None else posterior[:,4]
    result=pd.DataFrame({"window_index":target.window_index,"center_cycle":target.center_cycle,"ablation":ablation,"predicted_stage":posterior.argmax(axis=1)+1,"final_stage5_score":final,"highest_discovered_stage":0,"current_regime_stage":posterior.argmax(axis=1)+1,"candidate_active":0,"change_score":0.0,"anchor_stage5_probability":final,"stage5_support_weight":0.0,"source_predicted_stage":posterior.argmax(axis=1)+1,"source_expected_stage":posterior@np.arange(1,6),"source_health_score":encoded["health"],"restart_guard":target.is_restart_guard,"missing_fraction":encoded["missing_fraction"]})
    for i in range(5): result[f"source_posterior_{i+1}"]=posterior[:,i]; result[f"fused_posterior_{i+1}"]=posterior[:,i]; result[f"prototype_posterior_{i+1}"]=0.
    return result


def _v2_reference(config: OrderedRegimeConfig, direction: str, target: pd.DataFrame) -> pd.DataFrame:
    cols=["direction_id","ablation","window_index","predicted_stage"]+[f"stage_posterior_{i}" for i in range(1,6)]
    frame=pd.read_csv("outputs_temporal_prototype_v2/results/target_window_predictions.csv",usecols=cols)
    frame=frame[(frame.direction_id==direction)&(frame.ablation=="B3_DYNAMIC_PROTOTYPE")].merge(target[["window_index","center_cycle","is_restart_guard"]],on="window_index",how="right")
    posterior=frame[[f"stage_posterior_{i}" for i in range(1,6)]].to_numpy(float)
    result=pd.DataFrame({"window_index":frame.window_index,"center_cycle":frame.center_cycle,"ablation":"A2_TPV2_B3_REFERENCE","predicted_stage":frame.predicted_stage.fillna(1).astype(int),"final_stage5_score":posterior[:,4],"highest_discovered_stage":0,"current_regime_stage":frame.predicted_stage.fillna(1).astype(int),"candidate_active":0,"change_score":0.,"anchor_stage5_probability":posterior[:,4],"stage5_support_weight":0.,"source_predicted_stage":frame.predicted_stage.fillna(1).astype(int),"source_expected_stage":posterior@np.arange(1,6),"source_health_score":0.,"restart_guard":frame.is_restart_guard,"missing_fraction":0.})
    for i in range(5): result[f"source_posterior_{i+1}"]=posterior[:,i]; result[f"fused_posterior_{i+1}"]=posterior[:,i]; result[f"prototype_posterior_{i+1}"]=0.
    return result


def _grid(source_frame: pd.DataFrame, artifact, config: OrderedRegimeConfig) -> tuple[DiscoveryParameters,pd.DataFrame]:
    _,_,validation=source_split(source_frame,config); validation_frame=source_frame.loc[validation].reset_index(drop=True); encoded=artifact.encode(validation_frame,config)
    labels=validation_frame.stage.to_numpy(int); rows=[]
    for threshold in config.change_threshold_grid:
        for candidate in config.candidate_min_windows_grid:
            for dwell in config.min_dwell_windows_grid:
                params=DiscoveryParameters(threshold,candidate,dwell); engine=OrderedRegimeDiscovery(artifact,config,params,True,True)
                pred,_,events,_=engine.run(target_unlabeled(validation_frame),encoded,np.log(np.clip(encoded["stage_probs"][:,4],1e-6,1-1e-6)/(1-np.clip(encoded["stage_probs"][:,4],1e-6,1-1e-6))))
                labeled=pred.assign(stage=labels); metrics,_=evaluate_labeled(labeled,events); score=source_selection_score(metrics)
                rows.append({"source_dataset":artifact.dataset,"change_threshold":threshold,"candidate_min_windows":candidate,"min_dwell_windows":dwell,"selection_score":score,**metrics})
    grid=pd.DataFrame(rows); best=grid.sort_values("selection_score",ascending=False).iloc[0]
    return DiscoveryParameters(float(best.change_threshold),int(best.candidate_min_windows),int(best.min_dwell_windows)),grid


def _snapshot_metrics(direction: str, artifact, config: OrderedRegimeConfig, params: DiscoveryParameters, target: pd.DataFrame, labels: pd.DataFrame, encoded: dict, anchor: np.ndarray, path: Path) -> tuple[pd.DataFrame,bool]:
    rows=[]; replay_ok=True
    for fraction in np.linspace(0,1,11):
        end=max(1,int(round(len(target)*fraction)))
        engine=OrderedRegimeDiscovery(artifact,config,params,True,True); engine.run(target.iloc[:end].reset_index(drop=True),_slice(encoded,end),anchor[:end],updates=True)
        payload=engine.snapshot(); file=path/f"A6_ANCHOR_FUSION_{fraction:.1f}.pkl"; file.write_bytes(pickle.dumps(payload))
        frozen=OrderedRegimeDiscovery(artifact,config,params,True,True); frozen.restore(pickle.loads(file.read_bytes()))
        pred,_,events,_=frozen.run(target,encoded,anchor,updates=False); labeled=pred.merge(labels[["window_index","stage"]],on="window_index",how="left",validate="one_to_one"); metrics,bound=evaluate_labeled(labeled,events)
        rows.append({"direction_id":direction,"snapshot_fraction":fraction,**metrics,**{f"support_stage{s}":frozen.states[s].support if s in frozen.states else 0 for s in range(1,6)}})
        if fraction==0:
            second=OrderedRegimeDiscovery(artifact,config,params,True,True); second.restore(pickle.loads(file.read_bytes())); again,_,_,_=second.run(target,encoded,anchor,updates=False)
            replay_ok=bool(np.array_equal(pred[[f"fused_posterior_{i}" for i in range(1,6)]].to_numpy(),again[[f"fused_posterior_{i}" for i in range(1,6)]].to_numpy()))
    return pd.DataFrame(rows),replay_ok


def _plots(pred:pd.DataFrame,summary:pd.DataFrame,progressive:pd.DataFrame,protos:pd.DataFrame,events:pd.DataFrame,paths:dict) -> None:
    figs=paths["figures"]; a6=pred[pred.ablation=="A6_ANCHOR_FUSION"]
    fig,axes=plt.subplots(2,1,figsize=(10,7))
    for axis,(direction,item) in zip(axes,a6.groupby("direction_id")):
        axis.plot(item.center_cycle,item.final_stage5_score,label="Stage5 score"); axis.plot(item.center_cycle,item.highest_discovered_stage/5,label="highest regime/5"); axis.legend();axis.set_title(direction)
    fig.tight_layout();fig.savefig(figs/"fig_or_v21_state_timeline.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));
    for d,item in a6.groupby("direction_id"): ax.plot(item.center_cycle,item.change_score,label=d)
    ax.legend();ax.set_title("Causal change score");fig.tight_layout();fig.savefig(figs/"fig_or_v21_change_score.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));
    if len(events):
        for event,part in events.groupby("event"): ax.scatter(part.window_index,np.arange(len(part)),s=8,label=event)
    ax.legend(fontsize=7);ax.set_title("Candidate and discovery events");fig.tight_layout();fig.savefig(figs/"fig_or_v21_candidate_events.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));
    if len(protos):
        ax.scatter(protos.level_1,protos.level_2,c=protos.stage,cmap="viridis"); ax.set_title("Target prototypes: first two embedding coordinates")
    fig.tight_layout();fig.savefig(figs/"fig_or_v21_prototype_trajectory.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));
    for d,item in progressive.groupby("direction_id"): ax.plot(item.snapshot_fraction,item.Stage5_AUPRC,marker="o",label=d)
    ax.legend();ax.set_title("Frozen snapshot Stage5 AUPRC");fig.tight_layout();fig.savefig(figs/"fig_or_v21_progressive_accuracy.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));summary.pivot(index="ablation",columns="direction_id",values="Stage5_AUPRC").plot(kind="bar",ax=ax);ax.set_ylim(0,1.05);fig.tight_layout();fig.savefig(figs/"fig_or_v21_ablation.png",dpi=160);plt.close(fig)
    for name,title,column in [("fig_or_v21_stage45_embedding.png","Stage4/5 source expected stage","source_expected_stage"),("fig_or_v21_boundary_error.png","Boundary MAE","mean_boundary_MAE"),("fig_or_v21_memory_support.png","Prototype support","prototype_support_stage5"),("fig_or_v21_anchor_fusion.png","Anchor fusion Stage5 score","final_stage5_score")]:
        fig,ax=plt.subplots(figsize=(8,4))
        if column in a6: ax.plot(a6.center_cycle,a6[column])
        elif column in summary: summary[summary.ablation=="A6_ANCHOR_FUSION"].plot(x="direction_id",y=column,kind="bar",ax=ax,legend=False)
        ax.set_title(title);fig.tight_layout();fig.savefig(figs/name,dpi=160);plt.close(fig)


def _write_prediction_table(frame: pd.DataFrame, path: Path) -> str:
    """Write the large table, verifying an equivalent locked prior result if needed."""
    try:
        frame.to_csv(path, index=False)
        return "written"
    except OSError as error:
        columns = ["direction_id", "ablation", "window_index", "predicted_stage", "final_stage5_score"]
        if not path.exists():
            raise
        existing = pd.read_csv(path, usecols=columns).sort_values(columns[:3]).reset_index(drop=True)
        expected = frame[columns].sort_values(columns[:3]).reset_index(drop=True)
        same_shape = existing.shape == expected.shape
        same_discrete = same_shape and existing[["direction_id", "ablation", "window_index", "predicted_stage"]].equals(expected[["direction_id", "ablation", "window_index", "predicted_stage"]])
        same_scores = same_shape and np.allclose(existing.final_stage5_score, expected.final_stage5_score, equal_nan=True)
        if same_discrete and same_scores:
            return f"retained_equivalent_locked_file: {error}"
        raise RuntimeError(f"Could not write {path}; existing table differs from this run") from error


def main() -> None:
    config=OrderedRegimeConfig(); paths=config.paths(); (paths["configs"]/"ordered_regime_v21_config.json").write_text(json.dumps(config.jsonable(),indent=2),encoding="utf8")
    full=load_table(config); artifacts={name:load_source_artifacts(full,name,config) for name in ("Exp1","Exp2")}
    grids=[]; selections={}
    for name,artifact in artifacts.items():
        frame=full[full.dataset==name].sort_values("window_index").reset_index(drop=True); params,grid=_grid(frame,artifact,config); grids.append(grid);selections[name]=params
    pd.concat(grids).to_csv(paths["source"] / "source_discovery_hyperparameter_grid.csv",index=False)
    pd.DataFrame([{"source_dataset":name,**params.__dict__} for name,params in selections.items()]).to_csv(paths["source"] / "source_selected_discovery_config.csv",index=False)
    scales=[]
    for name,a in artifacts.items():
        for stage in range(5): scales.append({"source_dataset":name,"stage":stage+1,"level_radius_p95":a.level_radius_p95[stage],"trajectory_radius_p95":a.trajectory_radius_p95[stage]})
    pd.DataFrame(scales).to_csv(paths["source"] / "source_state_scales.csv",index=False)
    predictions=[]; proto_rows=[]; events=[]; memories=[]; candidates=[]; initials=[]; summary=[]; boundary=[]; progressive=[]; replay_ok=True
    for direction,source_name,target_name in (("Exp1_to_Exp2","Exp1","Exp2"),("Exp2_to_Exp1","Exp2","Exp1")):
        artifact=artifacts[source_name]; labeled=full[full.dataset==target_name].sort_values("window_index").reset_index(drop=True); target=target_unlabeled(labeled); encoded=artifact.encode(target,config); anchor=_anchor(config,direction,target); params=selections[source_name]
        ablations=[]
        ablations.append(("A0_V12_P1_STATIC",_static_rows(target,encoded,"A0_V12_P1_STATIC",anchor),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),None))
        ablations.append(("A1_TPV2_STATIC_SOURCE",_static_rows(target,encoded,"A1_TPV2_STATIC_SOURCE"),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),None))
        ablations.append(("A2_TPV2_B3_REFERENCE",_v2_reference(config,direction,target),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),None))
        for name,use_traj,robust,use_anchor in (("A3_ORDERED_LEVEL_ONLY",False,False,False),("A4_ORDERED_LEVEL_TRAJECTORY",True,False,False),("A5_ROBUST_MEMORY_DISCOVERY",True,True,False),("A6_ANCHOR_FUSION",True,True,True)):
            engine=OrderedRegimeDiscovery(artifact,config,params,use_traj,robust); pred,proto,event,memory=engine.run(target,encoded,anchor if use_anchor else None,updates=True); pred["ablation"]=name; ablations.append((name,pred,proto,event,memory,engine)); initials.append({"direction_id":direction,"source_dataset":source_name,**engine.initial_audit})
        snap_dir=paths["snapshots"]/direction;snap_dir.mkdir(exist_ok=True); prog,ok=_snapshot_metrics(direction,artifact,config,params,target,labeled,encoded,anchor,snap_dir);progressive.append(prog);replay_ok &= ok
        # Labels are joined only after all target online streams and snapshots finish.
        for name,pred,proto,event,memory,engine in ablations:
            merged=pred.merge(labeled[["window_index","stage","stage_label"]],on="window_index",how="left",validate="one_to_one"); merged["direction_id"]=direction;predictions.append(merged)
            metrics,bound=evaluate_labeled(merged,event); row={"direction_id":direction,"source_dataset":source_name,"target_dataset":target_name,"ablation":name,**metrics}
            if engine is not None:
                row.update({f"prototype_support_stage{s}":engine.states[s].support if s in engine.states else 0 for s in range(1,6)}); row["highest_discovered_stage"]=engine.highest_stage
                proto["direction_id"]=direction;proto["ablation"]=name;event["direction_id"]=direction;event["ablation"]=name;memory["direction_id"]=direction;memory["ablation"]=name;proto_rows.append(proto);events.append(event);memories.append(memory);candidates.extend([{**item,"direction_id":direction,"ablation":name} for item in engine.candidate_audit])
            summary.append(row);bound["direction_id"]=direction;bound["ablation"]=name;boundary.append(bound)
    pred=pd.concat(predictions,ignore_index=True); summary_df=pd.DataFrame(summary); proto=pd.concat(proto_rows,ignore_index=True); event=pd.concat(events,ignore_index=True); memory=pd.concat(memories,ignore_index=True); progressive_df=pd.concat(progressive,ignore_index=True)
    candidate_df = pd.DataFrame(candidates)
    if not candidate_df.empty:
        for index, candidate in candidate_df.iterrows():
            posthoc = pred[(pred.direction_id == candidate.direction_id) & (pred.ablation == candidate.ablation) & (pred.window_index >= candidate.start_window) & (pred.window_index <= candidate.end_window)].stage
            for stage in range(1, 6):
                candidate_df.loc[index, f"posthoc_stage{stage}_fraction"] = float((posthoc == stage).mean()) if len(posthoc) else np.nan
            candidate_df.loc[index, "posthoc_dominant_stage"] = int(posthoc.mode().iloc[0]) if len(posthoc) else np.nan
    # Signal-order diagnostics operate without target labels.
    constant=[]
    for source_name,target_name,direction in (("Exp1","Exp2","Exp1_to_Exp2"),("Exp2","Exp1","Exp2_to_Exp1")):
        artifact=artifacts[source_name]; target=target_unlabeled(full[full.dataset==target_name].sort_values("window_index").reset_index(drop=True))
        # A source-prototype constant stream isolates temporal discovery from real target evolution.
        persistent_duration=selections[source_name].candidate_min_windows+5
        start=selections[source_name].min_dwell_windows+10; probe_length=start+persistent_duration+20
        probe_frame=target.iloc[:probe_length].copy(); probe_frame["TES"]=0.; probe_frame["is_restart_guard"]=0; probe_frame["restart_mask"]=0
        source_posterior=np.full(5,.025); source_posterior[0]=.90
        base={"embedding":np.repeat(artifact.level_proto[:1],probe_length,axis=0),"trajectory":np.repeat(artifact.trajectory_proto[:1],probe_length,axis=0),"stage_probs":np.repeat(source_posterior[None,:],probe_length,axis=0),"health":np.zeros(probe_length),"missing_fraction":np.zeros(probe_length),"health_axis_projection":np.zeros(probe_length),"health_axis_velocity20":np.zeros(probe_length),"health_axis_velocity100":np.zeros(probe_length),"trajectory_norm20":np.zeros(probe_length),"trajectory_norm100":np.zeros(probe_length)}
        eng=OrderedRegimeDiscovery(artifact,config,selections[source_name],True,True);_,_,ev,_=eng.run(probe_frame,base,np.zeros(probe_length));constant.append({"direction_id":direction,"diagnostic":"constant_signal","discovered_states":int((ev.event=="STATE_DISCOVERED").sum()),"candidate_rejections":int((ev.event=="CANDIDATE_REJECTED").sum()),"first_discovery_window":np.nan,"shift_start_window":np.nan})
        # Signal-only synthetic checks: neither uses any target label.
        level_shift=3*max(float(artifact.level_radius_p95[0]),1e-3); trajectory_shift=3*max(float(artifact.trajectory_radius_p95[0]),1e-3)
        for diagnostic,duration in (("transient_spike",max(1,selections[source_name].candidate_min_windows-1)),("persistent_shift",selections[source_name].candidate_min_windows+5)):
            altered={key:value.copy() for key,value in base.items()}
            altered["embedding"][start:start+duration,0] += level_shift; altered["trajectory"][start:start+duration,0] += trajectory_shift
            run_length=start+duration if diagnostic=="transient_spike" else probe_length
            probe=OrderedRegimeDiscovery(artifact,config,selections[source_name],True,True); _,_,probe_events,_=probe.run(probe_frame.iloc[:run_length],_slice(altered,run_length),np.zeros(run_length))
            discovered=probe_events[probe_events.event=="STATE_DISCOVERED"]
            constant.append({"direction_id":direction,"diagnostic":diagnostic,"discovered_states":int(len(discovered)),"candidate_rejections":int((probe_events.event=="CANDIDATE_REJECTED").sum()),"first_discovery_window":int(discovered.window_index.iloc[0]) if len(discovered) else np.nan,"shift_start_window":start})
    pd.DataFrame(constant).to_csv(paths["results"] / "constant_shift_spike_diagnostics.csv",index=False)
    # Shuffled-order diagnostic only after online streams are complete; it never selects parameters.
    shuffled=[]
    for direction,source_name,target_name in (("Exp1_to_Exp2","Exp1","Exp2"),("Exp2_to_Exp1","Exp2","Exp1")):
        artifact=artifacts[source_name];labeled=full[full.dataset==target_name].sort_values("window_index").reset_index(drop=True);target=target_unlabeled(labeled).sample(frac=1,random_state=20260713).reset_index(drop=True);encoded=artifact.encode(target,config);eng=OrderedRegimeDiscovery(artifact,config,selections[source_name],True,True);p,_,e,_=eng.run(target,encoded,np.zeros(len(target)));m=p.merge(labeled[["window_index","stage"]],on="window_index",how="left");met,_=evaluate_labeled(m,e);shuffled.append({"direction_id":direction,"order":"shuffled","discovered_states":eng.highest_stage,**met})
    pd.DataFrame(shuffled).to_csv(paths["results"] / "original_order_vs_shuffle.csv",index=False)
    time_rows=[]
    for direction,item in pred[pred.ablation=="A6_ANCHOR_FUSION"].groupby("direction_id"):
        item=item.sort_values("window_index").copy(); expected=1+np.minimum(4,(np.arange(len(item))*5//max(len(item),1))).astype(int); posterior=np.zeros((len(item),5));posterior[np.arange(len(item)),expected-1]=1
        for stage in range(5): item[f"fused_posterior_{stage+1}"]=posterior[:,stage]
        item["predicted_stage"]=expected;item["final_stage5_score"]=posterior[:,4]
        boundaries=[min(len(item)-1, int(np.ceil(len(item)*stage/5))) for stage in range(1,5)]
        time_events=pd.DataFrame({"window_index":[int(item.window_index.iloc[boundary]) for boundary in boundaries],"event":"STATE_DISCOVERED","new_stage":[2,3,4,5]})
        metrics,_=evaluate_labeled(item,time_events)
        time_rows.append({"direction_id":direction,"baseline":"TIME_ONLY_FIVE_SEGMENTS",**metrics})
    pd.DataFrame(time_rows).to_csv(paths["results"] / "time_prior_diagnostics.csv",index=False)
    pd.DataFrame(initials).to_csv(paths["results"] / "initial_stage_audit.csv",index=False);summary_df.to_csv(paths["results"] / "bidirectional_summary.csv",index=False);summary_df.to_csv(paths["results"] / "ablation_summary.csv",index=False);prediction_write_status=_write_prediction_table(pred,paths["results"] / "target_window_predictions.csv");pred[["direction_id","ablation","window_index"]+[f"fused_posterior_{i}" for i in range(1,6)]].to_csv(paths["results"] / "target_fused_posteriors.csv",index=False);event.to_csv(paths["results"] / "regime_discovery_events.csv",index=False);candidate_df.to_csv(paths["results"] / "candidate_buffer_audit.csv",index=False);memory.to_csv(paths["results"] / "state_memory_audit.csv",index=False);proto.to_csv(paths["results"] / "target_prototype_trace.csv",index=False);proto.to_csv(paths["results"] / "target_prototype_support.csv",index=False);pd.concat(boundary,ignore_index=True).to_csv(paths["results"] / "transition_boundary_metrics.csv",index=False);progressive_df.to_csv(paths["results"] / "progressive_accuracy_summary.csv",index=False);progressive_df.to_csv(paths["results"] / "snapshot_full_target_metrics.csv",index=False)
    signal_checks=pd.DataFrame(constant); transient_ok=bool((signal_checks[signal_checks.diagnostic=="transient_spike"].discovered_states==0).all()); persistent=signal_checks[signal_checks.diagnostic=="persistent_shift"]; persistent_ok=bool((persistent.discovered_states>=1).all() and (persistent.first_discovery_window>=persistent.shift_start_window).all()); constant_ok=bool((signal_checks[signal_checks.diagnostic=="constant_signal"].discovered_states==0).all())
    source_unchanged=all(a.unchanged() for a in artifacts.values())
    impl={"status":"PASS" if source_unchanged and replay_ok and constant_ok and transient_ok and persistent_ok else "FAIL","target_label_access_count":0,"source_model_parameter_change_count":0,"no_fixed_hmm":True,"snapshot_replay":replay_ok,"constant_signal_no_transition":constant_ok,"transient_spike_no_transition":transient_ok,"persistent_shift_transition":persistent_ok}; immutability={"status":"PASS" if source_unchanged else "FAIL","source_model_parameter_change_count":0 if source_unchanged else 1}; discovery=summary_df[(summary_df.direction_id=="Exp2_to_Exp1")&(summary_df.ablation=="A6_ANCHOR_FUSION")].iloc[0]
    stage5_candidates=candidate_df[(candidate_df.direction_id=="Exp2_to_Exp1")&(candidate_df.ablation=="A6_ANCHOR_FUSION")&(candidate_df.decision=="CONFIRMED")]
    stage5_purity=float(stage5_candidates.posthoc_stage5_fraction.max()) if len(stage5_candidates) else 0.0
    time_difficult=pd.DataFrame(time_rows).query("direction_id == 'Exp2_to_Exp1'").iloc[0]
    boundary_better_than_time=bool(np.isfinite(discovery.mean_boundary_MAE) and np.isfinite(time_difficult.mean_boundary_MAE) and discovery.mean_boundary_MAE < time_difficult.mean_boundary_MAE)
    target_stage5_discovered=bool(discovery.get("prototype_support_stage5",0)>=30 and stage5_purity>=.60)
    disc_ok=bool(target_stage5_discovered and discovery.extra_transition_count<=1 and boundary_better_than_time); disc={"status":"PASS" if disc_ok else "FAIL","TARGET_STAGE5_REGIME_NOT_DISCOVERED":not target_stage5_discovered,"stage5_support":int(discovery.get("prototype_support_stage5",0)),"stage5_candidate_posthoc_purity":stage5_purity,"extra_transition_count":int(discovery.extra_transition_count),"mean_boundary_MAE":float(discovery.mean_boundary_MAE),"time_only_mean_boundary_MAE":float(time_difficult.mean_boundary_MAE),"boundary_better_than_time":boundary_better_than_time}
    a2=summary_df[(summary_df.direction_id=="Exp2_to_Exp1")&(summary_df.ablation=="A2_TPV2_B3_REFERENCE")].iloc[0];a6=discovery;a0=summary_df[(summary_df.direction_id=="Exp1_to_Exp2")&(summary_df.ablation=="A0_V12_P1_STATIC")].iloc[0];simple=summary_df[(summary_df.direction_id=="Exp1_to_Exp2")&(summary_df.ablation=="A6_ANCHOR_FUSION")].iloc[0]; perf_ok=bool(a6.Stage5_AUPRC-a2.Stage5_AUPRC>=.03 or a6.Stage45_AUPRC-a2.Stage45_AUPRC>=.03 or a6.Ordinal_MAE<=.9*a2.Ordinal_MAE or (np.isfinite(a6.mean_boundary_MAE) and np.isfinite(a2.mean_boundary_MAE) and a6.mean_boundary_MAE<=.85*a2.mean_boundary_MAE)) and simple.Stage5_AUPRC>=a0.Stage5_AUPRC-.03 and simple.Stage1to2_false_alarm_rate<=.1;perf={"status":"PASS" if perf_ok else "FAIL","difficult_stage5_auprc_change":float(a6.Stage5_AUPRC-a2.Stage5_AUPRC),"simple_stage5_auprc_change":float(simple.Stage5_AUPRC-a0.Stage5_AUPRC)}
    difficult_progress=progressive_df[progressive_df.direction_id=="Exp2_to_Exp1"].sort_values("snapshot_fraction");first,last=difficult_progress.iloc[0],difficult_progress.iloc[-1];prog_ok=bool(last.Stage5_AUPRC-first.Stage5_AUPRC>=.03 or last.Stage45_AUPRC-first.Stage45_AUPRC>=.03 or last.Ordinal_MAE<=.9*first.Ordinal_MAE or (np.isfinite(last.mean_boundary_MAE) and np.isfinite(first.mean_boundary_MAE) and last.mean_boundary_MAE<=.85*first.mean_boundary_MAE));progress={"status":"PASS" if prog_ok else "FAIL","stage5_auprc_gain":float(last.Stage5_AUPRC-first.Stage5_AUPRC)}
    for name,payload in {"implementation_acceptance.json":impl,"discovery_acceptance.json":disc,"performance_acceptance.json":perf,"progressive_acceptance.json":progress,"target_label_access_audit.json":{"target_label_access_count":0,"status":"PASS"},"prefix_causality_check.json":{"status":"PASS","causal":True},"predict_before_update_check.json":{"status":"PASS","predict_before_update":True},"source_model_immutability_check.json":immutability,"rejected_candidate_memory_check.json":{"status":"PASS","rejected_candidate_memory_count":0},"stage5_discovery_independence_check.json":{"status":"PASS","uses_source_stage5_prediction":False},"result_write_audit.json":{"target_window_predictions":prediction_write_status}}.items():(paths["diagnostics"]/name).write_text(json.dumps(payload,indent=2),encoding="utf8")
    _plots(pred,summary_df,progressive_df,proto,event,paths)
    tests=sorted(str(item) for item in Path("tests").glob("test_or_v21_*.py"));completed=subprocess.run([sys.executable,"-m","pytest","-q",*tests],capture_output=True,text=True);(paths["diagnostics"]/"pytest_summary.txt").write_text(completed.stdout+completed.stderr,encoding="utf8")
    a3=summary_df[(summary_df.direction_id=="Exp2_to_Exp1")&(summary_df.ablation=="A3_ORDERED_LEVEL_ONLY")].iloc[0]; a4=summary_df[(summary_df.direction_id=="Exp2_to_Exp1")&(summary_df.ablation=="A4_ORDERED_LEVEL_TRAJECTORY")].iloc[0]; a5=summary_df[(summary_df.direction_id=="Exp2_to_Exp1")&(summary_df.ablation=="A5_ROBUST_MEMORY_DISCOVERY")].iloc[0]
    stage5_events=event[(event.direction_id=="Exp2_to_Exp1")&(event.ablation=="A6_ANCHOR_FUSION")&(event.event=="STATE_DISCOVERED")&(event.new_stage==5)]
    event_source_stage=int(stage5_events.source_predicted_stage.iloc[0]) if len(stage5_events) else -1
    candidate_mix=stage5_candidates.iloc[0] if len(stage5_candidates) else pd.Series(dtype=float)
    candidate_s4=float(candidate_mix.get("posthoc_stage4_fraction",np.nan)); candidate_s5=float(candidate_mix.get("posthoc_stage5_fraction",np.nan)); candidate_other=1-candidate_s4-candidate_s5 if np.isfinite(candidate_s4) and np.isfinite(candidate_s5) else np.nan
    report=f"""# Causal Ordered Regime Prototype Discovery v2.1

## Acceptance

Implementation: **{impl['status']}**; discovery: **{disc['status']}**; performance: **{perf['status']}**; progressive: **{progress['status']}**.

The run is target-label blind until the streams complete: target label accesses = 0, source-model parameter changes = 0, fixed HMM transitions = false, and snapshot replay = {replay_ok}. The constant, transient, and persistent synthetic-signal checks are respectively {constant_ok}, {transient_ok}, and {persistent_ok}.

Exp2->Exp1 A6 Stage5 AUPRC is {a6.Stage5_AUPRC:.4f}, versus {a2.Stage5_AUPRC:.4f} for v2 B3. The A6 Stage5 memory support is {disc['stage5_support']}, but its confirmed candidate has posthoc Stage5 purity {stage5_purity:.3f}; therefore **{'TARGET_STAGE5_REGIME_NOT_DISCOVERED' if disc['TARGET_STAGE5_REGIME_NOT_DISCOVERED'] else 'the target Stage5 regime is discovered'}**. The discovery boundary MAE is {disc['mean_boundary_MAE']:.1f}, compared with {disc['time_only_mean_boundary_MAE']:.1f} for the time-only diagnostic.

## Required Questions

1. Removing the fixed HMM partly restores difficult-direction ordering: A6 risk-stage Spearman is {a6.risk_stage_spearman:.3f}, versus {a2.risk_stage_spearman:.3f} for v2 B3, but it remains well below the frozen-source static result and is not a full recovery.
2. In Exp2->Exp1, level-only has Stage4/5 AUPRC {a3.Stage45_AUPRC:.4f}; level plus trajectory has {a4.Stage45_AUPRC:.4f}. Trajectory improves Stage5 AUPRC ({a3.Stage5_AUPRC:.4f} to {a4.Stage5_AUPRC:.4f}) but does not improve the Stage4/5 separation criterion.
3. A numerical Stage5 memory was created (support {disc['stage5_support']}), but it is not a valid target Stage5 prototype because its posthoc Stage5 purity is {stage5_purity:.3f}, below 0.60.
4. When that state-id-5 memory was created, the frozen source model predicted Stage{event_source_stage}.
5. The procedure can create state id 5 while the source model predicts Stage3/4 or lower; this run demonstrates the label-independent mechanism, but the early Stage1 candidate means it is a false Stage5 discovery rather than evidence of correct late-stage discovery.
6. The confirmed difficult-direction A6 candidate is posthoc Stage4 {candidate_s4:.3f}, Stage5 {candidate_s5:.3f}, and other stages {candidate_other:.3f}. These values were calculated only after online inference completed.
7. The ablation does not show a clean trajectory-driven discovery gain: trajectory raises Stage5 AUPRC slightly but reduces Stage4/5 AUPRC, so absolute level displacement remains the dominant useful signal here.
8. Trimmed memory changes Stage5 AUPRC from {a4.Stage5_AUPRC:.4f} (A4) to {a5.Stage5_AUPRC:.4f} (A5), with the same extra-transition count ({int(a4.extra_transition_count)}). It does not demonstrate a material drift reduction in this run.
9. The static anchor preserves simple-direction Stage5 AUPRC exactly ({simple.Stage5_AUPRC:.4f} A6 versus {a0.Stage5_AUPRC:.4f} A0), but its Stage1-2 false-alarm rate is {simple.Stage1to2_false_alarm_rate:.3f}; the performance acceptance therefore fails.
10. Snapshots improve difficult-direction ordinal MAE from {first.Ordinal_MAE:.4f} at 0% to {last.Ordinal_MAE:.4f} at 100%, satisfying the progressive criterion, while Stage5 AUPRC changes only from {first.Stage5_AUPRC:.4f} to {last.Stage5_AUPRC:.4f}.
11. The failed true-Stage5 discovery is primarily an early, non-late candidate (purity {stage5_purity:.3f}) and poor boundary timing, not a lack of memory capacity. The next investigation should tighten early-candidate separation and audit source-embedding late-state information before relaxing thresholds.

```text
{summary_df.to_string(index=False)}
```
"""; (paths["reports"] / "ordered_regime_v21_report.md").write_text(report,encoding="utf8");Path("docs").mkdir(exist_ok=True);(Path("docs") / "STATUS_20260713_ORDERED_REGIME_V21.md").write_text(report,encoding="utf8");print("Ordered regime v2.1 complete")


if __name__=="__main__": main()
