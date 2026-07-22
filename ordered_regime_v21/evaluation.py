from __future__ import annotations

import numpy as np
import pandas as pd

from temporal_prototype_v2.evaluation import stage_metrics


def stable_detection_cycle(frame: pd.DataFrame, threshold: float = .5, required: int = 10) -> float:
    run=0
    for row in frame.itertuples(index=False):
        run = run + 1 if row.final_stage5_score >= threshold else 0
        if run >= required: return float(row.center_cycle)
    return float("nan")


def boundary_metrics(frame: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    true=[]
    for stage in range(2,6):
        part=frame[frame.stage==stage]
        if len(part): true.append((stage,float(part.center_cycle.min())))
    discovered=[] if events.empty else [(int(row.new_stage),float(frame.loc[frame.window_index==row.window_index,"center_cycle"].iloc[0])) for row in events.itertuples(index=False) if row.event=="STATE_DISCOVERED" and int(row.new_stage)<=5]
    rows=[]
    for stage,cycle in true:
        found=[value for discovered_stage,value in discovered if discovered_stage==stage]
        discovered_cycle=found[0] if found else np.nan
        rows.append({"stage":stage,"true_transition_cycle":cycle,"discovered_transition_cycle":discovered_cycle,"signed_boundary_error":discovered_cycle-cycle if np.isfinite(discovered_cycle) else np.nan,"absolute_boundary_error":abs(discovered_cycle-cycle) if np.isfinite(discovered_cycle) else np.nan})
    result=pd.DataFrame(rows)
    return result,{"mean_boundary_MAE":float(result.absolute_boundary_error.mean()) if len(result) else np.nan,"missed_transition_count":int(result.discovered_transition_cycle.isna().sum()) if len(result) else 0,"extra_transition_count":max(0,len(discovered)-len(true)),"early_false_transition_count":int(sum(value < dict(true).get(stage,-np.inf) for stage,value in discovered))}


def evaluate_labeled(frame: pd.DataFrame, events: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    posterior=frame[[f"fused_posterior_{i}" for i in range(1,6)]].to_numpy(float)
    metrics=stage_metrics(frame.stage.to_numpy(int),posterior)
    # Anchor fusion is the declared Stage5 outcome, independent of regime discovery.
    y=(frame.stage.to_numpy(int)==5).astype(int); score=frame.final_stage5_score.to_numpy(float)
    from temporal_prototype_v2.evaluation import _average_precision, _binary_auc
    predicted=frame.predicted_stage.to_numpy(int)
    metrics.update({"Stage5_AUROC":_binary_auc(y,score),"Stage5_AUPRC":_average_precision(y,score),"exact_Stage5_recall":float(np.mean(predicted[y==1]==5)) if y.any() else np.nan,"LateState_recall":float(np.mean(predicted[y==1]>=4)) if y.any() else np.nan,"stable_stage5_detection_cycle":stable_detection_cycle(frame)})
    start=float(frame.loc[frame.stage==5,"center_cycle"].min()) if (frame.stage==5).any() else np.nan
    metrics["lead_lag_cycles"]=start-metrics["stable_stage5_detection_cycle"] if np.isfinite(metrics["stable_stage5_detection_cycle"]) else np.nan
    early=frame.stage.isin([1,2]).to_numpy(); high=frame.final_stage5_score.to_numpy(float)>=.5
    metrics["Stage1to2_false_alarm_rate"]=float(high[early].mean()) if early.any() else np.nan
    bound,extra=boundary_metrics(frame,events); metrics.update(extra)
    return metrics,bound


def source_selection_score(metrics: dict) -> float:
    def val(key,default=0):
        value=metrics.get(key,default); return default if not np.isfinite(value) else float(value)
    boundary=1/(1+val("mean_boundary_MAE",1e6)/500) if np.isfinite(metrics.get("mean_boundary_MAE",np.nan)) else 0
    return .25*boundary+.20*val("ordinal_macro_F1")+.20*(1-min(val("Ordinal_MAE")/4,1))+.15*val("Stage45_AUPRC")+.10*val("Stage5_AUPRC")+.10*(1-min(val("extra_transition_count")/4,1))
