from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import OrderedRegimeConfig
from .data import reject_target_labels
from .regime_memory import StateMemory
from .source_bundle import SourceArtifacts
from .trajectory import robust_z, trimmed_mean


@dataclass(frozen=True)
class DiscoveryParameters:
    change_threshold: float
    candidate_min_windows: int
    min_dwell_windows: int


class OrderedRegimeDiscovery:
    def __init__(self, source: SourceArtifacts, config: OrderedRegimeConfig, parameters: DiscoveryParameters, use_trajectory: bool = True, robust_memory: bool = True) -> None:
        self.source, self.config, self.parameters = source, config, parameters
        self.use_trajectory, self.robust_memory = use_trajectory, robust_memory
        self.states: dict[int, StateMemory] = {}
        self.current_stage = 1; self.highest_stage = 1; self.dwell = 0; self.cooldown_until = -1
        self.candidate: list[dict] = []; self.candidate_id = 0; self.candidate_gap = 0
        self.events: list[dict] = []; self.candidate_audit: list[dict] = []; self.memory_audit: list[dict] = []
        self.stable_history: list[bool] = []; self.initial_audit: dict | None = None

    def _new_state(self, stage: int) -> StateMemory:
        index = stage - 1
        return StateMemory(stage, self.source.level_proto[index].copy(), self.source.trajectory_proto[index].copy(), float(self.source.level_radius_p95[index]), float(self.source.trajectory_radius_p95[index]))

    def initialize(self, frame: pd.DataFrame, encoded: dict[str, np.ndarray]) -> None:
        mask = (frame.center_cycle.to_numpy(float) <= 500) & ~frame.is_restart_guard.to_numpy(bool) & (frame.TES.to_numpy(float) <= self.source.source_tes_p99) & (encoded["missing_fraction"] == 0)
        if not mask.any():
            mask[:min(20, len(mask))] = True
        posterior = encoded["stage_probs"][mask].mean(axis=0)
        level = encoded["embedding"][mask]
        distance = np.asarray([np.median(np.linalg.norm(level - self.source.level_proto[i], axis=1)) / max(self.source.level_radius_p95[i], 1e-6) for i in range(5)])
        inverse = 1 / (1 + distance); inverse /= inverse.sum()
        score = .60 * posterior + .40 * inverse
        initial = int(np.argmax(score[:4]) + 1)
        self.current_stage = self.highest_stage = initial
        for stage in range(1, initial + 1):
            self.states[stage] = self._new_state(stage)
        self.initial_audit = {"initial_target_stage": initial, **{f"posterior_stage{i+1}":posterior[i] for i in range(5)}, **{f"distance_stage{i+1}":distance[i] for i in range(5)}, **{f"score_stage{i+1}":score[i] for i in range(5)}}

    def _posterior(self, record: dict) -> tuple[np.ndarray, np.ndarray, float]:
        source_p = record["source_posterior"]
        proto = np.zeros(5)
        for stage, state in self.states.items():
            ld = np.linalg.norm(record["level"] - state.level_proto) / max(state.level_radius, 1e-6)
            td = np.linalg.norm(record["trajectory"] - state.trajectory_proto) / max(state.trajectory_radius, 1e-6)
            proto[stage - 1] = np.exp(-(ld + (td if self.use_trajectory else 0)))
        if proto.sum(): proto /= proto.sum()
        weight = min(.75, max((state.support for state in self.states.values()), default=0) / 100.0)
        fused = (1 - weight) * source_p + weight * proto
        fused /= fused.sum()
        return fused, proto, weight

    def _candidate_scores(self, record: dict, history: list[dict]) -> tuple[float, dict]:
        state = self.states[self.current_stage]
        level_distance = float(np.linalg.norm(record["level"] - state.level_proto))
        trajectory_distance = float(np.linalg.norm(record["trajectory"] - state.trajectory_proto))
        lref, tref = state.distances()
        local = np.median(np.stack([item["level"] for item in history[-20:]]), axis=0) if history else record["level"]
        local_distance = float(np.linalg.norm(record["level"] - local))
        level_z = robust_z(level_distance, lref); trajectory_z = robust_z(trajectory_distance, tref)
        local_z = robust_z(local_distance, self.source.level_distance_ref[self.current_stage - 1])
        score = .45 * level_z + .40 * trajectory_z + .15 * local_z if self.use_trajectory else .85 * level_z + .15 * local_z
        return score, {"level_distance": level_distance, "trajectory_distance": trajectory_distance, "local_distance": local_distance, "level_z": level_z, "trajectory_z": trajectory_z, "local_z": local_z}

    def _eligible(self, row: pd.Series, record: dict, index: int) -> tuple[bool, str]:
        if bool(row.is_restart_guard): return False, "RESTART_GUARD"
        if index <= self.cooldown_until: return False, "POST_RESTART_COOLDOWN"
        if float(row.TES) > self.source.source_tes_p99: return False, "TES_OUTLIER"
        if record["missing_fraction"] > 0: return False, "MISSING"
        return True, ""

    def _candidate_decision(self, end_index: int, force_reason: str | None = None) -> None:
        if not self.candidate: return
        values = self.candidate; level = np.stack([item["level"] for item in values]); trajectory = np.stack([item["trajectory"] for item in values])
        lp = trimmed_mean(level); tp = trimmed_mean(trajectory)
        lr = float(np.percentile(np.linalg.norm(level-lp,axis=1),95)); tr=float(np.percentile(np.linalg.norm(trajectory-tp,axis=1),95))
        state=self.states[self.current_stage]; sep=float(np.linalg.norm(lp-state.level_proto)); joint=float(np.linalg.norm(tp-state.trajectory_proto))
        recent=sum(item["window_index"] >= values[-1]["window_index"]-9 for item in values)
        accepted = force_reason is None and len(values) >= self.parameters.candidate_min_windows and lr <= self.config.candidate_radius_multiplier*self.source.typical_level_radius and (not self.use_trajectory or tr <= self.config.candidate_radius_multiplier*self.source.typical_trajectory_radius) and sep >= self.config.candidate_separation_multiplier*state.level_radius and (not self.use_trajectory or joint >= self.config.candidate_separation_multiplier*state.trajectory_radius) and recent >= min(7, len(values)) and self.highest_stage < 5
        reason = "CONFIRMED" if accepted else (force_reason or "CANDIDATE_CRITERIA_FAILED")
        audit={"candidate_id":self.candidate_id,"start_window":values[0]["window_index"],"end_window":end_index,"valid_count":len(values),"change_score_median":float(np.median([item["change_score"] for item in values])),"level_radius":lr,"trajectory_radius":tr,"separation":sep,"joint_separation":joint,"decision":"CONFIRMED" if accepted else "REJECTED","rejection_reason":"" if accepted else reason}
        self.candidate_audit.append(audit)
        if accepted:
            new_stage=self.highest_stage+1; new=StateMemory(new_stage,lp,tp,max(lr,1e-6),max(tr,1e-6))
            for item in values: new.add(item,self.config.memory_per_state,10,self.robust_memory)
            self.states[new_stage]=new; self.highest_stage=self.current_stage=new_stage; self.dwell=0
            self.events.append({"window_index":values[-1]["window_index"],"event":"STATE_DISCOVERED","candidate_id":self.candidate_id,"new_stage":new_stage,"source_predicted_stage":int(np.argmax(values[-1]["source_posterior"])+1)})
        else:
            self.events.append({"window_index":values[-1]["window_index"],"event":"CANDIDATE_REJECTED","candidate_id":self.candidate_id,"reason":reason})
        self.candidate=[]; self.candidate_gap=0

    def _try_stable_memory(self, record: dict, eligible: bool) -> bool:
        if not eligible or self.candidate: return False
        state=self.states[self.current_stage]
        level_ok=np.linalg.norm(record["level"]-state.level_proto)<=state.level_radius
        trajectory_ok=(not self.use_trajectory) or np.linalg.norm(record["trajectory"]-state.trajectory_proto)<=state.trajectory_radius
        self.stable_history.append(bool(level_ok and trajectory_ok)); recent=sum(self.stable_history[-10:])
        if not (level_ok and trajectory_ok and recent >= 7): return False
        record["candidate_id"]="STABLE"; updated=state.add(record,self.config.memory_per_state,self.config.prototype_recompute_every,self.robust_memory)
        self.memory_audit.append({"window_index":record["window_index"],"stage":self.current_stage,"candidate_id":"STABLE","prototype_recomputed":int(updated)})
        return True

    def run(self, frame: pd.DataFrame, encoded: dict[str, np.ndarray], anchor_logit: np.ndarray | None = None, updates: bool = True) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        reject_target_labels(frame)
        if not self.states: self.initialize(frame,encoded)
        history=[]; rows=[]
        for index,row in frame.reset_index(drop=True).iterrows():
            if bool(row.restart_mask): self.cooldown_until=index+self.config.post_restart_cooldown_windows
            record={"window_index":int(row.window_index),"center_cycle":float(row.center_cycle),"level":encoded["embedding"][index].copy(),"trajectory":encoded["trajectory"][index].copy(),"source_posterior":encoded["stage_probs"][index].copy(),"missing_fraction":float(encoded["missing_fraction"][index]),"TES":float(row.TES),"restart_guard":int(row.is_restart_guard),"confidence":float(encoded["stage_probs"][index].max())}
            fused, proto, weight=self._posterior(record)
            predicted=int(np.argmax(fused)+1); anchor=float(1/(1+np.exp(-anchor_logit[index]))) if anchor_logit is not None else float(fused[4])
            stage5_weight=min(.50,self.states.get(5,StateMemory(5,np.zeros(16),np.zeros(48),1,1)).support/100.0) if 5 in self.states else 0.0
            final5=(1-stage5_weight)*anchor+stage5_weight*fused[4]
            change,detail=self._candidate_scores(record,history)
            active_before=bool(self.candidate)
            rows.append({"window_index":int(row.window_index),"center_cycle":float(row.center_cycle),"current_regime_stage":self.current_stage,"highest_discovered_stage":self.highest_stage,"predicted_stage":predicted,"source_predicted_stage":int(np.argmax(record["source_posterior"])+1),"source_expected_stage":float(record["source_posterior"]@np.arange(1,6)),"source_health_score":float(encoded["health"][index]),"change_score":change,"candidate_active":int(active_before),"anchor_stage5_probability":anchor,"stage5_support_weight":stage5_weight,"final_stage5_score":final5,"restart_guard":int(row.is_restart_guard),"missing_fraction":record["missing_fraction"],**detail,**{f"source_posterior_{i+1}":record["source_posterior"][i] for i in range(5)},**{f"fused_posterior_{i+1}":fused[i] for i in range(5)},**{f"prototype_posterior_{i+1}":proto[i] for i in range(5)},"health_axis_projection":encoded["health_axis_projection"][index],"health_axis_velocity20":encoded["health_axis_velocity20"][index],"health_axis_velocity100":encoded["health_axis_velocity100"][index],"trajectory_norm20":encoded["trajectory_norm20"][index],"trajectory_norm100":encoded["trajectory_norm100"][index]})
            eligible, reason=self._eligible(row,record,index)
            qualifies=eligible and self.dwell >= self.parameters.min_dwell_windows and change >= self.parameters.change_threshold
            if updates:
                if qualifies:
                    if not self.candidate: self.candidate_id+=1; self.events.append({"window_index":index,"event":"CANDIDATE_ACTIVE","candidate_id":self.candidate_id})
                    record["change_score"]=change; record["candidate_id"]=self.candidate_id; self.candidate.append(record); self.candidate_gap=0
                    if len(self.candidate) >= self.parameters.candidate_min_windows: self._candidate_decision(index)
                elif self.candidate:
                    self.candidate_gap+=1
                    if self.candidate_gap > self.config.candidate_gap_tolerance: self._candidate_decision(index,"GAP_TOLERANCE_EXCEEDED")
                else:
                    self._try_stable_memory(record,eligible)
            history.append(record); self.dwell+=1
        if updates and self.candidate: self._candidate_decision(len(frame)-1,"STREAM_END")
        proto_rows=[]
        for stage,state in self.states.items():
            proto_rows.append({"stage":stage,"support":state.support,"level_radius":state.level_radius,"trajectory_radius":state.trajectory_radius,**{f"level_{i+1}":value for i,value in enumerate(state.level_proto)},**{f"trajectory_{i+1}":value for i,value in enumerate(state.trajectory_proto)}})
        return (pd.DataFrame(rows), pd.DataFrame(proto_rows),
                pd.DataFrame(self.events, columns=["window_index", "event", "candidate_id", "new_stage", "source_predicted_stage", "reason"]),
                pd.DataFrame(self.memory_audit, columns=["window_index", "stage", "candidate_id", "prototype_recomputed"]))

    def snapshot(self) -> dict:
        return copy.deepcopy({"states":self.states,"current_stage":self.current_stage,"highest_stage":self.highest_stage,"dwell":self.dwell,"cooldown_until":self.cooldown_until,"candidate":self.candidate,"candidate_id":self.candidate_id,"candidate_gap":self.candidate_gap,"stable_history":self.stable_history,"initial_audit":self.initial_audit})

    def restore(self,payload:dict) -> None:
        for key,value in payload.items(): setattr(self,key,copy.deepcopy(value))
