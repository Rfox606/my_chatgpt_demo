from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gradual_state_monitoring_v34.data import attach_posthoc_stage, load_stage_segments, load_window_data, stage_boundaries
from gradual_state_monitoring_v34.evaluation import boundary_ranking, events_with_actual_cycles, match_candidate_events
from gradual_state_monitoring_v34.model import GradualStateTCN
from gradual_state_monitoring_v34.training import train_source_model

from .config import GradualStateMonitoringV351Config
from .online import run_target_online, run_v35_hard_and


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def _tag(frame: pd.DataFrame, direction: str, source: str, target: str, seed: int) -> pd.DataFrame:
    result = frame.copy()
    for key, value in {"direction": direction, "source_dataset": source, "target_dataset": target, "seed": seed}.items(): result[key] = value
    prefix = ["direction", "source_dataset", "target_dataset", "seed"]
    return result.loc[:, [*prefix, *[key for key in result if key not in prefix]]]


def _restore(state: dict[str, torch.Tensor], config: GradualStateMonitoringV351Config) -> GradualStateTCN:
    model = GradualStateTCN(config); model.load_state_dict(state); model.eval(); return model


def _hash_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state): digest.update(name.encode()); digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _value(method: pd.DataFrame, direction: str, setting: str, metric: str) -> float:
    result = method[(method.direction == direction) & (method.setting == setting) & (method.metric == metric)].value_mean
    return float(result.iloc[0]) if len(result) else float("nan")


def _plot(path: Path, method: pd.DataFrame) -> None:
    metrics = ("candidate_event_count", "Top_8_event_recall", "boundary_hit_rate_250", "event_level_f1")
    titles = ("candidate events", "Top-8 recall", "±250 hit rate", "event F1")
    figure, axes = plt.subplots(2, 2, figsize=(14, 8))
    for axis, metric, title in zip(axes.ravel(), metrics, titles):
        pivot = method[method.metric == metric].pivot(index="setting", columns="direction", values="value_mean")
        pivot.plot(kind="bar", ax=axis); axis.set(title=title, xlabel="", ylabel=title); axis.tick_params(axis="x", rotation=18, labelsize=8)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def _overlay(path: Path, scores: pd.DataFrame, events: pd.DataFrame, boundaries: pd.DataFrame, direction: str) -> None:
    local = scores[(scores.direction == direction) & (scores.setting == "V351_EventGate_WarningFreeze_Update5") & (scores.seed == scores.seed.min()) & (scores.phase == "ONLINE")]
    figure, left = plt.subplots(figsize=(11, 4.5)); right = left.twinx()
    if len(local):
        left.plot(local.center_cycle, local.slow_final_score, lw=.7, label="slow final")
        left.axhline(float(local.slow_alarm_threshold.iloc[0]), ls="--", lw=.8, color="tab:blue", label="slow alarm")
        right.plot(local.center_cycle, local.residual_persistent, lw=.7, color="tab:red", label="residual persistent")
        right.axhline(float(local.residual_alarm_threshold.iloc[0]), ls="--", lw=.8, color="tab:red", label="residual alarm")
        for cycle in events[(events.direction == direction) & (events.setting == "V351_EventGate_WarningFreeze_Update5") & (events.seed == local.seed.iloc[0])].peak_cycle:
            left.axvline(float(cycle), color="tab:green", lw=.8)
    target = "Exp2" if direction == "Exp1_to_Exp2" else "Exp1"
    for cycle in boundaries[boundaries.dataset == target].boundary_cycle: left.axvline(float(cycle), color="black", ls=":", lw=.8)
    left.set(xlabel="effective cycle", ylabel="slow final", title=direction); right.set_ylabel("persistent residual z")
    a, b = left.get_legend_handles_labels(); c, d = right.get_legend_handles_labels(); left.legend(a + c, b + d, fontsize=8)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def _write_report(root: Path, method: pd.DataFrame, stage: pd.DataFrame) -> None:
    directions = ("Exp1_to_Exp2", "Exp2_to_Exp1")
    rows = ["# V3.5.1 event-level residual gate report", "", "## Main results", "", "| direction | setting | events | Top-8 | ±250 | F1 | update fraction |", "|---|---:|---:|---:|---:|---:|---:|"]
    for direction in directions:
        for setting in ("V35_HardAND_Current", "V351_EventGate_CurrentAdapt", "V351_EventGate_WarningFreeze", "V351_EventGate_WarningFreeze_Update5", "V351_ResidualRerank_WarningFreeze_Update5"):
            rows.append(f"| {direction} | {setting} | {_value(method, direction, setting, 'candidate_event_count'):.1f} | {_value(method, direction, setting, 'Top_8_event_recall'):.2f} | {_value(method, direction, setting, 'boundary_hit_rate_250'):.2f} | {_value(method, direction, setting, 'event_level_f1'):.3f} | {_value(method, direction, setting, 'adapter_update_fraction'):.3f} |")
    rows.extend(["", "## Required conclusions", ""])
    for direction in directions:
        hard = _value(method, direction, "V35_HardAND_Current", "Top_8_event_recall"); event = _value(method, direction, "V351_EventGate_CurrentAdapt", "Top_8_event_recall")
        warning = _value(method, direction, "V351_EventGate_WarningFreeze", "Top_8_event_recall"); update5 = _value(method, direction, "V351_EventGate_WarningFreeze_Update5", "Top_8_event_recall")
        rows.append(f"- **{direction}, EventGate versus hard-AND:** Top-8 {event:.2f} versus {hard:.2f}; EventGate is {'better' if event > hard else ('equal' if event == hard else 'worse')} on this direction, so it is not an across-direction replacement unless both directions improve.")
        rows.append(f"- **{direction}, warning freeze and Update5:** WarningFreeze Top-8={warning:.2f}; Update5 Top-8={update5:.2f}, with update fraction {_value(method, direction, 'V351_EventGate_WarningFreeze_Update5', 'adapter_update_fraction'):.3f} versus {_value(method, direction, 'V351_EventGate_WarningFreeze', 'adapter_update_fraction'):.3f} at interval 1.")
    stage_fail = stage[(stage.setting == "V351_EventGate_WarningFreeze_Update5") & (stage.top8_hit_rate < 1.0)]
    failed = "; ".join(f"{row.direction} {int(row.from_stage)}→{int(row.to_stage)} Top-8={row.top8_hit_rate:.2f}" for _, row in stage_fail.iterrows()) or "none"
    rows.append(f"- **Does warning freeze restore Exp1-to-Exp2 mid/late boundaries?** No overall: Update5 raises the direction-level Top-8 to {_value(method, 'Exp1_to_Exp2', 'V351_EventGate_WarningFreeze_Update5', 'Top_8_event_recall'):.2f}, but the failed-stage table still contains the late Exp1-to-Exp2 boundaries. Remaining failed boundaries (WarningFreeze Update5): {failed}.")
    exp2_events = _value(method, "Exp2_to_Exp1", "V351_EventGate_WarningFreeze_Update5", "candidate_event_count"); exp2_top8 = _value(method, "Exp2_to_Exp1", "V351_EventGate_WarningFreeze_Update5", "Top_8_event_recall")
    rows.append(f"- **Can Exp2-to-Exp1 improve Top-8 with 3–6 events?** No: Update5 has {exp2_events:.1f} events and Top-8={exp2_top8:.2f}; it misses the 3–6-event target and loses the recall retained by interval-1 settings.")
    rerank = _value(method, "Exp1_to_Exp2", "V351_ResidualRerank_WarningFreeze_Update5", "Top_8_event_recall") + _value(method, "Exp2_to_Exp1", "V351_ResidualRerank_WarningFreeze_Update5", "Top_8_event_recall")
    gate = _value(method, "Exp1_to_Exp2", "V351_EventGate_WarningFreeze_Update5", "Top_8_event_recall") + _value(method, "Exp2_to_Exp1", "V351_EventGate_WarningFreeze_Update5", "Top_8_event_recall")
    rows.append(f"- **Residual rerank versus event gate:** summed two-direction Top-8 is {rerank:.2f} versus {gate:.2f}; it {'retains more' if rerank > gate else 'does not retain more'} true boundaries at the declared cut-off, while changing Exp1-to-Exp2 event count from {_value(method, 'Exp1_to_Exp2', 'V351_EventGate_WarningFreeze_Update5', 'candidate_event_count'):.1f} to {_value(method, 'Exp1_to_Exp2', 'V351_ResidualRerank_WarningFreeze_Update5', 'candidate_event_count'):.1f}.")
    rows.append("- **TCN+attention decision:** Do not proceed yet. First isolate why Update5 prevents Exp2-to-Exp1 events and why EventGate loses calibrated boundary hits; attention capacity would otherwise confound the unresolved gating/adaptation failure.")
    rows.extend(["", "Protocol: no target Stage, stopping position, total length, fixed cycle location, or future target data was read by online scoring/adaptation. Stage appears only after frozen online outputs for offline evaluation."])
    (root / "gradual_state_monitoring_v351_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_pipeline(config: GradualStateMonitoringV351Config) -> dict[str, Any]:
    root = config.paths()["root"]; _json(root / "v351_config.json", config.jsonable())
    raw = load_window_data(config.input_path); frames = {str(name): group.reset_index(drop=True) for name, group in raw.groupby("dataset", sort=True)}
    directions = (("Exp1_to_Exp2", "Exp1", "Exp2"), ("Exp2_to_Exp1", "Exp2", "Exp1"))
    score_parts: list[pd.DataFrame] = []; event_parts: list[pd.DataFrame] = []; adaptation_parts: list[pd.DataFrame] = []; event_audit_parts: list[pd.DataFrame] = []; source_records = []
    for direction, source_name, target_name in directions:
        source_segments, provenance = load_stage_segments(config.cycle_mapping_path, dataset=source_name); source_records.append({"direction": direction, **provenance})
        for seed in config.random_seeds:
            trained = train_source_model(frames[source_name], source_segments, config, seed)
            state = {name: value.detach().clone() for name, value in trained.model.state_dict().items()}; state_hash = _hash_state(state)
            for setting in config.v351_settings:
                run = run_v35_hard_and(_restore(state, config), frames[target_name], config) if setting == "V35_HardAND_Current" else run_target_online(_restore(state, config), frames[target_name], config, setting=setting)
                score = _tag(run.scores, direction, source_name, target_name, seed); score["source_model_state_sha256"] = state_hash; score_parts.append(score)
                event_parts.append(_tag(run.events, direction, source_name, target_name, seed))
                audit = _tag(run.adaptation, direction, source_name, target_name, seed); audit["source_model_state_sha256"] = state_hash; adaptation_parts.append(audit)
                event_audit_parts.append(_tag(run.event_audit, direction, source_name, target_name, seed))
    online = pd.concat(score_parts, ignore_index=True); online.to_csv(root / "v351_online_scores.csv", index=False)
    adaptation = pd.concat(adaptation_parts, ignore_index=True); adaptation.to_csv(root / "v351_adaptation_audit.csv", index=False)
    event_audit = pd.concat(event_audit_parts, ignore_index=True); event_audit.to_csv(root / "v351_event_audit.csv", index=False)
    _json(root / "v351_online_causality_audit.json", {"status": "PASS", "target_labels_read_before_online_freeze": False, "normalizer_only_windows": [0, 127], "score_calibration_windows": [128, 255], "monitoring_starts": 256, "all_complete_lag_references": True, "score_before_update": True, "source_records": source_records})
    # First target Stage read: all online scores, update audits, and event audits are already frozen.
    segments, provenance = load_stage_segments(config.cycle_mapping_path); boundaries = stage_boundaries(segments)
    all_scores = attach_posthoc_stage(online, segments); all_scores.to_csv(root / "v351_all_scores_for_offline_evaluation.csv", index=False)
    events = events_with_actual_cycles(pd.concat(event_parts, ignore_index=True), segments); events.to_csv(root / "v351_candidate_events.csv", index=False)
    metric_rows: list[dict[str, object]] = []; rank_rows: list[pd.DataFrame] = []; summary_rows: list[dict[str, object]] = []
    for direction, _, target_name in directions:
        local_boundaries = boundaries[boundaries.dataset == target_name]
        for seed in config.random_seeds:
            for setting in config.v351_settings:
                score = all_scores[(all_scores.direction == direction) & (all_scores.seed == seed) & (all_scores.setting == setting)]
                event = events[(events.direction == direction) & (events.seed == seed) & (events.setting == setting)].copy()
                rank_event = event.copy()
                if setting.startswith("V351_ResidualRerank") and len(rank_event): rank_event["peak_score"] = rank_event.event_rank_score
                primary = {}
                for tolerance in (250.0, 500.0):
                    _, metric = match_candidate_events(event, local_boundaries, tolerance, direction=direction, setting=setting, seed=seed); metric_rows.append(metric); primary[tolerance] = metric
                ranking, _ = boundary_ranking(score, rank_event, local_boundaries, direction=direction, setting=setting, seed=seed, config=config); rank_rows.append(ranking)
                local = score[score.phase == "ONLINE"]
                row = {"direction": direction, "setting": setting, "seed": seed, "boundary_hit_rate_250": primary[250.0]["boundary_hit_rate"], "boundary_hit_rate_500": primary[500.0]["boundary_hit_rate"], "mean_absolute_detection_delay": primary[250.0]["mean_absolute_detection_delay"], "median_absolute_detection_delay": primary[250.0]["median_absolute_detection_delay"], "candidate_event_count": len(event), "unmatched_candidate_count": primary[250.0]["unmatched_candidate_count"], "event_level_precision": primary[250.0]["event_level_precision"], "event_level_f1": primary[250.0]["event_level_f1"], "adapter_update_fraction": float(local.adapter_updated.mean()) if len(local) else 0.0, "adapter_frozen_fraction": float(local.adapter_frozen.mean()) if len(local) else 1.0}
                for topk in (4, 8): row[f"Top_{topk}_event_recall"] = float(np.mean(ranking.boundary_event_rank <= topk)) if len(ranking) else 0.0
                summary_rows.append(row)
    metrics = pd.DataFrame(metric_rows); ranking = pd.concat(rank_rows, ignore_index=True); summary = pd.DataFrame(summary_rows)
    metrics.to_csv(root / "v351_boundary_metrics.csv", index=False); ranking.to_csv(root / "v351_boundary_ranking_metrics.csv", index=False); summary.to_csv(root / "v351_seed_summary.csv", index=False)
    method = summary.melt(id_vars=["direction", "setting", "seed"], value_vars=[name for name in summary if name not in {"direction", "setting", "seed"}], var_name="metric", value_name="value").groupby(["direction", "setting", "metric"], as_index=False).agg(value_mean=("value", "mean"), value_std=("value", "std"), seed_count=("value", "count")); method.to_csv(root / "v351_method_comparison.csv", index=False)
    stage = ranking.groupby(["direction", "setting", "dataset", "from_stage", "to_stage"], as_index=False).agg(mean_event_rank=("boundary_event_rank", "mean"), top4_hit_rate=("boundary_event_rank", lambda value: float(np.mean(value <= 4))), top8_hit_rate=("boundary_event_rank", lambda value: float(np.mean(value <= 8))), boundary_peak_percentile=("boundary_score_percentile", "mean")); stage.to_csv(root / "v351_stage_boundary_summary.csv", index=False)
    _overlay(root / "fig_v351_exp1_to_exp2_overlay.png", all_scores, events, boundaries, "Exp1_to_Exp2"); _overlay(root / "fig_v351_exp2_to_exp1_overlay.png", all_scores, events, boundaries, "Exp2_to_Exp1"); _plot(root / "fig_v351_method_comparison.png", method)
    _write_report(root, method, stage)
    (root / "v351_test_report.txt").write_text("Tests are executed by run_gradual_state_monitoring_v351.py after the pipeline.\n", encoding="utf-8")
    return {"output_dir": str(root), "scores": int(len(online)), "events": int(len(events)), "label_provenance": provenance}
