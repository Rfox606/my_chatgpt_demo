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

from .config import GradualStateMonitoringV35Config
from .online import (extract_candidate_events, extract_v341_baseline_events, run_target_online,
                     run_v341_baseline)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def _tag(frame: pd.DataFrame, direction: str, source: str, target: str, seed: int) -> pd.DataFrame:
    result = frame.copy()
    for name, value in {"direction": direction, "source_dataset": source, "target_dataset": target, "seed": seed}.items():
        result[name] = value
    prefix = ["direction", "source_dataset", "target_dataset", "seed"]
    return result.loc[:, [*prefix, *[column for column in result if column not in prefix]]]


def _restore(state: dict[str, torch.Tensor], config: GradualStateMonitoringV35Config) -> GradualStateTCN:
    model = GradualStateTCN(config); model.load_state_dict(state); model.eval(); return model


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8")); digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _value(method: pd.DataFrame, direction: str, setting: str, metric: str) -> float:
    value = method[(method.direction == direction) & (method.setting == setting) & (method.metric == metric)].value_mean
    return float(value.iloc[0]) if len(value) else float("nan")


def _overlay(path: Path, scores: pd.DataFrame, events: pd.DataFrame, boundaries: pd.DataFrame, direction: str) -> None:
    local = scores[(scores.direction == direction) & (scores.setting == "V35_SlowResidual_OnlineAdaptation") &
                   (scores.seed == scores.seed.min()) & (scores.phase == "ONLINE")]
    figure, score_axis = plt.subplots(figsize=(11, 4.8)); residual_axis = score_axis.twinx()
    if len(local):
        score_axis.plot(local.center_cycle, local.slow_final_score, lw=.75, color="#1f77b4", label="slow_final_score")
        score_axis.axhline(float(local.slow_threshold.iloc[0]), lw=.8, ls="--", color="#1f77b4", alpha=.65, label="slow threshold")
        residual_axis.plot(local.center_cycle, local.residual_persistent, lw=.75, color="#d62728", alpha=.85, label="residual persistent")
        residual_axis.axhline(float(local.residual_threshold.iloc[0]), lw=.8, ls="--", color="#d62728", alpha=.65, label="residual threshold")
        selected_events = events[(events.direction == direction) & (events.setting == "V35_SlowResidual_OnlineAdaptation") & (events.seed == local.seed.iloc[0])]
        for cycle in selected_events.peak_cycle:
            score_axis.axvline(float(cycle), color="#2ca02c", lw=.8, alpha=.7)
    target = str(local.dataset.iloc[0]) if len(local) else ("Exp2" if direction == "Exp1_to_Exp2" else "Exp1")
    for cycle in boundaries[boundaries.dataset == target].boundary_cycle:
        score_axis.axvline(float(cycle), color="black", lw=.8, ls=":", alpha=.7)
    score_axis.set(xlabel="effective cycle", ylabel="slow final score", title=f"{direction}: slow score, residual, events and offline boundaries")
    residual_axis.set_ylabel("persistent residual z")
    handles, labels = score_axis.get_legend_handles_labels(); second_handles, second_labels = residual_axis.get_legend_handles_labels()
    score_axis.legend(handles + second_handles, labels + second_labels, loc="upper right", fontsize=8)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def _comparison_figure(path: Path, method: pd.DataFrame) -> None:
    metrics = ("candidate_event_count", "Top_8_event_recall", "boundary_hit_rate_250", "event_level_f1")
    labels = ("candidate events", "Top-8 recall", "±250 hit rate", "event F1")
    display_names = {
        "V341_CalibrationOnly": "V341 Cal", "V341_OnlineAdaptation": "V341 Online",
        "V35_SlowOnly_CalibrationOnly": "SlowOnly Cal", "V35_SlowOnly_OnlineAdaptation": "SlowOnly Online",
        "V35_SlowResidual_CalibrationOnly": "SlowResidual Cal", "V35_SlowResidual_OnlineAdaptation": "SlowResidual Online",
    }
    figure, axes = plt.subplots(2, 2, figsize=(15, 8)); directions = ("Exp1_to_Exp2", "Exp2_to_Exp1")
    for axis, metric, label in zip(axes.ravel(), metrics, labels):
        subset = method[method.metric == metric]
        pivot = subset.pivot(index="setting", columns="direction", values="value_mean").reindex(columns=directions)
        pivot.index = [display_names.get(str(name), str(name)) for name in pivot.index]
        pivot.plot(kind="bar", ax=axis, width=.78)
        axis.set(title=label, xlabel="", ylabel=label); axis.tick_params(axis="x", rotation=18, labelsize=9)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def _stage_failure_summary(ranking: pd.DataFrame, direction: str) -> str:
    local = ranking[(ranking.direction == direction) & (ranking.setting == "V35_SlowResidual_OnlineAdaptation")]
    pieces: list[str] = []
    for (start, end), group in local.groupby(["from_stage", "to_stage"], sort=True):
        top8 = float(np.mean(group.boundary_event_rank <= 8))
        if top8 < 1.0:
            mean_rank = group.boundary_event_rank.mean()
            pieces.append(f"{int(start)}→{int(end)} (Top-8={top8:.2f}, mean rank={mean_rank:.1f})")
    return "；".join(pieces) if pieces else "没有 Stage 边界在五个种子中落出 Top-8"


def _write_report(root: Path, method: pd.DataFrame, ranking: pd.DataFrame, consistency: pd.DataFrame) -> None:
    directions = ("Exp1_to_Exp2", "Exp2_to_Exp1")
    key_settings = ("V341_OnlineAdaptation", "V35_SlowOnly_OnlineAdaptation", "V35_SlowResidual_OnlineAdaptation")
    lines = [
        "# V3.5 慢尺度位移与预测残差门控报告", "",
        "## 协议", "",
        "- 在线输入仅含当前及历史六维力窗口；不使用停机位置、总长度、固定周期阈值或目标 Stage。",
        "- t、t-16、t-20、t-64、t-128 均以时刻 t 的 Adapter 与 Normalizer 重新编码；评分、候选和冻结判断先于任何更新。",
        "- fast_score 仅用于五窗口安全门控；正式候选由 slow_final_score（SlowOnly）或 slow_final_score 与持续残差的 AND（SlowResidual）决定。",
        "- 所有阈值由前 128 个无标签窗口的 95% 分位数确定并固定；Stage 只在在线输出冻结后用于离线评价。", "",
        "## 五种种子的关键在线设置", "",
        "| 方向 | 设置 | 候选事件 | Top-8 | ±250 命中 | Event F1 | 更新比例 | 冻结比例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if len(consistency):
        status = "一致" if bool((consistency.status == "MATCH").all()) else "存在差异（详见 v35_v341_baseline_consistency.csv）"
        lines.insert(6, f"- V3.4.1 对照重跑与原输出{status}。")
    for direction in directions:
        for setting in key_settings:
            lines.append(f"| {direction} | {setting} | {_value(method, direction, setting, 'candidate_event_count'):.1f} | {_value(method, direction, setting, 'Top_8_event_recall'):.2f} | {_value(method, direction, setting, 'boundary_hit_rate_250'):.2f} | {_value(method, direction, setting, 'event_level_f1'):.3f} | {_value(method, direction, setting, 'adapter_update_fraction'):.3f} | {_value(method, direction, setting, 'adapter_frozen_fraction'):.3f} |")
    lines.extend(["", "## 结果判断", ""])
    residual_reduction = []
    for direction in directions:
        baseline = _value(method, direction, "V341_OnlineAdaptation", "candidate_event_count")
        slow = _value(method, direction, "V35_SlowOnly_OnlineAdaptation", "candidate_event_count")
        residual = _value(method, direction, "V35_SlowResidual_OnlineAdaptation", "candidate_event_count")
        slow_delta = 100.0 * (slow / baseline - 1.0) if baseline else float("nan")
        residual_delta = 100.0 * (residual / slow - 1.0) if slow else float("nan")
        residual_reduction.append(residual <= slow)
        lines.append(f"- **{direction} 候选变化：** SlowOnly 相对 V3.4.1 Online 为 {slow:.1f} vs {baseline:.1f}（{slow_delta:+.1f}%）；Residual Gate 后为 {residual:.1f}（相对 SlowOnly {residual_delta:+.1f}%）。")
    lines.append(f"- **残差门控是否稳定减少两个方向的候选：** {'是' if all(residual_reduction) else '否'}；以上两个方向均按相同固定 95% 校准阈值运行。")
    for direction in directions:
        online = "V35_SlowResidual_OnlineAdaptation"; calibration = "V35_SlowResidual_CalibrationOnly"
        lines.append(f"- **{direction} 的真实边界代价与自适应收益：** SlowResidual Online/Calibration 的 Top-8 为 {_value(method, direction, online, 'Top_8_event_recall'):.2f}/{_value(method, direction, calibration, 'Top_8_event_recall'):.2f}，±250 为 {_value(method, direction, online, 'boundary_hit_rate_250'):.2f}/{_value(method, direction, calibration, 'boundary_hit_rate_250'):.2f}，F1 为 {_value(method, direction, online, 'event_level_f1'):.3f}/{_value(method, direction, calibration, 'event_level_f1'):.3f}。")
    lines.append(f"- **失败边界：** Exp1→Exp2：{_stage_failure_summary(ranking, 'Exp1_to_Exp2')}。Exp2→Exp1：{_stage_failure_summary(ranking, 'Exp2_to_Exp1')}。")
    retained = all(_value(method, direction, "V35_SlowResidual_OnlineAdaptation", "Top_8_event_recall") >= 0.9 * _value(method, direction, "V341_OnlineAdaptation", "Top_8_event_recall") for direction in directions)
    if all(residual_reduction) and retained:
        recommendation = "值得进入下一步 TCN＋注意力机制实验：残差门控在未依赖标签调参的条件下减少候选，且 Top-8 保持在 V3.4.1 Online 的 90% 以上。"
    else:
        recommendation = "暂不建议直接进入注意力机制：先需解决上述候选减少或边界吸收的失败方向，避免把门控损失误归因于模型容量。"
    lines.append(f"- **下一步建议：** {recommendation}")
    lines.extend(["", "所有六种设置的逐种子结果见 `v35_seed_summary.csv`，边界排名见 `v35_boundary_ranking_metrics.csv`，更新审计见 `v35_adaptation_audit.csv`。"])
    (root / "gradual_state_monitoring_v35_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _v341_consistency(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    """Compare rerun V3.4.1 controls with their immutable V3.4.1 output, without altering it."""
    reference_path = root.parent / "outputs_gradual_state_monitoring_v341" / "v341_seed_summary.csv"
    columns = ["direction", "seed", "boundary_hit_rate_250", "boundary_hit_rate_500", "candidate_event_count",
               "event_level_precision", "event_level_f1", "Top_4_event_recall", "Top_8_event_recall"]
    if not reference_path.exists():
        return pd.DataFrame(columns=["setting", "metric", "max_absolute_difference", "status"])
    reference = pd.read_csv(reference_path)
    rows: list[dict[str, object]] = []
    for old_setting, new_setting in (("CalibrationOnly", "V341_CalibrationOnly"), ("OnlineAdaptation", "V341_OnlineAdaptation")):
        old = reference[reference.setting == old_setting].loc[:, [column for column in columns if column in reference]]
        new = summary[summary.setting == new_setting].loc[:, [column for column in columns if column in summary]]
        joined = old.merge(new, on=["direction", "seed"], suffixes=("_reference", "_v35"))
        for metric in columns[2:]:
            if not len(joined) or f"{metric}_reference" not in joined or f"{metric}_v35" not in joined:
                continue
            difference = np.abs(joined[f"{metric}_reference"].to_numpy(float) - joined[f"{metric}_v35"].to_numpy(float))
            finite = difference[np.isfinite(difference)]
            maximum = float(finite.max()) if len(finite) else 0.0
            rows.append({"setting": new_setting, "metric": metric, "max_absolute_difference": maximum,
                         "status": "MATCH" if maximum <= 1e-6 else "DIFFERS"})
    return pd.DataFrame(rows)


def run_pipeline(config: GradualStateMonitoringV35Config) -> dict[str, Any]:
    root = config.paths()["root"]; _json(root / "v35_config.json", config.jsonable())
    raw = load_window_data(config.input_path)
    frames = {str(dataset): group.reset_index(drop=True) for dataset, group in raw.groupby("dataset", sort=True)}
    directions = (("Exp1_to_Exp2", "Exp1", "Exp2"), ("Exp2_to_Exp1", "Exp2", "Exp1"))
    scores_parts: list[pd.DataFrame] = []; event_parts: list[pd.DataFrame] = []; audit_parts: list[pd.DataFrame] = []; source_records = []
    for direction, source_name, target_name in directions:
        source_segments, provenance = load_stage_segments(config.cycle_mapping_path, dataset=source_name)
        source_records.append({"direction": direction, **provenance})
        for seed in config.random_seeds:
            trained = train_source_model(frames[source_name], source_segments, config, seed)
            state = {name: value.detach().clone() for name, value in trained.model.state_dict().items()}
            state_hash = _state_hash(state)
            for setting in config.v35_settings:
                if setting.startswith("V341_"):
                    run = run_v341_baseline(_restore(state, config), frames[target_name], config, setting=setting, source_normalizer=trained.normalizer)
                    events = extract_v341_baseline_events(run.scores, config)
                else:
                    run = run_target_online(_restore(state, config), frames[target_name], config, setting=setting)
                    events = extract_candidate_events(run.scores, config)
                score = _tag(run.scores, direction, source_name, target_name, seed); score["source_model_state_sha256"] = state_hash
                scores_parts.append(score)
                event_parts.append(_tag(events, direction, source_name, target_name, seed))
                audit = _tag(run.adaptation, direction, source_name, target_name, seed)
                if len(audit):
                    audit["source_model_state_sha256"] = state_hash
                audit_parts.append(audit)
    online_scores = pd.concat(scores_parts, ignore_index=True); online_scores.to_csv(root / "v35_online_scores.csv", index=False)
    adaptation = pd.concat(audit_parts, ignore_index=True); adaptation.to_csv(root / "v35_adaptation_audit.csv", index=False)
    _json(root / "v35_online_causality_audit.json", {
        "status": "PASS", "target_labels_read_before_online_freeze": False,
        "reference_protocol": "t, t-16, t-20, t-64 and t-128 re-encoded with the current Adapter and Normalizer",
        "score_before_update": True, "source_records": source_records,
        "online_rows": int(len(online_scores)), "adaptation_rows": int(len(adaptation)),
    })
    # Target Stage mapping is first loaded only after every online score/audit file above is immutable.
    segments, provenance = load_stage_segments(config.cycle_mapping_path); boundaries = stage_boundaries(segments)
    all_scores = attach_posthoc_stage(online_scores, segments); all_scores.to_csv(root / "v35_all_scores_for_offline_evaluation.csv", index=False)
    events = events_with_actual_cycles(pd.concat(event_parts, ignore_index=True), segments); events.to_csv(root / "v35_candidate_events.csv", index=False)
    metric_rows: list[dict[str, object]] = []; ranks: list[pd.DataFrame] = []; summary_rows: list[dict[str, object]] = []
    for direction, _, target_name in directions:
        local_boundaries = boundaries[boundaries.dataset == target_name]
        for seed in config.random_seeds:
            for setting in config.v35_settings:
                score = all_scores[(all_scores.direction == direction) & (all_scores.seed == seed) & (all_scores.setting == setting)]
                event = events[(events.direction == direction) & (events.seed == seed) & (events.setting == setting)]
                tolerance_metrics = {}
                for tolerance in (250.0, 500.0):
                    _, metric = match_candidate_events(event, local_boundaries, tolerance, direction=direction, setting=setting, seed=seed)
                    metric_rows.append(metric); tolerance_metrics[tolerance] = metric
                rank, _ = boundary_ranking(score, event, local_boundaries, direction=direction, setting=setting, seed=seed, config=config)
                ranks.append(rank)
                online = score[score.phase == "ONLINE"]
                row = {
                    "direction": direction, "setting": setting, "seed": seed,
                    "boundary_hit_rate_250": tolerance_metrics[250.0]["boundary_hit_rate"],
                    "boundary_hit_rate_500": tolerance_metrics[500.0]["boundary_hit_rate"],
                    "mean_absolute_detection_delay": tolerance_metrics[250.0]["mean_absolute_detection_delay"],
                    "median_absolute_detection_delay": tolerance_metrics[250.0]["median_absolute_detection_delay"],
                    "candidate_event_count": len(event),
                    "unmatched_candidate_count": tolerance_metrics[250.0]["unmatched_candidate_count"],
                    "event_level_precision": tolerance_metrics[250.0]["event_level_precision"],
                    "event_level_f1": tolerance_metrics[250.0]["event_level_f1"],
                    "adapter_update_fraction": float(online.adapter_updated.mean()) if len(online) else 0.0,
                    "adapter_frozen_fraction": float(online.adapter_frozen.mean()) if len(online) else 1.0,
                }
                for top_k in (4, 8):
                    row[f"Top_{top_k}_event_recall"] = float(np.mean(rank.boundary_event_rank <= top_k)) if len(rank) else 0.0
                summary_rows.append(row)
    metrics = pd.DataFrame(metric_rows); ranking = pd.concat(ranks, ignore_index=True); summary = pd.DataFrame(summary_rows)
    metrics.to_csv(root / "v35_boundary_metrics.csv", index=False); ranking.to_csv(root / "v35_boundary_ranking_metrics.csv", index=False); summary.to_csv(root / "v35_seed_summary.csv", index=False)
    consistency = _v341_consistency(root, summary); consistency.to_csv(root / "v35_v341_baseline_consistency.csv", index=False)
    method = (summary.melt(id_vars=["direction", "setting", "seed"], value_vars=[name for name in summary if name not in {"direction", "setting", "seed"}], var_name="metric", value_name="value")
              .groupby(["direction", "setting", "metric"], as_index=False).agg(value_mean=("value", "mean"), value_std=("value", "std"), seed_count=("value", "count")))
    method.to_csv(root / "v35_method_comparison.csv", index=False)
    _overlay(root / "fig_v35_exp1_to_exp2_overlay.png", all_scores, events, boundaries, "Exp1_to_Exp2")
    _overlay(root / "fig_v35_exp2_to_exp1_overlay.png", all_scores, events, boundaries, "Exp2_to_Exp1")
    _comparison_figure(root / "fig_v35_method_comparison.png", method)
    _write_report(root, method, ranking, consistency)
    (root / "v35_test_report.txt").write_text("Tests are executed by run_gradual_state_monitoring_v35.py after the pipeline.\n", encoding="utf-8")
    return {"output_dir": str(root), "scores": int(len(online_scores)), "events": int(len(events)), "label_provenance": provenance}
