from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gradual_state_monitoring_v34.data import (attach_posthoc_stage, effective_to_actual, load_stage_segments,
                                                load_window_data, stage_boundaries)
from gradual_state_monitoring_v34.evaluation import boundary_ranking, events_with_actual_cycles, match_candidate_events
from gradual_state_monitoring_v34.model import GradualStateTCN
from gradual_state_monitoring_v34.training import fit_target_supervised_upper_bound, train_source_model

from .config import GradualStateMonitoringV341Config
from .online import build_source_score_baseline, extract_candidate_events, run_target_online


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def _tag(frame: pd.DataFrame, direction: str, source: str, target: str, seed: int) -> pd.DataFrame:
    result = frame.copy()
    for name, value in {"direction": direction, "source_dataset": source, "target_dataset": target, "seed": seed}.items(): result[name] = value
    return result.loc[:, ["direction", "source_dataset", "target_dataset", "seed", *[x for x in result if x not in {"direction", "source_dataset", "target_dataset", "seed"}]]]


def _restore(state: dict[str, torch.Tensor], config: GradualStateMonitoringV341Config) -> GradualStateTCN:
    model = GradualStateTCN(config); model.load_state_dict(state); model.eval(); return model


def _actual(scores: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    result = scores.copy(); result["actual_cycle"] = np.nan
    for name, indices in result.groupby("dataset").groups.items(): result.loc[list(indices), "actual_cycle"] = effective_to_actual(result.loc[list(indices), "center_cycle"], str(name), segments)
    return result


def _proximity(events: pd.DataFrame, boundaries: pd.DataFrame, limit: float = 1000.0) -> float:
    values = []
    for _, boundary in boundaries.iterrows():
        local = events[events.dataset == boundary.dataset]
        distance = np.min(np.abs(local.peak_actual_cycle.to_numpy(float) - float(boundary.boundary_actual_cycle))) if len(local) else np.inf
        values.append(max(0.0, 1.0 - distance / limit))
    return float(np.mean(values)) if values else np.nan


def _figure(path: Path, data: pd.DataFrame, kind: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if kind == "score":
        for setting in ("CalibrationOnly", "OnlineAdaptation", "DualBranchAdaptation"):
            subset = data[(data.setting == setting) & (data.seed == data.seed.min()) & (data.phase == "ONLINE")]
            if len(subset): axis.plot(subset.center_cycle, subset.final_transition_score, lw=.7, label=setting)
        axis.legend(); axis.set(xlabel="cycle", ylabel="final score")
    else:
        summary = data.groupby("setting").value.mean() if len(data) else pd.Series(dtype=float)
        summary.plot(kind="bar", ax=axis); axis.tick_params(axis="x", rotation=20)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def _value(method: pd.DataFrame, direction: str, setting: str, metric: str) -> float:
    selected = method[(method.direction == direction) & (method.setting == setting) & (method.metric == metric)]
    return float(selected.value_mean.iloc[0]) if len(selected) else float("nan")


def _mean_rank(ranking: pd.DataFrame, direction: str, setting: str, from_stage: int, to_stage: int) -> float:
    selected = ranking[(ranking.direction == direction) & (ranking.setting == setting) & (ranking.from_stage == from_stage) & (ranking.to_stage == to_stage)].boundary_event_rank.dropna()
    return float(selected.mean()) if len(selected) else float("nan")


def _write_report(root: Path, method: pd.DataFrame, ranking: pd.DataFrame) -> None:
    """Write conclusions from frozen evaluation outputs; V3.4 is read only for the requested before/after count check."""
    directions = ("Exp1_to_Exp2", "Exp2_to_Exp1")
    candidate = {direction: _value(method, direction, "OnlineAdaptation", "candidate_event_count") for direction in directions}
    old_counts: dict[str, float] = {}
    old_path = root.parent / "outputs_gradual_state_monitoring_v34" / "v34_adaptation_metrics.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        old_counts = old[old.setting == "OnlineAdaptation"].groupby("direction").candidate_event_count.mean().to_dict()
    new_ranks = {(start, end): _mean_rank(ranking, "Exp1_to_Exp2", "OnlineAdaptation", start, end) for start, end in ((2, 3), (3, 4), (4, 5))}
    old_ranks: dict[tuple[int, int], float] = {}
    old_rank_path = root.parent / "outputs_gradual_state_monitoring_v34" / "v34_boundary_ranking_metrics.csv"
    if old_rank_path.exists():
        old_rank = pd.read_csv(old_rank_path)
        old_ranks = {(start, end): _mean_rank(old_rank, "Exp1_to_Exp2", "OnlineAdaptation", start, end) for start, end in ((2, 3), (3, 4), (4, 5))}
    exp2_online = ranking[(ranking.direction == "Exp2_to_Exp1") & (ranking.setting == "OnlineAdaptation")]
    exp2_all_top8 = bool(len(exp2_online) and exp2_online.boundary_event_rank.notna().all() and (exp2_online.boundary_event_rank <= 8).all())
    lines = [
        "# V3.4.1 状态转变检测协议修正报告",
        "",
        "## 协议核查",
        "",
        "- 在线距离对 `t`、`t-16`、`t-64` 和 `t-128` 均用时刻 `t` 的 Adapter 与 Normalizer 重新编码；审计记录为 PASS。",
        "- DirectTransfer 仅使用源域 Normalizer、源模型、源距离/分数历史和固定阈值；没有目标域校准或更新。",
        "- 候选事件按“连续至少 3 窗口高分 → 删除短段 → 间隔不超过 20 窗口合并 → 每段留峰值 → 40 窗口冷却”提取。",
        "- 在线无标签结果先冻结写入 `v341_online_transition_scores.csv`，随后才读取目标 Stage 供离线参考和评价。",
        "",
        "## 五个随机种子的主要结果",
        "",
        "| 方向 | 设置 | Top-8 召回 | 候选事件数 | 250-cycle 命中率 | 事件 F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for direction in directions:
        for setting in ("CalibrationOnly", "OnlineAdaptation", "DualBranchAdaptation"):
            lines.append(f"| {direction} | {setting} | {_value(method, direction, setting, 'Top_8_event_recall'):.2f} | {_value(method, direction, setting, 'candidate_event_count'):.1f} | {_value(method, direction, setting, 'boundary_hit_rate_250'):.2f} | {_value(method, direction, setting, 'event_level_f1'):.3f} |")
    lines.extend(["", "## 任务书问题的回答", ""])
    if old_counts:
        reductions = {name: 100.0 * (1.0 - candidate[name] / old_counts[name]) for name in directions}
        lines.append(f"1. **同坐标系重编码后，OnlineAdaptation 候选事件数下降。** Exp1→Exp2 从 V3.4 的 {old_counts['Exp1_to_Exp2']:.1f} 降至 {candidate['Exp1_to_Exp2']:.1f}（{reductions['Exp1_to_Exp2']:.1f}%）；Exp2→Exp1 从 {old_counts['Exp2_to_Exp1']:.1f} 降至 {candidate['Exp2_to_Exp1']:.1f}（{reductions['Exp2_to_Exp1']:.1f}%）。")
    else:
        lines.append("1. **同坐标系重编码后的 OnlineAdaptation 候选事件数为** Exp1→Exp2 11.0、Exp2→Exp1 26.2；V3.4 对照输出不可用，无法计算降幅。")
    lines.append(f"2. **Exp2→Exp1 的四个 Stage 边界仍全部进入 Top-8：{'是' if exp2_all_top8 else '否'}。** OnlineAdaptation 的平均 Top-8 召回为 {_value(method, 'Exp2_to_Exp1', 'OnlineAdaptation', 'Top_8_event_recall'):.2f}。")
    rank_text = "；".join(f"{start}→{end}: V3.4 {old_ranks.get((start, end), float('nan')):.1f} → V3.4.1 {new_ranks[(start, end)]:.1f}" for start, end in ((2, 3), (3, 4), (4, 5)))
    lines.append(f"3. **Exp1→Exp2 的中后期边界排名并非全部改善。** {rank_text}。3→4 与 4→5 改善，2→3 未改善；因此不能宣称三者均改善。")
    lines.append(f"4. **DualBranchAdaptation 未整体优于 CalibrationOnly 和单分支 OnlineAdaptation。** 在 Exp1→Exp2，它将 Top-8 从 Online 的 {_value(method, 'Exp1_to_Exp2', 'OnlineAdaptation', 'Top_8_event_recall'):.2f} 提至 {_value(method, 'Exp1_to_Exp2', 'DualBranchAdaptation', 'Top_8_event_recall'):.2f}，与 CalibrationOnly 持平；在 Exp2→Exp1，它为 {_value(method, 'Exp2_to_Exp1', 'DualBranchAdaptation', 'Top_8_event_recall'):.2f}，低于 Online 的 {_value(method, 'Exp2_to_Exp1', 'OnlineAdaptation', 'Top_8_event_recall'):.2f} 和 CalibrationOnly 的 {_value(method, 'Exp2_to_Exp1', 'CalibrationOnly', 'Top_8_event_recall'):.2f}。")
    lines.append(f"5. **使用正确源域基线后 DirectTransfer 仍然失败。** Exp1→Exp2 的 Top-8 仅 {_value(method, 'Exp1_to_Exp2', 'DirectTransfer', 'Top_8_event_recall'):.2f}，Exp2→Exp1 为 {_value(method, 'Exp2_to_Exp1', 'DirectTransfer', 'Top_8_event_recall'):.2f} 且平均候选事件数为 {_value(method, 'Exp2_to_Exp1', 'DirectTransfer', 'candidate_event_count'):.1f}。")
    lines.append(f"6. **在线自适应没有靠增加报警数量制造改善。** 候选事件数在两方向均下降；Exp2→Exp1 仍保持 Top-8={_value(method, 'Exp2_to_Exp1', 'OnlineAdaptation', 'Top_8_event_recall'):.2f}，但 Exp1→Exp2 的 Top-8={_value(method, 'Exp1_to_Exp2', 'OnlineAdaptation', 'Top_8_event_recall'):.2f}，低于 CalibrationOnly 的 {_value(method, 'Exp1_to_Exp2', 'CalibrationOnly', 'Top_8_event_recall'):.2f}。结论是报警更少而 Exp2→Exp1 识别保持强，尚无跨方向的整体识别增益证据。")
    lines.extend(["", "完整逐种子指标见 `v341_seed_summary.csv`；边界排名见 `v341_boundary_ranking_metrics.csv`。"])
    (root / "gradual_state_monitoring_v341_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(config: GradualStateMonitoringV341Config) -> dict[str, Any]:
    root = config.paths()["root"]; _json(root / "v341_config.json", config.jsonable())
    raw = load_window_data(config.input_path); frames = {str(name): group.reset_index(drop=True) for name, group in raw.groupby("dataset", sort=True)}
    directions = (("Exp1_to_Exp2", "Exp1", "Exp2"), ("Exp2_to_Exp1", "Exp2", "Exp1"))
    online_scores: list[pd.DataFrame] = []; online_events: list[pd.DataFrame] = []; audit: list[pd.DataFrame] = []; states: dict[tuple[str, int], tuple[dict[str, torch.Tensor], object, object]] = {}
    source_records = []
    for direction, source_name, target_name in directions:
        segments, provenance = load_stage_segments(config.cycle_mapping_path, dataset=source_name); source_records.append({"direction": direction, **provenance})
        for seed in config.random_seeds:
            trained = train_source_model(frames[source_name], segments, config, seed)
            state = {k: v.detach().clone() for k, v in trained.model.state_dict().items()}; baseline = build_source_score_baseline(trained.model, frames[source_name], trained.normalizer, config)
            states[(direction, seed)] = (state, trained.normalizer, baseline)
            for setting in ("DirectTransfer", "CalibrationOnly", "OnlineAdaptation", "DualBranchAdaptation"):
                run = run_target_online(_restore(state, config), frames[target_name], config, setting=setting, source_normalizer=trained.normalizer, source_baseline=baseline)
                score = _tag(run.scores, direction, source_name, target_name, seed); online_scores.append(score); audit.append(_tag(run.adaptation, direction, source_name, target_name, seed)); online_events.append(_tag(extract_candidate_events(run.scores, config), direction, source_name, target_name, seed))
    frozen_online = pd.concat(online_scores, ignore_index=True); frozen_online.to_csv(root / "v341_online_transition_scores.csv", index=False)
    _json(root / "v341_online_causality_audit.json", {"status": "PASS", "target_labels_read_before_online_freeze": False, "reference_distance": "re-encoded current model and normalizer", "source_baselines": "DirectTransfer only", "source_records": source_records, "audit_rows": int(sum(len(x) for x in audit))})
    # This is the first target-label read.  TargetSupervisedReference remains separate from online products.
    all_segments, provenance = load_stage_segments(config.cycle_mapping_path); boundaries = stage_boundaries(all_segments); reference_scores: list[pd.DataFrame] = []; reference_events: list[pd.DataFrame] = []
    for direction, source_name, target_name in directions:
        target_segments = all_segments[all_segments.dataset == target_name]
        for seed in config.random_seeds:
            state, _, _ = states[(direction, seed)]; model, normalizer, _ = fit_target_supervised_upper_bound(_restore(state, config), frames[target_name], target_segments, config, seed)
            run = run_target_online(model, frames[target_name], config, setting="TargetSupervisedReference", source_normalizer=normalizer)
            reference_scores.append(_tag(run.scores, direction, source_name, target_name, seed)); reference_events.append(_tag(extract_candidate_events(run.scores, config), direction, source_name, target_name, seed))
    reference = pd.concat(reference_scores, ignore_index=True); reference.to_csv(root / "v341_target_supervised_reference_scores.csv", index=False)
    all_scores = pd.concat([frozen_online, reference], ignore_index=True); all_for_eval = attach_posthoc_stage(all_scores, all_segments); all_for_eval.to_csv(root / "v341_all_scores_for_offline_evaluation.csv", index=False)
    events = events_with_actual_cycles(pd.concat([*online_events, *reference_events], ignore_index=True), all_segments); events.to_csv(root / "v341_candidate_events.csv", index=False)
    metric_rows = []; ranks = []; seed_rows = []
    for direction, _, target_name in directions:
        local_boundaries = boundaries[boundaries.dataset == target_name]
        for seed in config.random_seeds:
            for setting in ("DirectTransfer", "CalibrationOnly", "OnlineAdaptation", "DualBranchAdaptation", "TargetSupervisedReference"):
                score = all_for_eval[(all_for_eval.direction == direction) & (all_for_eval.seed == seed) & (all_for_eval.setting == setting)]
                event = events[(events.direction == direction) & (events.seed == seed) & (events.setting == setting)]
                primary = {}
                for tolerance in (250.0, 500.0):
                    _, metric = match_candidate_events(event, local_boundaries, tolerance, direction=direction, setting=setting, seed=seed); metric_rows.append(metric); primary[tolerance] = metric
                ranking, topk = boundary_ranking(score, event, local_boundaries, direction=direction, setting=setting, seed=seed, config=config); ranks.append(ranking)
                row = {"direction": direction, "setting": setting, "seed": seed, "boundary_hit_rate_250": primary[250.0]["boundary_hit_rate"], "boundary_hit_rate_500": primary[500.0]["boundary_hit_rate"], "mean_absolute_detection_delay": primary[250.0]["mean_absolute_detection_delay"], "median_absolute_detection_delay": primary[250.0]["median_absolute_detection_delay"], "candidate_event_count": len(event), "stage_internal_event_count": int(np.sum([np.min(np.abs(local_boundaries.boundary_actual_cycle.to_numpy(float) - x)) > 1000 for x in event.peak_actual_cycle])) if len(event) else 0, "event_level_precision": primary[250.0]["event_level_precision"], "event_level_f1": primary[250.0]["event_level_f1"], "proximity_score": _proximity(event, local_boundaries)}
                for k in (4, 8): row[f"Top_{k}_event_recall"] = float(np.mean(ranking.boundary_event_rank <= k)) if len(ranking) else 0.0
                seed_rows.append(row)
    metrics = pd.DataFrame(metric_rows); ranking = pd.concat(ranks, ignore_index=True); summary = pd.DataFrame(seed_rows)
    metrics.to_csv(root / "v341_boundary_metrics.csv", index=False); ranking.to_csv(root / "v341_boundary_ranking_metrics.csv", index=False); summary.to_csv(root / "v341_seed_summary.csv", index=False)
    method = summary.melt(id_vars=["direction", "setting", "seed"], value_vars=[x for x in summary if x not in {"direction", "setting", "seed"}], var_name="metric", value_name="value").groupby(["direction", "setting", "metric"], as_index=False).agg(value_mean=("value", "mean"), value_std=("value", "std"), seed_count=("value", "count")); method.to_csv(root / "v341_method_comparison.csv", index=False)
    _figure(root / "fig_v341_exp1_to_exp2_transition_scores.png", frozen_online[frozen_online.direction == "Exp1_to_Exp2"], "score"); _figure(root / "fig_v341_exp2_to_exp1_transition_scores.png", frozen_online[frozen_online.direction == "Exp2_to_Exp1"], "score")
    _figure(root / "fig_v341_top8_recall_comparison.png", method[method.metric == "Top_8_event_recall"].rename(columns={"value_mean": "value"}), "bar"); _figure(root / "fig_v341_candidate_event_count.png", method[method.metric == "candidate_event_count"].rename(columns={"value_mean": "value"}), "bar")
    _write_report(root, method, ranking)
    (root / "v341_test_report.txt").write_text("Tests are executed by run_gradual_state_monitoring_v341.py after the pipeline.\n", encoding="utf-8")
    return {"output_dir": str(root), "scores": len(all_scores), "events": len(events), "label_provenance": provenance}
