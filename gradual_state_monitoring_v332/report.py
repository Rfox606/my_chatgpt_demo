from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import GradualStateMonitoringV332Config


def _number(value: object, digits: int = 3) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "（无可用行）"
    output = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    headers = output.columns.tolist()
    def cell(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        return str(value).replace("|", "\\|")
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    rows.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in output.itertuples(index=False, name=None))
    return "\n".join(rows)


def write_report(root: Path, config: GradualStateMonitoringV332Config, provenance: dict[str, object], boundaries: pd.DataFrame,
                 stable: pd.DataFrame, boundary: pd.DataFrame, match_log: pd.DataFrame, mapping_summary: list[dict[str, float]],
                 forecast: pd.DataFrame, comparison: pd.DataFrame, gates: dict[str, object]) -> Path:
    """Write an evidence-first report: label detection/stability outrank prediction MAE."""
    baseline_boundary = boundary[(boundary.model == "V331") & (boundary.setting == "v331_frozen") &
                                 (boundary.tolerance_actual_cycles == config.primary_boundary_tolerance_actual_cycles)].sort_values("dataset")
    baseline_stable = stable[(stable.model == "V331") & (stable.setting == "v331_frozen")].sort_values(["dataset", "stage"])
    baseline_compare = comparison[comparison.model == "V331"].sort_values("dataset")
    missed = match_log[(match_log.model == "V331") & (match_log.setting == "v331_frozen") &
                       (match_log.tolerance_actual_cycles == config.primary_boundary_tolerance_actual_cycles) &
                       (match_log.match_status == "MISSED")].sort_values(["dataset", "boundary_actual_cycle"])
    fragments = baseline_stable.sort_values(["detected_state_count", "state_switch_count"], ascending=False).head(5)
    exp2_boundary = baseline_boundary[baseline_boundary.dataset == "Exp2"]
    exp2_transition = baseline_compare[baseline_compare.dataset == "Exp2"]
    long_forecast = forecast[forecast.horizon.isin((100, 300))].copy()
    nonlinear = long_forecast[long_forecast.model.isin(("TCN", "GRU"))].sort_values("relative_improvement_vs_Persistence", ascending=False)
    best_nonlinear = nonlinear.iloc[0] if len(nonlinear) else pd.Series(dtype=float)
    rbf = long_forecast[long_forecast.model == "RBF_Ridge"].sort_values("relative_improvement_vs_Persistence", ascending=False)
    best_rbf = rbf.iloc[0] if len(rbf) else pd.Series(dtype=float)
    exp2_starts = forecast[(forecast.target_dataset == "Exp2") & forecast.model.eq("Persistence")].groupby("training_setting").prediction_start_index.min().to_dict()
    mapping_frame = pd.DataFrame(mapping_summary)
    lines = [
        "# 渐变状态监测 V3.3.2：标签验证与非线性模型对比报告",
        "",
        "## 结论",
        "",
        f"总体 Gate：**{gates['overall_status']}**。本轮以真实 Stage 边界检测和阶段内部稳定性为主判据，而不是以预测 MAE 或模型间状态一致性判定成功。",
        "V3.3.1 的推理代码和输出没有重新运行、修改或覆盖：本版本只读取其冻结状态轨迹，并在所有新模型的无标签在线推理完成后才附加既有 Stage 定义。",
        "",
        "## 标签来源与因果边界",
        "",
        f"标签来源：`{provenance['source_path']}`（SHA-256 `{provenance['source_sha256']}`）。该文件原有 Exp1/Exp2 的 Stage 1–5 有效周期—实际周期映射；本版本没有重新猜测、平移或针对结果调整任何边界。",
        "Stage、形貌、磨屑及未来窗口均被在线输入检查拒绝。TCN/GRU/Ridge/RBF-Ridge 在任何标签文件被读取前完成训练、预测、prediction surprise、Evidence Engine 与状态机回放；Stage 仅在此后用于 CSV 指标和图的背景色。",
        "",
        "## 1. V3.3.1 状态变化与真实 Stage 边界",
        "",
        "主结果采用 ±250 实际循环，并只把 `STABLE → TRANSITION` 计作报警。",
        "",
        _table(baseline_boundary, ["dataset", "boundary_precision", "boundary_recall", "boundary_f1", "mean_detection_delay", "missed_boundary_count", "false_alarm_count", "duplicate_alarm_count"]),
        "",
        f"对应边界数为 {int(len(boundaries))}；V3.3.1 在两个实验的匹配报警数见上表。漏检边界如下（若表为空则无漏检）：",
        "",
        _table(missed, ["dataset", "from_stage", "to_stage", "boundary_actual_cycle", "match_status"]),
        "",
        "## 2. 阶段内部稳定性与过度分割",
        "",
        "每个 Stage 在真实边界前后 ±250 实际循环被移出稳定性分母。`stage_stable_rate` 仅计 `STABLE`；`detected_state_count` 使用本地 `local_state_id`。",
        "",
        _table(baseline_stable, ["dataset", "stage", "stage_stable_rate", "stage_transition_rate", "dominant_detected_state", "stage_purity", "detected_state_count", "state_switch_count", "false_switches_per_1000_windows"]),
        "",
        "分割最明显的阶段（按检测状态数与切换数排序）：",
        "",
        _table(fragments, ["dataset", "stage", "detected_state_count", "state_switch_count", "false_switches_per_1000_windows"]),
        "",
        "## 3. Exp2 的大量 TRANSITION 是否主要是真实边界",
        "",
        f"Exp2 的 V3.3.1 `TRANSITION` 占比为 {_number(exp2_transition.transition_fraction.iloc[0] if len(exp2_transition) else np.nan)}。在 ±250 循环下，其边界匹配、误报、重复报警和漏检分别为 "
        f"{int(exp2_boundary.matched_boundary_count.iloc[0]) if len(exp2_boundary) else 0}、{int(exp2_boundary.false_alarm_count.iloc[0]) if len(exp2_boundary) else 0}、"
        f"{int(exp2_boundary.duplicate_alarm_count.iloc[0]) if len(exp2_boundary) else 0}、{int(exp2_boundary.missed_boundary_count.iloc[0]) if len(exp2_boundary) else 0}。"
        " 因此 Exp2 的长 TRANSITION 不能仅解释为真实 Stage 边界；应以边界精度与阶段内部切换共同审计。",
        "",
        "## 4. 顺序约束状态对应",
        "",
        "映射按 `local_state_id` 首次出现顺序施加非递减 Stage 约束；未使用无序 Hungarian 匹配。多个检测状态可映射同一 Stage。",
        "",
        _table(mapping_frame, ["dataset", "ordered_mapping_accuracy", "macro_f1", "ARI", "NMI"]),
        "",
        "完整对应和计数见 `v332_ordered_state_mapping.csv`、`v332_confusion_matrix.csv`。",
        "",
        "## 5. 非线性预测能力",
        "",
        "所有模型使用相同的最近 128 个原始六特征窗口、相同的 20/100/300 horizon 与预先固定的训练协议。within 实验只用时间前 40% 的无标签前缀训练并评估全部后缀；跨实验使用全部历史 Exp1 预训练，Exp2 前 128 个窗口仅用于冻结 scaler。",
        "",
        _table(forecast.sort_values(["training_setting", "target_dataset", "horizon", "model"]), ["training_setting", "target_dataset", "model", "horizon", "MAE", "stable_interval_MAE", "transition_interval_MAE", "relative_improvement_vs_Persistence"]),
        "",
        f"最佳中长 horizon 非线性行：模型 `{best_nonlinear.get('model', 'NA')}`，设置 `{best_nonlinear.get('training_setting', 'NA')}`，h={best_nonlinear.get('horizon', 'NA')}，相对 Persistence 改善 {_number(best_nonlinear.get('relative_improvement_vs_Persistence', np.nan))}。"
        f"最佳 RBF-Ridge 同类行的相对改善为 {_number(best_rbf.get('relative_improvement_vs_Persistence', np.nan))}。Gate A 的结论以所有预注册中长 horizon 行为准，不从中挑选评价区间。",
        "",
        "## 6. 预测 MAE 是否同步改善状态检测",
        "",
        _table(comparison.sort_values(["dataset", "training_setting", "model"]), ["dataset", "training_setting", "model", "boundary_precision", "boundary_recall", "boundary_f1", "stage_stable_rate", "false_alarms_per_1000_windows", "transition_fraction", "state_fragmentation"]),
        "",
        "同一 V3.3.1 Evidence Engine、分布漂移、漂移速度和状态机参数被固定；模型差异只进入 prediction surprise。因而预测 MAE 改善若未同时改善 Boundary F1 或稳定区误报，不能称为状态检测升级。Gate D 的逐模型审计保存在 `v332_gate_decision.json`。",
        "",
        "## 7. 跨实验预训练的早期效应",
        "",
        f"Exp1→Exp2 的预测从 Exp2 窗口 {exp2_starts.get('Exp1_pretrain_to_Exp2', 'NA')} 开始；within_Exp2 的预注册前缀训练完成后才从窗口 {exp2_starts.get('within_Exp2', 'NA')} 开始。"
        " 这说明跨实验模型具有更早的可用预测，但本轮没有在目标早期构造带标签或未来目标的 scratch 对照，因此不能把“更早可用”表述为已经证实的早期检测增益。",
        "",
        "## 8. Gate 与研究结论",
        "",
        f"- Gate A（预测能力）：{gates['gate_A_forecast']['status']}\n"
        f"- Gate B（标签边界检测）：{gates['gate_B_label_boundary_detection']['status']}\n"
        f"- Gate C（阶段内部稳定性）：{gates['gate_C_stage_interior_stability']['status']}\n"
        f"- Gate D（模型升级价值）：{gates['gate_D_model_upgrade_value']['status']}",
        "",
        f"因此当前证据总体为 **{gates['overall_status']}**。只有当标签边界召回/精度和阶段内部稳定性共同通过时，才足以支持“接触力信号能够表征磨损状态演化”的强结论；本次结果不以模型自身状态一致性或预测 MAE 代替该验证。",
        "",
        "## 生成文件",
        "",
        "`v332_label_provenance.json`、`v332_stage_boundaries.csv`、`v332_stage_stability_metrics.csv`、`v332_boundary_detection_metrics.csv`、`v332_boundary_match_log.csv`、`v332_ordered_state_mapping.csv`、`v332_confusion_matrix.csv`、`v332_forecast_metrics.csv`、`v332_model_state_detection_comparison.csv`、`v332_gate_decision.json`，以及四张 Stage 对齐/汇总图。",
    ]
    output = root / "gradual_state_monitoring_v332_report.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
