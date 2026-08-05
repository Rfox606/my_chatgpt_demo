from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import GradualStateMonitoringV331Config


def _truth(value: bool) -> str:
    return "是" if value else "否"


def _evidence_importance(ablation: pd.DataFrame, dataset: str) -> str:
    rows = ablation[(ablation.dataset == dataset) & (ablation.variant != "full")].copy()
    if rows.empty:
        return "无可用消融结果"
    full = ablation[(ablation.dataset == dataset) & (ablation.variant == "full")].iloc[0]
    rows["path_effect"] = (rows.state_agreement_with_full - 1.0).abs()
    if float(rows.path_effect.max()) <= 1e-12:
        return "本组删除任一证据后的完整重跑状态路径均未改变；当前阈值下不能区分单项重要性"
    selected = rows.sort_values("path_effect", ascending=False).iloc[0]
    return str(selected.variant)


def write_report(path: Path, config: GradualStateMonitoringV331Config, forecast: pd.DataFrame, scaler: pd.DataFrame,
                 ablation: pd.DataFrame, overlap: pd.DataFrame, delayed: pd.DataFrame, synthetic: pd.DataFrame,
                 divergence: pd.DataFrame, event_audit: pd.DataFrame, gate: dict[str, Any]) -> None:
    frozen = scaler[scaler.scaler_mode == "frozen_descriptor_scaler"].set_index("dataset")
    rolling = scaler[scaler.scaler_mode == "rolling_descriptor_scaler_v33"].set_index("dataset")
    scaler_summary = []
    for dataset in frozen.index:
        rolling_row = rolling.loc[dataset]
        frozen_row = frozen.loc[dataset]
        scaler_summary.append(f"{dataset}: Slow MAE {rolling_row.slow_mae:.6g}→{frozen_row.slow_mae:.6g}，Fast MAE {rolling_row.fast_mae:.6g}→{frozen_row.fast_mae:.6g}，状态切换 {int(rolling_row.state_switches)}→{int(frozen_row.state_switches)}")
    full = ablation[ablation.variant == "full"].set_index("dataset")
    overlap_text = "; ".join(
        f"{dataset} 50%={group.loc[np.isclose(group.window_overlap_ratio, .5), 'state_agreement'].iloc[0]:.3f}, 0%={group.loc[np.isclose(group.window_overlap_ratio, 0), 'state_agreement'].iloc[0]:.3f}"
        for dataset, group in overlap.groupby("dataset")
    )
    delayed_text = "; ".join(
        f"{dataset}: agreement={group.state_agreement.mean():.3f}, IoU={group.transition_interval_iou.mean():.3f}, local-ID difference={group.local_state_count_difference.max()}"
        for dataset, group in delayed.groupby("dataset")
    )
    boundary_share = float(event_audit.within_boundary_guard.mean()) if len(event_audit) else np.nan
    divergence_text = "; ".join(
        f"{dataset}: without={group.loc[group.variant == 'without_model_divergence', 'transition_fraction'].iloc[0]:.3f}, with={group.loc[group.variant == 'with_model_divergence', 'transition_fraction'].iloc[0]:.3f}"
        for dataset, group in divergence.groupby("dataset")
    )
    text = f"""# 渐变状态监测 V3.3.1 审计修正版报告

## 运行边界

V3.3.1 建立在 V3.3 提交 `0900f577e50bd7e00e5c5cc515366c7a2180663f` 之上。在线部分仍只使用六个批准的接触力窗口特征。数据组边界、实验分段映射与形貌元数据只在 `v331_event_boundary_audit.csv` 的离线解释中使用，绝不进入描述符、预测、证据或状态机。

## 审计结论

1. 冻结 scaler 后，预测误差与状态数量确有变化：{'；'.join(scaler_summary)}。冻结模式在第 {config.descriptor_scaler_calibration_windows} 个已到达窗口后固定坐标，随后 Slow/Fast RLS 才开始训练和产生有效预测。
2. 证据重要性（以完整重跑后相对 full 状态路径的变化衡量）：Exp1 最敏感于 `{_evidence_importance(ablation, 'Exp1')}`；Exp2 最敏感于 `{_evidence_importance(ablation, 'Exp2')}`。四种证据配置均重新运行了 Evidence Engine、状态机、状态事件与延迟接入，而非复用 full 标签。
3. Exp2 的 TRANSITION 占比为 {full.loc['Exp2', 'transition_fraction']:.3f}；是否下降应与 V3.3 的历史输出并列解释，不能仅凭本版本宣称改善。
4. 窗口重叠稳定性：{overlap_text}。Gate B2 对 50% 和 0% 重叠率使用预先固定的阈值。
5. 完整 Slow/Fast 延迟接入的收敛：{delayed_text}。PASS 不再只取最后 100 个窗口，而同时检查状态一致率、TRANSITION IoU 和本地状态数量差异。
6. 历史状态复用：`v331_state_reuse_log.csv` 记录每个确认新稳定状态是 created 还是 reused；单元验证覆盖 `1 → 2 → 3 → 2`。
7. 状态事件的边界接近比例为 {boundary_share:.3f}；这些事件均只标记、未删除，详见 `v331_event_boundary_audit.csv`。
8. Gate B：{gate['gate_B_gradual_detection']['status']}（B1={gate['gate_B1_synthetic_gradual_detection']['status']}，B2={gate['gate_B2_real_data_stability']['status']}）；Gate C：{gate['gate_C_new_state_and_online_consistency']['status']}。所有 FAIL 保留在 `v331_gate_decision.json`。
9. 当前仍不具备“真实磨损状态”的充分解释能力：在线模型只表征接触力统计轨迹，本地状态不等同于跨实验可比的磨损机理或形貌状态。

## 预测与模型分离信号

RBF-Ridge 是否优于线性 Ridge，应以 `v331_forecast_metrics.csv` 按 dataset/horizon 比较；Gate A 仍要求中长 horizon 超越 Persistence。`fast_slow_prediction_difference` 始终单独保存为 `model_divergence`。默认三证据结果不加入它；完整 `with_model_divergence` / `without_model_divergence` 重放的 TRANSITION 占比分别为：{divergence_text}。

## Gate

* Gate A：{gate['gate_A_stable_dynamics']['status']}
* Gate B1：{gate['gate_B1_synthetic_gradual_detection']['status']}
* Gate B2：{gate['gate_B2_real_data_stability']['status']}
* Gate C：{gate['gate_C_new_state_and_online_consistency']['status']}
* 总体：{gate['overall_status']}
"""
    path.write_text(text, encoding="utf-8")
