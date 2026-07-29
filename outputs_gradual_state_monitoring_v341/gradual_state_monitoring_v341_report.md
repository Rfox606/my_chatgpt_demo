# V3.4.1 状态转变检测协议修正报告

## 协议核查

- 在线距离对 `t`、`t-16`、`t-64` 和 `t-128` 均用时刻 `t` 的 Adapter 与 Normalizer 重新编码；审计记录为 PASS。
- DirectTransfer 仅使用源域 Normalizer、源模型、源距离/分数历史和固定阈值；没有目标域校准或更新。
- 候选事件按“连续至少 3 窗口高分 → 删除短段 → 间隔不超过 20 窗口合并 → 每段留峰值 → 40 窗口冷却”提取。
- 在线无标签结果先冻结写入 `v341_online_transition_scores.csv`，随后才读取目标 Stage 供离线参考和评价。

## 五个随机种子的主要结果

| 方向 | 设置 | Top-8 召回 | 候选事件数 | 250-cycle 命中率 | 事件 F1 |
|---|---:|---:|---:|---:|---:|
| Exp1→Exp2 | CalibrationOnly | 0.55 | 13.8 | 0.50 | 0.216 |
| Exp1→Exp2 | OnlineAdaptation | 0.25 | 11.0 | 0.25 | 0.114 |
| Exp1→Exp2 | DualBranchAdaptation | 0.55 | 14.0 | 0.45 | 0.194 |
| Exp2→Exp1 | CalibrationOnly | 0.90 | 28.4 | 0.85 | 0.208 |
| Exp2→Exp1 | OnlineAdaptation | 1.00 | 26.2 | 0.85 | 0.227 |
| Exp2→Exp1 | DualBranchAdaptation | 0.85 | 26.2 | 0.80 | 0.209 |

## 任务书问题的回答

1. **同坐标系重编码后，OnlineAdaptation 候选事件数下降。** Exp1→Exp2 从 V3.4 的 17.8 降至 11.0（38.2%）；Exp2→Exp1 从 46.6 降至 26.2（43.8%）。
2. **Exp2→Exp1 的四个 Stage 边界仍全部进入 Top-8：是。** OnlineAdaptation 的平均 Top-8 召回为 1.00。
3. **Exp1→Exp2 的中后期边界排名并非全部改善。** 2→3：V3.4 13.0 → V3.4.1 15.0；3→4：10.0 → 8.0；4→5：10.5 → 9.5。3→4 与 4→5 改善，2→3 未改善；不能宣称三者均改善。
4. **DualBranchAdaptation 未整体优于 CalibrationOnly 和单分支 OnlineAdaptation。** 在 Exp1→Exp2，它将 Top-8 从 Online 的 0.25 提至 0.55，与 CalibrationOnly 持平；在 Exp2→Exp1，它为 0.85，低于 Online 的 1.00 和 CalibrationOnly 的 0.90。
5. **使用正确源域基线后 DirectTransfer 仍然失败。** Exp1→Exp2 的 Top-8 仅 0.30；Exp2→Exp1 为 0.00，平均候选事件数也为 0.0。
6. **在线自适应没有靠增加报警数量制造改善。** 候选事件数在两方向均下降；Exp2→Exp1 仍保持 Top-8=1.00，但 Exp1→Exp2 的 Top-8=0.25，低于 CalibrationOnly 的 0.55。报警更少而 Exp2→Exp1 识别保持强，尚无跨方向的整体识别增益证据。

完整逐种子指标见 `v341_seed_summary.csv`；边界排名见 `v341_boundary_ranking_metrics.csv`。
