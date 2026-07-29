# V3.5 慢尺度位移与预测残差门控报告

## 协议

- 在线输入仅含当前及历史六维力窗口；不使用停机位置、总长度、固定周期阈值或目标 Stage。
- t、t-16、t-20、t-64、t-128 均以时刻 t 的 Adapter 与 Normalizer 重新编码；评分、候选和冻结判断先于任何更新。
- V3.4.1 对照重跑与原输出一致。
- fast_score 仅用于五窗口安全门控；正式候选由 slow_final_score（SlowOnly）或 slow_final_score 与持续残差的 AND（SlowResidual）决定。
- 所有阈值由前 128 个无标签窗口的 95% 分位数确定并固定；Stage 只在在线输出冻结后用于离线评价。

## 五种种子的关键在线设置

| 方向 | 设置 | 候选事件 | Top-8 | ±250 命中 | Event F1 | 更新比例 | 冻结比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp1_to_Exp2 | V341_OnlineAdaptation | 11.0 | 0.25 | 0.25 | 0.114 | 0.718 | 0.282 |
| Exp1_to_Exp2 | V35_SlowOnly_OnlineAdaptation | 5.2 | 0.25 | 0.05 | 0.025 | 0.797 | 0.203 |
| Exp1_to_Exp2 | V35_SlowResidual_OnlineAdaptation | 3.0 | 0.15 | 0.15 | 0.157 | 0.797 | 0.203 |
| Exp2_to_Exp1 | V341_OnlineAdaptation | 26.2 | 1.00 | 0.85 | 0.227 | 0.383 | 0.617 |
| Exp2_to_Exp1 | V35_SlowOnly_OnlineAdaptation | 5.8 | 0.75 | 0.50 | 0.428 | 0.932 | 0.068 |
| Exp2_to_Exp1 | V35_SlowResidual_OnlineAdaptation | 3.0 | 0.55 | 0.55 | 0.611 | 0.932 | 0.068 |

## 结果判断

- **Exp1_to_Exp2 候选变化：** SlowOnly 相对 V3.4.1 Online 为 5.2 vs 11.0（-52.7%）；Residual Gate 后为 3.0（相对 SlowOnly -42.3%）。
- **Exp2_to_Exp1 候选变化：** SlowOnly 相对 V3.4.1 Online 为 5.8 vs 26.2（-77.9%）；Residual Gate 后为 3.0（相对 SlowOnly -48.3%）。
- **残差门控是否稳定减少两个方向的候选：** 是；以上两个方向均按相同固定 95% 校准阈值运行。
- **Exp1_to_Exp2 的真实边界代价与自适应收益：** SlowResidual Online/Calibration 的 Top-8 为 0.15/0.35，±250 为 0.15/0.65，F1 为 0.157/0.315。
- **Exp2_to_Exp1 的真实边界代价与自适应收益：** SlowResidual Online/Calibration 的 Top-8 为 0.55/0.90，±250 为 0.55/0.70，F1 为 0.611/0.251。
- **失败边界：** Exp1→Exp2：1→2 (Top-8=0.60, mean rank=2.0)；2→3 (Top-8=0.00, mean rank=nan)；3→4 (Top-8=0.00, mean rank=nan)；4→5 (Top-8=0.00, mean rank=nan)。Exp2→Exp1：1→2 (Top-8=0.40, mean rank=1.0)；2→3 (Top-8=0.60, mean rank=3.0)；3→4 (Top-8=0.80, mean rank=2.0)；4→5 (Top-8=0.40, mean rank=3.0)。
- **下一步建议：** 暂不建议直接进入注意力机制：先需解决上述候选减少或边界吸收的失败方向，避免把门控损失误归因于模型容量。

所有六种设置的逐种子结果见 `v35_seed_summary.csv`，边界排名见 `v35_boundary_ranking_metrics.csv`，更新审计见 `v35_adaptation_audit.csv`。
