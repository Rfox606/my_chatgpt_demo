# 多阶段轨迹审计与状态切换原型 v2 报告

## 最终状态

`FAIL`。工程验证为 `PASS`：完整 pytest 151 项，失败 0 项，错误 0 项；严格前缀因果、无标签泄漏和历史完整性均为 PASS。

## Phase A：多阶段轨迹审计

| 实验 | 持续方向反转 | 滚动 ranker 不稳定 | 回环状态 | 可复现变化点 | 结论 |
| --- | --- | --- | --- | --- | --- |
| Exp1 | PASS | PASS | FAIL | PASS | PASS |
| Exp2 | PASS | PASS | FAIL | PASS | PASS |

Phase A 使用离线稳健平滑仅作审计，在线模型未使用该结果；Stage、形貌和磨屑未被读取。

## Phase B：有界/无硬限幅 adapter

预注册消融结论：`FAIL`。无硬限幅组没有数值安全中止，但未同时满足 AUC、Target Local 一致性、总变差、饱和与共同未来前缀收敛的全部条件；所有方向与模型行保留在 `results/adapter_ablation_future_frozen_v2.csv`。

## Phase C：在线状态切换原型

合成场景：`PASS`；真实状态模型：`FAIL`。失败并非状态编号单调化或标签泄漏：两方向未来冻结重构误差均未达到相对 Single-Axis CEAP v1 的 0.98 门槛；Exp1→Exp2 的边界 bootstrap 稳定性也低于 0.60。

| 方向 | 模型 | 平均未来 NLL | 平均未来重构误差 | 平均后验熵 |
| --- | --- | ---: | ---: | ---: |
| Exp1_to_Exp2 | Adaptive_Regime_Model | 2.565 | 8.516 | 1 |
| Exp1_to_Exp2 | Elapsed_Time_Diagnostic | nan | nan | nan |
| Exp1_to_Exp2 | Single_Axis_CEAP_v1 | nan | 5.374 | nan |
| Exp1_to_Exp2 | Source_Only_State | 2.517 | 8.5 | 0.9999 |
| Exp1_to_Exp2 | Target_Local_Segmentation | 0.09712 | 2.062 | 0.7618 |
| Exp2_to_Exp1 | Adaptive_Regime_Model | 0.4697 | 1.973 | 0.9698 |
| Exp2_to_Exp1 | Elapsed_Time_Diagnostic | nan | nan | nan |
| Exp2_to_Exp1 | Single_Axis_CEAP_v1 | nan | 1.845 | nan |
| Exp2_to_Exp1 | Source_Only_State | 0.4697 | 1.973 | 0.9698 |
| Exp2_to_Exp1 | Target_Local_Segmentation | 0.06827 | 2.13 | 0.4931 |

## 保留的失败与建议

- 无硬限幅 adapter：在不改变已注册学习率与 L2 的前提下，优先研究更合适的无监督未来目标，而不是降低饱和阈值或事后调参。
- 状态模型：应以新的、独立实验的重复轨迹验证源状态语义；当前不应将 `REGIME_n` 解释成绝对磨损或健康等级。
- 事后标签诊断：版本化标签文件不存在，因此明确标记为不可用，未将其补入模型或选择流程。
