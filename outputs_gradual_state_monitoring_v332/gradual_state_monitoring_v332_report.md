# 渐变状态监测 V3.3.2：标签验证与非线性模型对比报告

## 结论

总体 Gate：**FAIL**。本轮以真实 Stage 边界检测和阶段内部稳定性为主判据，而不是以预测 MAE 或模型间状态一致性判定成功。
V3.3.1 的推理代码和输出没有重新运行、修改或覆盖：本版本只读取其冻结状态轨迹，并在所有新模型的无标签在线推理完成后才附加既有 Stage 定义。

## 标签来源与因果边界

标签来源：`outputs_physical_validation_candidates_v1\configs\cycle_mapping_config.json`（SHA-256 `9cba8cc680d57a48d18a21cba9914e9671174d602cbc69ff0252d197d4eccaba`）。该文件原有 Exp1/Exp2 的 Stage 1–5 有效周期—实际周期映射；本版本没有重新猜测、平移或针对结果调整任何边界。
Stage、形貌、磨屑及未来窗口均被在线输入检查拒绝。TCN/GRU/Ridge/RBF-Ridge 在任何标签文件被读取前完成训练、预测、prediction surprise、Evidence Engine 与状态机回放；Stage 仅在此后用于 CSV 指标和图的背景色。

## 1. V3.3.1 状态变化与真实 Stage 边界

主结果采用 ±250 实际循环，并只把 `STABLE → TRANSITION` 计作报警。

| dataset | boundary_precision | boundary_recall | boundary_f1 | mean_detection_delay | missed_boundary_count | false_alarm_count | duplicate_alarm_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1 | 0.200 | 1.000 | 0.333 | 18.372 | 0 | 16 | 0 |
| Exp2 | 0.071 | 0.500 | 0.125 | -103.143 | 2 | 26 | 1 |

对应边界数为 8；V3.3.1 在两个实验的匹配报警数见上表。漏检边界如下（若表为空则无漏检）：

| dataset | from_stage | to_stage | boundary_actual_cycle | match_status |
| --- | --- | --- | --- | --- |
| Exp2 | 2.000 | 3.000 | 10500.000 | MISSED |
| Exp2 | 4.000 | 5.000 | 20000.000 | MISSED |

## 2. 阶段内部稳定性与过度分割

每个 Stage 在真实边界前后 ±250 实际循环被移出稳定性分母。`stage_stable_rate` 仅计 `STABLE`；`detected_state_count` 使用本地 `local_state_id`。

| dataset | stage | stage_stable_rate | stage_transition_rate | dominant_detected_state | stage_purity | detected_state_count | state_switch_count | false_switches_per_1000_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1 | 1 | 0.944 | 0.041 | 1 | 0.877 | 2 | 6 | 4.093 |
| Exp1 | 2 | 0.890 | 0.089 | 3 | 0.984 | 2 | 12 | 4.571 |
| Exp1 | 3 | 0.893 | 0.080 | 3 | 1.000 | 1 | 11 | 8.737 |
| Exp1 | 4 | 0.973 | 0.017 | 3 | 1.000 | 1 | 6 | 4.736 |
| Exp1 | 5 | 0.916 | 0.058 | 4 | 0.629 | 3 | 17 | 7.892 |
| Exp2 | 1 | 0.497 | 0.383 | 2 | 0.708 | 2 | 24 | 42.179 |
| Exp2 | 2 | 0.306 | 0.594 | 3 | 0.744 | 3 | 13 | 24.074 |
| Exp2 | 3 | 0.242 | 0.667 | 6 | 0.371 | 3 | 13 | 27.083 |
| Exp2 | 4 | 0.565 | 0.380 | 6 | 1.000 | 1 | 16 | 29.630 |
| Exp2 | 5 | 0.757 | 0.181 | 7 | 0.938 | 2 | 8 | 17.857 |

分割最明显的阶段（按检测状态数与切换数排序）：

| dataset | stage | detected_state_count | state_switch_count | false_switches_per_1000_windows |
| --- | --- | --- | --- | --- |
| Exp1 | 5 | 3 | 17 | 7.892 |
| Exp2 | 2 | 3 | 13 | 24.074 |
| Exp2 | 3 | 3 | 13 | 27.083 |
| Exp2 | 1 | 2 | 24 | 42.179 |
| Exp1 | 2 | 2 | 12 | 4.571 |

## 3. Exp2 的大量 TRANSITION 是否主要是真实边界

Exp2 的 V3.3.1 `TRANSITION` 占比为 0.452。在 ±250 循环下，其边界匹配、误报、重复报警和漏检分别为 2、26、1、2。 因此 Exp2 的长 TRANSITION 不能仅解释为真实 Stage 边界；应以边界精度与阶段内部切换共同审计。

## 4. 顺序约束状态对应

映射按 `local_state_id` 首次出现顺序施加非递减 Stage 约束；未使用无序 Hungarian 匹配。多个检测状态可映射同一 Stage。

| dataset | ordered_mapping_accuracy | macro_f1 | ARI | NMI |
| --- | --- | --- | --- | --- |
| Exp1 | 0.686 | 0.520 | 0.401 | 0.646 |
| Exp2 | 0.835 | 0.833 | 0.587 | 0.714 |

完整对应和计数见 `v332_ordered_state_mapping.csv`、`v332_confusion_matrix.csv`。

## 5. 非线性预测能力

所有模型使用相同的最近 128 个原始六特征窗口、相同的 20/100/300 horizon 与预先固定的训练协议。within 实验只用时间前 40% 的无标签前缀训练并评估全部后缀；跨实验使用全部历史 Exp1 预训练，Exp2 前 128 个窗口仅用于冻结 scaler。

| training_setting | target_dataset | model | horizon | MAE | stable_interval_MAE | transition_interval_MAE | relative_improvement_vs_Persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1_pretrain_to_Exp2 | Exp2 | GRU | 20 | 0.021 | 0.020 | 0.025 | -0.330 |
| Exp1_pretrain_to_Exp2 | Exp2 | Persistence | 20 | 0.016 | 0.015 | 0.020 | 0.000 |
| Exp1_pretrain_to_Exp2 | Exp2 | RBF_Ridge | 20 | 0.031 | 0.030 | 0.035 | -0.970 |
| Exp1_pretrain_to_Exp2 | Exp2 | Ridge | 20 | 0.906 | 0.915 | 0.812 | -56.876 |
| Exp1_pretrain_to_Exp2 | Exp2 | TCN | 20 | 0.017 | 0.017 | 0.021 | -0.088 |
| Exp1_pretrain_to_Exp2 | Exp2 | GRU | 100 | 0.034 | 0.034 | 0.037 | -0.004 |
| Exp1_pretrain_to_Exp2 | Exp2 | Persistence | 100 | 0.034 | 0.034 | 0.035 | 0.000 |
| Exp1_pretrain_to_Exp2 | Exp2 | RBF_Ridge | 100 | 0.081 | 0.080 | 0.088 | -1.358 |
| Exp1_pretrain_to_Exp2 | Exp2 | Ridge | 100 | 3.472 | 3.518 | 3.021 | -100.128 |
| Exp1_pretrain_to_Exp2 | Exp2 | TCN | 100 | 0.034 | 0.034 | 0.035 | -0.000 |
| Exp1_pretrain_to_Exp2 | Exp2 | GRU | 300 | 0.049 | 0.049 | 0.052 | -0.056 |
| Exp1_pretrain_to_Exp2 | Exp2 | Persistence | 300 | 0.047 | 0.046 | 0.049 | 0.000 |
| Exp1_pretrain_to_Exp2 | Exp2 | RBF_Ridge | 300 | 0.159 | 0.158 | 0.171 | -2.424 |
| Exp1_pretrain_to_Exp2 | Exp2 | Ridge | 300 | 3.600 | 3.545 | 4.093 | -76.380 |
| Exp1_pretrain_to_Exp2 | Exp2 | TCN | 300 | 0.047 | 0.046 | 0.050 | -0.002 |
| within_Exp1 | Exp1 | GRU | 20 | 0.002 | 0.002 | 0.009 | -0.201 |
| within_Exp1 | Exp1 | Persistence | 20 | 0.002 | 0.001 | 0.009 | 0.000 |
| within_Exp1 | Exp1 | RBF_Ridge | 20 | 0.002 | 0.002 | 0.009 | -0.104 |
| within_Exp1 | Exp1 | Ridge | 20 | 0.004 | 0.004 | 0.013 | -1.433 |
| within_Exp1 | Exp1 | TCN | 20 | 0.002 | 0.002 | 0.009 | -0.313 |
| within_Exp1 | Exp1 | GRU | 100 | 0.006 | 0.005 | 0.018 | -0.052 |
| within_Exp1 | Exp1 | Persistence | 100 | 0.005 | 0.005 | 0.018 | 0.000 |
| within_Exp1 | Exp1 | RBF_Ridge | 100 | 0.006 | 0.005 | 0.018 | -0.136 |
| within_Exp1 | Exp1 | Ridge | 100 | 0.013 | 0.013 | 0.025 | -1.572 |
| within_Exp1 | Exp1 | TCN | 100 | 0.005 | 0.005 | 0.018 | -0.026 |
| within_Exp1 | Exp1 | GRU | 300 | 0.011 | 0.011 | 0.018 | -0.012 |
| within_Exp1 | Exp1 | Persistence | 300 | 0.011 | 0.011 | 0.018 | 0.000 |
| within_Exp1 | Exp1 | RBF_Ridge | 300 | 0.013 | 0.012 | 0.019 | -0.116 |
| within_Exp1 | Exp1 | Ridge | 300 | 0.029 | 0.029 | 0.031 | -1.606 |
| within_Exp1 | Exp1 | TCN | 300 | 0.011 | 0.011 | 0.018 | 0.004 |
| within_Exp2 | Exp2 | GRU | 20 | 0.021 | 0.021 | 0.020 | -0.137 |
| within_Exp2 | Exp2 | Persistence | 20 | 0.018 | 0.019 | 0.018 | 0.000 |
| within_Exp2 | Exp2 | RBF_Ridge | 20 | 0.020 | 0.020 | 0.019 | -0.067 |
| within_Exp2 | Exp2 | Ridge | 20 | 0.113 | 0.116 | 0.094 | -5.134 |
| within_Exp2 | Exp2 | TCN | 20 | 0.020 | 0.020 | 0.020 | -0.079 |
| within_Exp2 | Exp2 | GRU | 100 | 0.038 | 0.038 | 0.033 | 0.013 |
| within_Exp2 | Exp2 | Persistence | 100 | 0.038 | 0.039 | 0.031 | 0.000 |
| within_Exp2 | Exp2 | RBF_Ridge | 100 | 0.039 | 0.040 | 0.033 | -0.034 |
| within_Exp2 | Exp2 | Ridge | 100 | 0.165 | 0.171 | 0.118 | -3.333 |
| within_Exp2 | Exp2 | TCN | 100 | 0.038 | 0.039 | 0.029 | 0.012 |
| within_Exp2 | Exp2 | GRU | 300 | 0.054 | 0.054 | 0.054 | -0.009 |
| within_Exp2 | Exp2 | Persistence | 300 | 0.054 | 0.054 | 0.053 | 0.000 |
| within_Exp2 | Exp2 | RBF_Ridge | 300 | 0.054 | 0.054 | 0.054 | -0.003 |
| within_Exp2 | Exp2 | Ridge | 300 | 0.355 | 0.372 | 0.240 | -5.592 |
| within_Exp2 | Exp2 | TCN | 300 | 0.056 | 0.055 | 0.059 | -0.036 |

最佳中长 horizon 非线性行：模型 `GRU`，设置 `within_Exp2`，h=100，相对 Persistence 改善 0.013。最佳 RBF-Ridge 同类行的相对改善为 -0.003。Gate A 的结论以所有预注册中长 horizon 行为准，不从中挑选评价区间。

## 6. 预测 MAE 是否同步改善状态检测

| dataset | training_setting | model | boundary_precision | boundary_recall | boundary_f1 | stage_stable_rate | false_alarms_per_1000_windows | transition_fraction | state_fragmentation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp1 | v331_frozen | V331 | 0.200 | 1.000 | 0.333 | 0.923 | 1.755 | 0.076 | 60 |
| Exp1 | within_Exp1 | GRU | 0.200 | 0.750 | 0.316 | 0.933 | 1.317 | 0.074 | 45 |
| Exp1 | within_Exp1 | Persistence | 0.200 | 0.750 | 0.316 | 0.937 | 1.317 | 0.071 | 45 |
| Exp1 | within_Exp1 | RBF_Ridge | 0.308 | 1.000 | 0.471 | 0.936 | 0.987 | 0.070 | 39 |
| Exp1 | within_Exp1 | Ridge | 0.286 | 1.000 | 0.444 | 0.944 | 1.097 | 0.061 | 42 |
| Exp1 | within_Exp1 | TCN | 0.308 | 1.000 | 0.471 | 0.945 | 0.987 | 0.063 | 39 |
| Exp2 | Exp1_pretrain_to_Exp2 | GRU | 0.071 | 0.500 | 0.125 | 0.459 | 9.230 | 0.472 | 84 |
| Exp2 | Exp1_pretrain_to_Exp2 | Persistence | 0.125 | 0.750 | 0.214 | 0.435 | 7.455 | 0.483 | 72 |
| Exp2 | Exp1_pretrain_to_Exp2 | RBF_Ridge | 0.136 | 0.750 | 0.231 | 0.387 | 6.745 | 0.542 | 66 |
| Exp2 | Exp1_pretrain_to_Exp2 | Ridge | 0.107 | 0.750 | 0.188 | 0.379 | 8.875 | 0.518 | 84 |
| Exp2 | Exp1_pretrain_to_Exp2 | TCN | 0.125 | 0.750 | 0.214 | 0.432 | 7.455 | 0.486 | 72 |
| Exp2 | v331_frozen | V331 | 0.071 | 0.500 | 0.125 | 0.473 | 9.230 | 0.452 | 87 |
| Exp2 | within_Exp2 | GRU | 0.125 | 0.750 | 0.214 | 0.436 | 7.455 | 0.483 | 72 |
| Exp2 | within_Exp2 | Persistence | 0.125 | 0.750 | 0.214 | 0.435 | 7.455 | 0.483 | 72 |
| Exp2 | within_Exp2 | RBF_Ridge | 0.125 | 0.750 | 0.214 | 0.436 | 7.455 | 0.482 | 72 |
| Exp2 | within_Exp2 | Ridge | 0.115 | 0.750 | 0.200 | 0.428 | 8.165 | 0.484 | 78 |
| Exp2 | within_Exp2 | TCN | 0.125 | 0.750 | 0.214 | 0.436 | 7.455 | 0.483 | 72 |

同一 V3.3.1 Evidence Engine、分布漂移、漂移速度和状态机参数被固定；模型差异只进入 prediction surprise。因而预测 MAE 改善若未同时改善 Boundary F1 或稳定区误报，不能称为状态检测升级。Gate D 的逐模型审计保存在 `v332_gate_decision.json`。

## 7. 跨实验预训练的早期效应

Exp1→Exp2 的预测从 Exp2 窗口 128 开始；within_Exp2 的预注册前缀训练完成后才从窗口 1126 开始。 这说明跨实验模型具有更早的可用预测，但本轮没有在目标早期构造带标签或未来目标的 scratch 对照，因此不能把“更早可用”表述为已经证实的早期检测增益。

## 8. Gate 与研究结论

- Gate A（预测能力）：PASS
- Gate B（标签边界检测）：FAIL
- Gate C（阶段内部稳定性）：FAIL
- Gate D（模型升级价值）：PASS

因此当前证据总体为 **FAIL**。只有当标签边界召回/精度和阶段内部稳定性共同通过时，才足以支持“接触力信号能够表征磨损状态演化”的强结论；本次结果不以模型自身状态一致性或预测 MAE 代替该验证。

## 生成文件

`v332_label_provenance.json`、`v332_stage_boundaries.csv`、`v332_stage_stability_metrics.csv`、`v332_boundary_detection_metrics.csv`、`v332_boundary_match_log.csv`、`v332_ordered_state_mapping.csv`、`v332_confusion_matrix.csv`、`v332_forecast_metrics.csv`、`v332_model_state_detection_comparison.csv`、`v332_gate_decision.json`，以及四张 Stage 对齐/汇总图。
