# 渐变状态监测模型 V3.3 报告

## 运行边界

本次运行只读取 `rx_mean`、`rx_q05`、`ry_mean`、`ry_q05`、`ry_p2p`、`rs_rms` 六个窗口特征及窗口标识和已到达的物理循环号。没有读取 Stage、总长度比例、未来窗口、形貌、磨屑或状态标签。描述符、线上尺度、预测器更新和状态决策均为因果流程；完整前缀复跑检验结果见 `v33_prefix_causality.json`。

## 结论

1. 非线性 RBF-Ridge 相对 Ridge 的价值：是。应以 `v33_forecast_metrics.csv` 中每个数据集、horizon 的实际 MAE 比较为准，不能把单一全程加权 MAE 当作状态检测成功依据。
2. 长 horizon 是否削弱 Persistence 优势：否；至少一个中长 horizon 的学习模型优于 Persistence：否。
3. prediction surprise 能否区分稳定期和变化期：是。分开保存的三个证据可在 `v33_change_evidence.csv` 与 `v33_change_detection_metrics.csv` 审计。
4. 缓慢漂移是否形成持续 TRANSITION：PASS。合成斜坡的最大检测宽度见 `v33_synthetic_test_results.csv`。
5. 短时异常是否会被误判为新状态：PASS；验收要求是出现 TEMPORARY_DISTURBANCE 且不形成 NEW_STABLE。
6. 延迟接入共同未来是否逐渐一致：末端 100 个共同窗口的平均状态一致率为 1.000（无可比窗口时为 NaN），逐项结果见 `v33_delayed_entry_evaluation.csv`。
7. 当前结果是否足以支持跨实验共享动力学：否。虽然 `v33_source_initialisation_metrics.csv` 审计了 Exp1→Exp2 的早期预测初始化，但 V3.3 有意只创建和复用实验内 `local_state_id`，未训练或验证跨实验共享状态动力学。

## Gate

* Gate A（稳定动力学建模）：FAIL
* Gate B（渐变状态检测）：PASS
* Gate C（新状态确认与线上一致性）：PASS
* 总体：FAIL

所有阈值、权重、随机种子、慢/快模型遗忘因子均固定在 `v33_config.json`；`v33_ablation_results.csv` 同时保留三证据消融、20 窗口特征在 75%/50%/0% 重叠率下的因果重放，以及三个 forecast horizon 的模型对比。Gate 的 FAIL 会被保留，不会通过降低阈值来制造状态数量。
