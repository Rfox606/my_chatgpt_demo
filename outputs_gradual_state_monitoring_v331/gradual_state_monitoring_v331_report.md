# 渐变状态监测 V3.3.1 审计修正版报告

## 运行边界

V3.3.1 建立在 V3.3 提交 `0900f577e50bd7e00e5c5cc515366c7a2180663f` 之上。在线部分仍只使用六个批准的接触力窗口特征。数据组边界、实验分段映射与形貌元数据只在 `v331_event_boundary_audit.csv` 的离线解释中使用，绝不进入描述符、预测、证据或状态机。

## 审计结论

1. 冻结 scaler 后，预测误差与状态数量确有变化：Exp1: Slow MAE 0.00415106→0.00390485，Fast MAE 0.0222182→0.0233993，状态切换 63→60；Exp2: Slow MAE 0.0311798→0.0359158，Fast MAE 0.10152→0.100585，状态切换 69→87。冻结模式在第 128 个已到达窗口后固定坐标，随后 Slow/Fast RLS 才开始训练和产生有效预测。
2. 证据重要性（以完整重跑后相对 full 状态路径的变化衡量）：Exp1 最敏感于 `本组删除任一证据后的完整重跑状态路径均未改变；当前阈值下不能区分单项重要性`；Exp2 最敏感于 `本组删除任一证据后的完整重跑状态路径均未改变；当前阈值下不能区分单项重要性`。四种证据配置均重新运行了 Evidence Engine、状态机、状态事件与延迟接入，而非复用 full 标签。
3. Exp2 的 TRANSITION 占比为 0.452；是否下降应与 V3.3 的历史输出并列解释，不能仅凭本版本宣称改善。
4. 窗口重叠稳定性：Exp1 50%=0.851, 0%=0.761; Exp2 50%=0.468, 0%=0.471。Gate B2 对 50% 和 0% 重叠率使用预先固定的阈值。
5. 完整 Slow/Fast 延迟接入的收敛：Exp1: agreement=0.999, IoU=0.994, local-ID difference=0; Exp2: agreement=0.840, IoU=0.738, local-ID difference=1。PASS 不再只取最后 100 个窗口，而同时检查状态一致率、TRANSITION IoU 和本地状态数量差异。
6. 历史状态复用：`v331_state_reuse_log.csv` 记录每个确认新稳定状态是 created 还是 reused；单元验证覆盖 `1 → 2 → 3 → 2`。
7. 状态事件的边界接近比例为 0.156；这些事件均只标记、未删除，详见 `v331_event_boundary_audit.csv`。
8. Gate B：FAIL（B1=PASS，B2=FAIL）；Gate C：PASS。所有 FAIL 保留在 `v331_gate_decision.json`。
9. 当前仍不具备“真实磨损状态”的充分解释能力：在线模型只表征接触力统计轨迹，本地状态不等同于跨实验可比的磨损机理或形貌状态。

## 预测与模型分离信号

RBF-Ridge 是否优于线性 Ridge，应以 `v331_forecast_metrics.csv` 按 dataset/horizon 比较；Gate A 仍要求中长 horizon 超越 Persistence。`fast_slow_prediction_difference` 始终单独保存为 `model_divergence`。默认三证据结果不加入它；完整 `with_model_divergence` / `without_model_divergence` 重放的 TRANSITION 占比分别为：Exp1: without=0.076, with=0.077; Exp2: without=0.452, with=0.452。

## Gate

* Gate A：FAIL
* Gate B1：PASS
* Gate B2：FAIL
* Gate C：PASS
* 总体：FAIL
