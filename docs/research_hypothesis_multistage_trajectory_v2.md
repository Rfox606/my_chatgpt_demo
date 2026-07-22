# 多阶段轨迹审计与状态切换原型 v2：研究假设与预注册

## 固定基线与交付顺序

- 基线提交：`3083f0799b288de2637805e34b869eab4d40fa7e`（CEAP v1）。
- 目标分支：`codex/multistage-trajectory-state-v2`。
- 执行顺序固定为 Phase A 多阶段轨迹审计、Phase B 有界/无硬限幅 adapter 消融、Phase C 状态切换在线原型。Phase C 不得替代或倒置前两阶段的结论。
- v4.5 和 CEAP v1 的历史代码与结果为只读基线；本版只增加 `multistage_trajectory_state_v2/`、v2 运行器、测试、文档与输出。

## 研究目标

检验力信号的观测轨迹是否支持局部多状态、可回访且路径依赖的描述，而非把“时间更晚”强制映射为更高的一维全程进程分数。若 Phase A 支持此假设，则在不使用标签、形貌、磨屑或未来样本的条件下，评估无硬限幅 adapter 及状态切换在线原型的未来冻结表现和稳定性。

正式在线输出为：`regime_probability`、`most_likely_regime`、`regime_duration`、`within_regime_progress`、`activity_score`、`trajectory_match_score`、`novelty_score`、`state_uncertainty`。

`within_regime_progress` 仅是当前状态内部的位置；状态编号只表示无物理命名的 `REGIME_n`，绝不表示绝对磨损等级、健康度或临床风险。

## 非目标与禁止偏移

- 不继续优化 CEAP v1 的单一全程 ranker，且不将其作为最终方案。
- 不强制任一特征、状态编号或状态内进展随时间单调增加；允许状态回访、长期平台和不同实验的不同状态序列。
- 不进行绝对磨损量比较、RUL、临床风险推断，亦不把高/低波动或高/低幅值赋予未验证的物理机制名称。
- Stage、形貌和磨屑只可在正式推理完成后的独立、版本化事后诊断中读取；不得用于变化点、K、阈值、模型参数、延迟接入或任何选择。
- 在线特征只能使用当前及历史原始窗口信息；离线对称平滑仅用于审计并明确标记 `offline_diagnostic_only=true`；不得使用未来样本。
- 不得把等价的 `F_xy` 与 `F_no_rs` 当作两个独立 ensemble 成员。
- 不得事后修改目标、参数网格、验收规则或隐藏失败、发散和反向结果。

## 固定输入、特征和审计规则

主输入固定为 `outputs_continuous_state_v45/results/window_feature_raw_v45.csv`。特征配置固定为：

- `F_core_v45`: `rx_mean, rx_absmean, rx_q05, ry_mean, ry_absmean, ry_q05, ry_p2p, rs_mean, rs_rms`
- `F_no_rs`: `rx_mean, rx_absmean, rx_q05, ry_mean, ry_absmean, ry_q05, ry_p2p`
- `F_reduced_independent`: `rx_mean, rx_q05, ry_mean, ry_q05, ry_p2p, rs_rms`

Phase A 的局部窗口固定为实验跨度的 10%、20%、30%，步长为 2.5%；滚动 ranker 的块宽固定为 20%、步长 5%。变化点固定使用 `ruptures.Pelt(model="rbf")` 与 `ruptures.Binseg(model="l2")`，并从运行前配置的 penalty 网格中选取。源原型的 K 只可在 `2,3,4,5` 中由源实验 BIC 或 silhouette 选择。

## 预注册成功标准

每个实验独立判定 Phase A：

- `PERSISTENT_DIRECTION_REVERSAL`: 同一关键特征至少 20% 局部窗口显著为正、至少 20% 显著为负，且正负方向块均持续超过总跨度 5%。
- `RANK_DIRECTION_INSTABILITY`: 相邻滚动块余弦中位数低于 0.5，或至少 20% 相邻块余弦小于 0，或主要系数在多个持续块稳定反转。
- `RECURRENT_OBSERVATION_STATE`: 远时间近邻比例高于时间块置换与随机游走 null 的 95% 分位数。
- `REPRODUCIBLE_CHANGE_POINT`: 同一位置由至少两种方法、两种特征配置支持，出现在至少 60% bootstrap 中，且聚合位置误差不超过总跨度 3%。

Phase A 满足 3/4 为 PASS，2/4 为 QUALIFIED PASS，少于 2/4 为 FAIL。

Phase B 只比较以下预声明组：`Bounded_Baseline`（norm 0.55，step 0.08，原 L2）、`Unbounded_L2`（无模型意义硬限幅、原 L2）和 `Unbounded_WeakL2`（无硬限幅、原 L2 的 25%）。无硬限幅只允许非有限值、norm 超过 100 或 loss 溢出的数值安全中止。每个方向、每个延迟接入点采用 20%、40%、60%、80% 前缀更新后冻结未来。去硬限幅仅在至少一个方向未来冻结表现改善、参数不发散、输出不饱和、不仅时间 AUC 提升、且同一未来窗口的多前缀差异下降时为 PASS。

Phase C 必须通过合成高—低—高—低场景、控制短尖峰误报、允许状态回访和 `UNKNOWN_NOVEL`，并在真实数据中至少一个方向相对 source-only 或 target-local 有改善；主指标为未来冻结 NLL/重构误差、后验收敛、边界稳定性与变化点一致性，不以时间对 AUC 为主。

工程 PASS 还要求完整 pytest、严格前缀因果、无标签泄漏、源结构冻结、无硬限幅数值安全、输出可复现及历史版本未修改。最终规则固定：三部分均通过为 PASS；审计通过而模型仅单向或有限改善为 QUALIFIED PASS；审计不支持、模型仅学习时间、发散或状态坍塌为 FAIL。

## 结果保留

所有方向（Exp1→Exp2 和 Exp2→Exp1）、所有延迟接入点、三组 adapter、失败诊断、不可用的事后标签诊断与完整 pytest 结果均写入版本化输出和报告，不进行筛选性汇报。

