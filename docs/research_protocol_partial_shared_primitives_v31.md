# Codex任务书：部分共享动态原语与实验特异连续进程模型 v3.1（完整化与方法修正）

## 总目标、术语与非目标

v3.1 只检验三个按顺序执行的 Gate。共享的是**严格离线源实验**训练的因果多步预测参数与源片段动态原语先验；目标实验只能从到达的自身窗口中形成 adapter、BOCPD 片段、私有状态与连续进程。每次源→目标运行独立进行，绝不把 Exp1 与 Exp2 联合在线训练成同一个共享主模型。

- **源→目标**：选择一实验为 source，使用其前 1,600 个窗口训练；目标只在自己的数据到达后预测和更新。方向必须双向报告。
- **多步预测**：在已到达的 32 个历史窗口上，预测未来 1、4、16 个窗口的九个力特征差分。预测误差，而非全程时间、Stage 或磨损量，是 Gate A 的主目标。
- **动态原语**：由 BOCPD 确认片段的片段级（均值、斜率、创新残差、活动）描述符聚类产生；任何单窗口 KMeans 均不称为动态原语。
- **私有状态**：目标实验只用已确认的自身片段按 BIC 选择 K、中心、语义和路径；它不会接收 source state centre/variance/transition，也不会对齐 state-ID。
- **连续进程**：独立于 state-ID 的累积因果证据，显式输出 cumulative progression、activity、initial prior、uncertainty 与 delayed-entry 收敛；它不是滚动 z 值、固定分类或绝对磨损量。

本任务不恢复五阶段分类、全程时间 rank、相对完整进度、绝对磨损比较或只给变化点而不给连续进程。

## 绝对禁止项

1. Exp1 与 Exp2 联合在线训练同一个共享主模型；
2. 读取目标最终长度、相对完整进度或任何未来目标窗口；
3. 用在线 SVD 产生会旋转的 `shared_z`；
4. 将单窗口 KMeans 称为动态原语；
5. 将滚动 z 值称为连续进程；
6. 固定目标状态数、对齐 state-ID 或复制 source state centre；
7. 省略 adapter、BOCPD、private state、delayed entry 或 uncertainty；
8. 使用 Stage、形貌、磨屑、绝对磨损量或未来目标数据选参；
9. 在运行后修改下述阈值；
10. 以简化或占位实现代替各 Gate 的完整输出。

## 运行前冻结的配置

输入固定为 `outputs_continuous_state_v45/results/window_feature_raw_v45.csv` 的九个 `F_core_v45` 力特征。随机种子=20260722。三步 horizon=`[1,4,16]`、历史=32、source train windows=1600、ridge=0.25、adapter learning rate=0.06、adapter warmup=128、adapter delayed-label 更新（每一 horizon 的真值到达后才更新）、负迁移阈值=相对 scratch 误差高 3%、连续确认=3 次。

BOCPD 使用 Student-t predictive、hazard=1/160、最大 run length=256、确认 posterior=0.65、确认连续=3、最小片段=16 windows。片段原语和私有状态的 BIC candidates 都是 `[2,3,4,5,6]`，最小群占比=3%，目标私有状态固定用其**前 6 个已确认片段**校准（不足时明确 FAIL，不用未来补齐）。

连续进程初始 prior 取 source 前 400 个可预测创新的固定分位尺度；increment=`0.65*log1p(adapter innovation energy)+0.35*log1p(activity energy)`，只累积非负的当期因果 increment；uncertainty 由多步残差离散度、adapter support deficit、BOCPD run-length entropy 构成。delayed entry 固定为 Exp1 cycles=`[0,8000,16000,24000]`、Exp2 cycles=`[0,3000,6000,9000]`，比较 latest-entry 后的固定 200 个共同到达窗口，不读取最终长度。

## 预注册 Gate 接受标准

每个方向独立给出 Gate A/B/C 的 PASS/FAIL 和原因；总 PASS 要求双向每一 Gate 都通过。

| Gate | 通过条件 |
|---|---|
| A：非对称迁移 | 三个 horizon 全部有非零可评分覆盖；Source+Adapter 相对 Source Frozen 的平均 MAE 改善至少 1%；若 adapter 相对 Target From Scratch 连续三次高 3%，负迁移 gate 必须启用，且启用后 adapter 的平均误差不高于 scratch 的 1.05 倍。 |
| B：BOCPD/原语/私有状态 | source 和 target 各至少 3 个确认片段；source 片段原语有效数至少 2；目标 K 由其前 6 个确认片段独立 BIC 选择（2--6），所有中心 provenance=`target_confirmed_segments_only`，无跨实验 ID 映射；原语输入行数必须等于确认片段数而非窗口数。 |
| C：连续进程/不确定性/延迟接入 | 每个目标窗口都有有限 cumulative progression、activity、initial prior、uncertainty；activity 标准差 >`1e-6`；合成 OOD 的 uncertainty 高于正常轨迹至少 20%；所有预注册 delayed entries 有 200 个共同到达窗口，latest-entry 后的 pairwise increment NRMSE <=0.50；连续模块读入 state-ID 次数=0。 |
| 通用 | 两个预注册 prefix cutoff=`[0.35,0.60]` 的预测、BOCPD、私有状态与连续输出最大差 `<=1e-12`；Stage/形貌/磨屑/绝对磨损/未来目标/目标最终长度读取次数=0；全部 pytest 通过。 |

每次正式运行后不得为提高结果重调参数。任何 Gate 失败、负迁移触发、校准不足或工程失败都必须完整保留到结果目录和自动报告。

