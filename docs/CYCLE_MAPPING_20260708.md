# Cycle Mapping 2026-07-08

本文件记录本轮物理验证候选窗口使用的“有效周期 -> 实际实验周期”映射关系。

## 映射原则

- 原始候选表中的 `center_cycle` 保留为有效周期口径。
- 新增 `center_cycle_effective` 与 `center_cycle_actual`，后续物理闭环验证优先使用 `center_cycle_actual`。
- 每个阶段内部采用线性映射：

```text
actual = actual_start
       + (effective - effective_start)
       * (actual_end - actual_start)
       / (effective_end - effective_start)
```

## EXP1

| stage | effective_start | effective_end | actual_start | actual_end |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 7575 | 1 | 8000 |
| 2 | 7575 | 21125 | 8000 | 24000 |
| 3 | 21125 | 27840 | 24000 | 32000 |
| 4 | 27840 | 34600 | 32000 | 40000 |
| 5 | 34600 | 45590 | 40000 | 53000 |

## EXP2

EXP2 前 500 个实际周期为 NaN，因此有效周期 1 对应实际周期 501。

| stage | effective_start | effective_end | actual_start | actual_end |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 3005 | 501 | 5500 |
| 2 | 3005 | 6005 | 5500 | 10500 |
| 3 | 6005 | 8705 | 10500 | 15000 |
| 4 | 8705 | 11705 | 15000 | 20000 |
| 5 | 11705 | 14100 | 20000 | 24000 |

## 输出文件

- `outputs_physical_validation_candidates_v1/results/cycle_mapping_table.csv`
- `outputs_physical_validation_candidates_v1/configs/cycle_mapping_config.json`
- `outputs_physical_validation_candidates_v1/results/physical_validation_candidates.csv`
