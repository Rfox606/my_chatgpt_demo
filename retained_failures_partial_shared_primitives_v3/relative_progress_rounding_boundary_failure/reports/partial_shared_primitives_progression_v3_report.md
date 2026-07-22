# 部分共享动态原语与实验特异连续进程模型 v3：自动报告

最终科学结论：**FAIL**。预注册阈值未通过时，本报告保留失败结果且不重调参数。

## 预注册验收

| 项目 | 结果 |
|---|---|
| shared_causal_prediction | FAIL |
| dynamic_primitive_bootstrap | FAIL |
| experiment_specific_local_states | PASS |
| independent_continuous_progression | PASS |
| prefix_causality | FAIL |
| synthetic_state_revisit | PASS |
| label_morphology_debris_future_input | PASS |
| fixed_five_classification | FAIL |
| global_time_ranker_used | FAIL |

## 解释边界

共享部分仅为因果预测参数和动态原语字典；Exp1/Exp2 的 K、state centre、state-ID、语义和路径均独立。连续进程分数不读取状态输出，且不是全程时间排名、固定五分类或绝对磨损量。

## 数值结果

```json
{
  "status": "FAIL",
  "predictor_evaluation": [
    {
      "dataset": "Exp1",
      "heldout_start_fraction": 0.7,
      "heldout_window_count": 2735,
      "shared_predictor_huber_mae": 0.027568146754162914,
      "persistence_mae": 0.025788844374826436,
      "mae_ratio_to_persistence": 1.0689950411687827,
      "relative_mae_improvement": -0.06899504116878274
    },
    {
      "dataset": "Exp2",
      "heldout_start_fraction": 0.7,
      "heldout_window_count": 846,
      "shared_predictor_huber_mae": 0.025661764653909513,
      "persistence_mae": 0.02715398159336728,
      "mae_ratio_to_persistence": 0.9450461091929789,
      "relative_mae_improvement": 0.054953890807021066
    }
  ],
  "primitive_bootstrap_ari_median": {
    "Exp1": 1.0,
    "Exp2": 1.0
  },
  "primitive_effective_counts": {
    "Exp1": 1,
    "Exp2": 1
  },
  "state_models": [
    {
      "dataset": "Exp1",
      "selected_local_k": 6,
      "state_centre_provenance": "local_experiment_only",
      "state_id_alignment_performed": false,
      "source_state_centre_used": false,
      "state_descriptor_columns": "shared_z0|shared_z1|shared_z2|shared_z3|shared_z4|shared_z5|forecast_mae|forecast_activity|primitive_p0|primitive_p1"
    },
    {
      "dataset": "Exp2",
      "selected_local_k": 2,
      "state_centre_provenance": "local_experiment_only",
      "state_id_alignment_performed": false,
      "source_state_centre_used": false,
      "state_descriptor_columns": "shared_z0|shared_z1|shared_z2|shared_z3|shared_z4|shared_z5|forecast_mae|forecast_activity|primitive_p0|primitive_p1"
    }
  ],
  "continuous_by_dataset": [
    {
      "dataset": "Exp1",
      "count": 9115,
      "std": 3.080445678947209
    },
    {
      "dataset": "Exp2",
      "count": 2817,
      "std": 7.945622623364437
    }
  ],
  "synthetic": {
    "returns_are_allowed": true,
    "no_monotonic_state_constraint": true
  },
  "selection_rules_locked_before_run": true,
  "predictor_hash": "30a15652bac3a26786cd512315eef7ce10d0e30205d000c3b9b4a36fa8f67d76"
}
```
