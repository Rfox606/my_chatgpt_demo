# 部分共享动态原语与实验特异连续进程模型 v3.1：自动报告

最终科学结论：**FAIL**。所有 Gate 使用运行前冻结的阈值；未按结果改参。

## Gate 判定

| 方向 | Gate A | Gate B | Gate C | 固定 cycle 因果审计 |
|---|---:|---:|---:|---:|
| Exp1_to_Exp2 | PASS | FAIL | FAIL | PASS |
- `Exp1_to_Exp2` / gate_a: `all_preregistered_conditions_met`
- `Exp1_to_Exp2` / gate_b: `fewer_than_6_confirmed_target_segments`
- `Exp1_to_Exp2` / gate_c: `coverage_activity_ood_or_delayed_threshold_not_met`
| Exp2_to_Exp1 | PASS | FAIL | FAIL | PASS |
- `Exp2_to_Exp1` / gate_a: `all_preregistered_conditions_met`
- `Exp2_to_Exp1` / gate_b: `fewer_than_6_confirmed_target_segments`
- `Exp2_to_Exp1` / gate_c: `coverage_activity_ood_or_delayed_threshold_not_met`

## 结果与失败保留

- Gate A 原始逐预测记录：`results/gate_a_multihorizon_forecasts_v31.csv`；adapter 与负迁移记录：`results/gate_a_adapter_and_negative_transfer_v31.csv`。
- Gate B 的 BOCPD、确认片段、动态原语和目标 private-state CSV 全部保留，包括空表（它们如实表示未达到六段校准条件）。
- Gate C 连续进程、activity、initial prior、uncertainty 和 delayed-entry 收敛 CSV 全部保留。
- 归一化 LMS 修复前的两次工程失败分别保留在 `diagnostics/attempt_001_unstable_online_update.json` 与 `diagnostics/attempt_002_unstable_online_update_terminal.json`；它们不是被覆盖或删除的负结果。

## 关键决策 JSON

```json
{
  "directions": {
    "Exp1_to_Exp2": {
      "source_dataset": "Exp1",
      "target_dataset": "Exp2",
      "gate_a": {
        "status": "PASS",
        "horizon_coverage": true,
        "frozen_mae": 0.050916851942383844,
        "adapter_gated_mae": 0.019295106183992498,
        "scratch_mae": 0.01966438181799261,
        "adapter_improvement_vs_frozen": 0.6210467566646436,
        "negative_transfer_gate_active": true,
        "post_gate_adapter_to_scratch_mae_ratio": 1.0,
        "negative_transfer_ok": true,
        "reason": "all_preregistered_conditions_met"
      },
      "gate_b": {
        "status": "FAIL",
        "reason": "fewer_than_6_confirmed_target_segments",
        "source_confirmed_segments": 1,
        "target_confirmed_segments": 1,
        "source_primitive_k": 0,
        "confirmed_segments": 1,
        "single_window_kmeans_used": false
      },
      "gate_c": {
        "status": "FAIL",
        "reason": "coverage_activity_ood_or_delayed_threshold_not_met",
        "continuous_coverage": true,
        "activity_std": 0.003468604760791981,
        "delayed_entry_ok": false,
        "state_id_input_count": 0,
        "synthetic_ood": {
          "normal_mean_uncertainty": 2.0310342454934096,
          "ood_mean_uncertainty": 10.03103424549341,
          "ood_to_normal_ratio": 4.938879916845774,
          "status": "PASS",
          "state_id_used": false,
          "activity_arrays_constructed": true
        }
      },
      "prefix": {
        "status": "PASS",
        "cutoffs": {
          "3000.0": {
            "forecast": 0.0,
            "target_bocpd": 0.0,
            "continuous": 0.0,
            "private_state": 0.0,
            "status": "PASS"
          },
          "9000.0": {
            "forecast": 0.0,
            "target_bocpd": 0.0,
            "continuous": 0.0,
            "private_state": 0.0,
            "status": "PASS"
          }
        }
      },
      "source_initial_prior": 0.0003639420310793881
    },
    "Exp2_to_Exp1": {
      "source_dataset": "Exp2",
      "target_dataset": "Exp1",
      "gate_a": {
        "status": "PASS",
        "horizon_coverage": true,
        "frozen_mae": 0.008085016977637917,
        "adapter_gated_mae": 0.002990569186227506,
        "scratch_mae": 0.0029567081152658665,
        "adapter_improvement_vs_frozen": 0.6301097209196934,
        "negative_transfer_gate_active": true,
        "post_gate_adapter_to_scratch_mae_ratio": 1.0,
        "negative_transfer_ok": true,
        "reason": "all_preregistered_conditions_met"
      },
      "gate_b": {
        "status": "FAIL",
        "reason": "fewer_than_6_confirmed_target_segments",
        "source_confirmed_segments": 1,
        "target_confirmed_segments": 1,
        "source_primitive_k": 0,
        "confirmed_segments": 1,
        "single_window_kmeans_used": false
      },
      "gate_c": {
        "status": "FAIL",
        "reason": "coverage_activity_ood_or_delayed_threshold_not_met",
        "continuous_coverage": true,
        "activity_std": 0.0005979022936728809,
        "delayed_entry_ok": false,
        "state_id_input_count": 0,
        "synthetic_ood": {
          "normal_mean_uncertainty": 2.0310342454934096,
          "ood_mean_uncertainty": 10.03103424549341,
          "ood_to_normal_ratio": 4.938879916845774,
          "status": "PASS",
          "state_id_used": false,
          "activity_arrays_constructed": true
        }
      },
      "prefix": {
        "status": "PASS",
        "cutoffs": {
          "3000.0": {
            "forecast": 0.0,
            "target_bocpd": 0.0,
            "continuous": 0.0,
            "private_state": 0.0,
            "status": "PASS"
          },
          "9000.0": {
            "forecast": 0.0,
            "target_bocpd": 0.0,
            "continuous": 0.0,
            "private_state": 0.0,
            "status": "PASS"
          }
        }
      },
      "source_initial_prior": 0.0006579688162035625
    }
  },
  "status": "FAIL",
  "tests": {
    "tests": 164,
    "failures": 0,
    "errors": 0,
    "status": "PASS"
  }
}
```

连续进程模块没有接收 state-ID，也没有使用 rolling z。动态原语仅来自 BOCPD 确认的片段级描述符；目标 private-state 从不使用 source state centre 或跨实验 ID 对齐。
