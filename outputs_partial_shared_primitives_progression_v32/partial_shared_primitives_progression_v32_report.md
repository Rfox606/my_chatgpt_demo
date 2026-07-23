# 部分共享动态原语与实验特异连续进程模型 v3.2 修正版

总体判定：**FAIL**。所有预注册的 BOCPD hazard、确认阈值和最短片段参数保持不变。

## Gate 判定

| 方向 | Gate A | Gate B | Gate C | 迁移结论 |
|---|---:|---:|---:|---|
| Exp1_to_Exp2 | PASS | FAIL | FAIL | 可以讨论 Adapter 迁移收益 |
| Exp2_to_Exp1 | PASS | FAIL | FAIL | 可以讨论 Adapter 迁移收益 |

## Gate A：冻结前缀后的共同未来预测

| 方向 | 前缀 | 模型 | 加权 MAE |
|---|---:|---|---:|
| Exp1_to_Exp2 | 10% | Negative_transfer_Gated_Mixture | 0.0215836 |
| Exp1_to_Exp2 | 10% | Persistence | 0.00437062 |
| Exp1_to_Exp2 | 10% | Source_Adapter | 0.00720387 |
| Exp1_to_Exp2 | 10% | Source_Frozen | 0.0178435 |
| Exp1_to_Exp2 | 10% | Target_From_Scratch | 0.0215836 |
| Exp1_to_Exp2 | 20% | Negative_transfer_Gated_Mixture | 0.0206231 |
| Exp1_to_Exp2 | 20% | Persistence | 0.00444848 |
| Exp1_to_Exp2 | 20% | Source_Adapter | 0.00857461 |
| Exp1_to_Exp2 | 20% | Source_Frozen | 0.0180437 |
| Exp1_to_Exp2 | 20% | Target_From_Scratch | 0.0206231 |
| Exp1_to_Exp2 | 40% | Negative_transfer_Gated_Mixture | 0.0181555 |
| Exp1_to_Exp2 | 40% | Persistence | 0.00506431 |
| Exp1_to_Exp2 | 40% | Source_Adapter | 0.00862291 |
| Exp1_to_Exp2 | 40% | Source_Frozen | 0.0196578 |
| Exp1_to_Exp2 | 40% | Target_From_Scratch | 0.0181555 |
| Exp1_to_Exp2 | 60% | Negative_transfer_Gated_Mixture | 0.0204271 |
| Exp1_to_Exp2 | 60% | Persistence | 0.005651 |
| Exp1_to_Exp2 | 60% | Source_Adapter | 0.0117843 |
| Exp1_to_Exp2 | 60% | Source_Frozen | 0.0189403 |
| Exp1_to_Exp2 | 60% | Target_From_Scratch | 0.0204271 |
| Exp1_to_Exp2 | 80% | Negative_transfer_Gated_Mixture | 0.0286017 |
| Exp1_to_Exp2 | 80% | Persistence | 0.00326764 |
| Exp1_to_Exp2 | 80% | Source_Adapter | 0.0227462 |
| Exp1_to_Exp2 | 80% | Source_Frozen | 0.0136727 |
| Exp1_to_Exp2 | 80% | Target_From_Scratch | 0.0286017 |
| Exp2_to_Exp1 | 10% | Negative_transfer_Gated_Mixture | 0.00454979 |
| Exp2_to_Exp1 | 10% | Persistence | 0.00068548 |
| Exp2_to_Exp1 | 10% | Source_Adapter | 0.00209327 |
| Exp2_to_Exp1 | 10% | Source_Frozen | 0.00447362 |
| Exp2_to_Exp1 | 10% | Target_From_Scratch | 0.00454979 |
| Exp2_to_Exp1 | 20% | Negative_transfer_Gated_Mixture | 0.00208473 |
| Exp2_to_Exp1 | 20% | Persistence | 0.000698295 |
| Exp2_to_Exp1 | 20% | Source_Adapter | 0.00174318 |
| Exp2_to_Exp1 | 20% | Source_Frozen | 0.00446366 |
| Exp2_to_Exp1 | 20% | Target_From_Scratch | 0.00208473 |
| Exp2_to_Exp1 | 40% | Negative_transfer_Gated_Mixture | 0.00295556 |
| Exp2_to_Exp1 | 40% | Persistence | 0.000753765 |
| Exp2_to_Exp1 | 40% | Source_Adapter | 0.00208258 |
| Exp2_to_Exp1 | 40% | Source_Frozen | 0.00444234 |
| Exp2_to_Exp1 | 40% | Target_From_Scratch | 0.00295556 |
| Exp2_to_Exp1 | 60% | Negative_transfer_Gated_Mixture | 0.00411292 |
| Exp2_to_Exp1 | 60% | Persistence | 0.000789256 |
| Exp2_to_Exp1 | 60% | Source_Adapter | 0.00313757 |
| Exp2_to_Exp1 | 60% | Source_Frozen | 0.00420716 |
| Exp2_to_Exp1 | 60% | Target_From_Scratch | 0.00411292 |
| Exp2_to_Exp1 | 80% | Negative_transfer_Gated_Mixture | 0.00413159 |
| Exp2_to_Exp1 | 80% | Persistence | 0.000817717 |
| Exp2_to_Exp1 | 80% | Source_Adapter | 0.0020201 |
| Exp2_to_Exp1 | 80% | Source_Frozen | 0.00371273 |
| Exp2_to_Exp1 | 80% | Target_From_Scratch | 0.00413159 |

逐 horizon 的 1/5/20-step MAE 见 `gate_a_prefix_prediction_metrics_v32.csv`；Adapter 与 Scratch 分别记录，Gated Mixture 另列输出。

## Gate B：片段与目标状态

- `Exp1_to_Exp2`：源确认片段 1，目标确认片段 1，源原语 0，目标 private state 1。
  - FAIL 原因：未确认足够多个 BOCPD 片段；因此不把整条实验误称为动态原语。
- `Exp2_to_Exp1`：源确认片段 1，目标确认片段 1，源原语 0，目标 private state 1。
  - FAIL 原因：未确认足够多个 BOCPD 片段；因此不把整条实验误称为动态原语。

## Gate C：连续进程与延迟接入

- `Exp1_to_Exp2` 初始进程先验 mean=0.5、std=0.35、匹配质量=0；平台期检查=True，短尖峰保护=True，延迟接入检查=False。
- `Exp2_to_Exp1` 初始进程先验 mean=0.5、std=0.35、匹配质量=0；平台期检查=True，短尖峰保护=True，延迟接入检查=False。

## 审计结论

- 历史目录 SHA 保持：`True`（28 个文件）。
- BOCPD r=0 使用先验 Student-t predictive，增长分支使用每个 run length 的后验 Student-t predictive；输出中保留两者审计列。
- progression 不读取 cycle、state-ID 或滚动 z 值；activity 可上下波动，而 progression_increment 仅使用持续趋势、持续预测异常、确认转变和持续新颖片段。
- 若 Gate A 失败，报告保留所有数值结果，但不声称历史实验已证明可迁移。

```json
{
  "version": "v3.2",
  "status": "FAIL",
  "directions": {
    "Exp1_to_Exp2": {
      "source_dataset": "Exp1",
      "target_dataset": "Exp2",
      "gate_a": {
        "status": "PASS",
        "reason": "adapter_has_prefix_transfer_evidence",
        "prefix_coverage": true,
        "adapter_better_than_source_frozen_fraction": 0.8,
        "adapter_close_to_scratch_fraction": 1.0,
        "negative_transfer_gate_active": true,
        "mean_gated_mixture_adapter_weight": 0.052567324955116686,
        "gated_mixture_is_not_adapter": true,
        "migration_claim_permitted": true
      },
      "gate_b": {
        "status": "FAIL",
        "reason": "fewer_than_two_confirmed_target_segments",
        "source_confirmed_segments": 1,
        "target_confirmed_segments": 1,
        "source_primitive_count": 0,
        "confirmed_segments": 1,
        "private_state_count": 1,
        "private_state_can_grow_online": true,
        "source_k_imposed_on_target": false,
        "source_state_centre_used": false,
        "segment_descriptor_rows": 2,
        "single_window_primitives_used": false
      },
      "gate_c": {
        "status": "FAIL",
        "reason": "coverage_platform_spike_or_delayed_entry_check_failed",
        "continuous_coverage": true,
        "platform_increment_not_higher_than_change_increment": true,
        "short_spike_increment_guard": true,
        "delayed_entry_ok": false,
        "delayed_common_window_ok": true,
        "delayed_increment_converges": false,
        "delayed_score_converges": true,
        "delayed_uncertainty_declines": false,
        "state_id_input_count": 0,
        "synthetic_ood": {
          "normal_mean_uncertainty": 0.14282856857085702,
          "ood_mean_uncertainty": 0.8544003745317531,
          "ood_to_normal_ratio": 5.98199914121443,
          "status": "PASS",
          "state_id_used": false
        }
      },
      "initial_progression_prior": {
        "initial_progression_prior_mean": 0.5,
        "initial_progression_prior_std": 0.35,
        "initial_match_quality": 0.0
      },
      "source_frozen_parameter_sha256": "4e1bffda1c228a62a3b3b10f12de7620a46755b074a34b29920d9ff45d4d08e4"
    },
    "Exp2_to_Exp1": {
      "source_dataset": "Exp2",
      "target_dataset": "Exp1",
      "gate_a": {
        "status": "PASS",
        "reason": "adapter_has_prefix_transfer_evidence",
        "prefix_coverage": true,
        "adapter_better_than_source_frozen_fraction": 1.0,
        "adapter_close_to_scratch_fraction": 1.0,
        "negative_transfer_gate_active": true,
        "mean_gated_mixture_adapter_weight": 0.012242651106462622,
        "gated_mixture_is_not_adapter": true,
        "migration_claim_permitted": true
      },
      "gate_b": {
        "status": "FAIL",
        "reason": "fewer_than_two_confirmed_target_segments",
        "source_confirmed_segments": 1,
        "target_confirmed_segments": 1,
        "source_primitive_count": 0,
        "confirmed_segments": 1,
        "private_state_count": 1,
        "private_state_can_grow_online": true,
        "source_k_imposed_on_target": false,
        "source_state_centre_used": false,
        "segment_descriptor_rows": 2,
        "single_window_primitives_used": false
      },
      "gate_c": {
        "status": "FAIL",
        "reason": "coverage_platform_spike_or_delayed_entry_check_failed",
        "continuous_coverage": true,
        "platform_increment_not_higher_than_change_increment": true,
        "short_spike_increment_guard": true,
        "delayed_entry_ok": false,
        "delayed_common_window_ok": true,
        "delayed_increment_converges": false,
        "delayed_score_converges": true,
        "delayed_uncertainty_declines": false,
        "state_id_input_count": 0,
        "synthetic_ood": {
          "normal_mean_uncertainty": 0.14282856857085702,
          "ood_mean_uncertainty": 0.8544003745317531,
          "ood_to_normal_ratio": 5.98199914121443,
          "status": "PASS",
          "state_id_used": false
        }
      },
      "initial_progression_prior": {
        "initial_progression_prior_mean": 0.5,
        "initial_progression_prior_std": 0.35,
        "initial_match_quality": 0.0
      },
      "source_frozen_parameter_sha256": "b276d1c36cf66f1b2c3f976e9e55b64219d482285cf746451d74ff2579eb9ac6"
    }
  },
  "history_sha_unchanged": true,
  "history_file_count": 28,
  "target_final_length_used": false,
  "relative_complete_progress_used": false,
  "source_frozen_updated": false,
  "adapter_and_scratch_independent": true,
  "all_failures_retained": true,
  "pytest": {
    "tests": 175,
    "failures": 0,
    "errors": 0,
    "status": "PASS"
  }
}
```
