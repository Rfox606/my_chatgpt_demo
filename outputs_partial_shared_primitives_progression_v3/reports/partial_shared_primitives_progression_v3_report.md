# 閮ㄥ垎鍏变韩鍔ㄦ€佸師璇笌瀹為獙鐗瑰紓杩炵画杩涚▼妯″瀷 v3锛氳嚜鍔ㄦ姤鍛?

鏈€缁堢瀛︾粨璁猴細**FAIL**銆傞娉ㄥ唽闃堝€兼湭閫氳繃鏃讹紝鏈姤鍛婁繚鐣欏け璐ョ粨鏋滀笖涓嶉噸璋冨弬鏁般€?

## 棰勬敞鍐岄獙鏀?

| 椤圭洰 | 缁撴灉 |
|---|---|
| shared_causal_prediction | FAIL |
| dynamic_primitive_bootstrap | FAIL |
| experiment_specific_local_states | PASS |
| independent_continuous_progression | PASS |
| prefix_causality | PASS |
| synthetic_state_revisit | PASS |
| label_morphology_debris_future_input | PASS |
| no_fixed_five_classification | PASS |
| no_global_time_ranker | PASS |

## 瑙ｉ噴杈圭晫

鍏变韩閮ㄥ垎浠呬负鍥犳灉棰勬祴鍙傛暟鍜屽姩鎬佸師璇瓧鍏革紱Exp1/Exp2 鐨?K銆乻tate centre銆乻tate-ID銆佽涔夊拰璺緞鍧囩嫭绔嬨€傝繛缁繘绋嬪垎鏁颁笉璇诲彇鐘舵€佽緭鍑猴紝涓斾笉鏄叏绋嬫椂闂存帓鍚嶃€佸浐瀹氫簲鍒嗙被鎴栫粷瀵圭（鎹熼噺銆?

## 鏁板€肩粨鏋?

```json
{
  "status": "FAIL",
  "predictor_evaluation": [
    {
      "dataset": "Exp1",
      "heldout_start_fraction": 0.7,
      "heldout_window_count": 2735,
      "shared_predictor_huber_mae": 0.027568225226210413,
      "persistence_mae": 0.025788844374826436,
      "mae_ratio_to_persistence": 1.0689980840367126,
      "relative_mae_improvement": -0.06899808403671259
    },
    {
      "dataset": "Exp2",
      "heldout_start_fraction": 0.7,
      "heldout_window_count": 846,
      "shared_predictor_huber_mae": 0.025661748845935764,
      "persistence_mae": 0.02715398159336728,
      "mae_ratio_to_persistence": 0.9450455270325434,
      "relative_mae_improvement": 0.05495447296745659
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
      "std": 3.080302266721941
    },
    {
      "dataset": "Exp2",
      "count": 2817,
      "std": 7.945582858193073
    }
  ],
  "synthetic": {
    "returns_are_allowed": true,
    "no_monotonic_state_constraint": true
  },
  "selection_rules_locked_before_run": true,
  "predictor_hash": "d6333c8d578e55f94e3eaa8c15f0dd1a6404ad434730016ac220377fe7133ac0",
  "engineering_test_status": "PASS",
  "retained_failure_runs": [
    "retained_failures_partial_shared_primitives_v3/cutoff_rounding_audit_failure",
    "retained_failures_partial_shared_primitives_v3/relative_progress_rounding_boundary_failure"
  ],
  "overall_status_after_tests": "FAIL"
}
```

## 宸蹭繚鐣欑殑瀹炵幇绾уけ璐ュ揩鐓?

- `retained_failures_partial_shared_primitives_v3/cutoff_rounding_audit_failure`
- `retained_failures_partial_shared_primitives_v3/relative_progress_rounding_boundary_failure`

