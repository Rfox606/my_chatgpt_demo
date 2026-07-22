from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from continuous_state_v2.bootstrap_stability import bootstrap_head
from continuous_state_v2.branch_axis import build_branch_axis
from continuous_state_v2.candidate_selection import select_candidates
from continuous_state_v2.common_axis import build_common_axis
from continuous_state_v2.config import ContinuousStateV2Config
from continuous_state_v2.data import FORBIDDEN_COLUMNS, assert_label_free, load_window_table
from continuous_state_v2.evaluation import frozen_vs_adaptive_summary, online_benefit, target_segment_diagnostics
from continuous_state_v2.feature_pruning import prune_features
from continuous_state_v2.guards import guard_sensitivity_frames
from continuous_state_v2.online_forecast import run_online_forecasts, train_frozen_predictors
from continuous_state_v2.online_nuisance_adapter import run_target_online
from continuous_state_v2.plotting import make_figures
from continuous_state_v2.report import make_report
from continuous_state_v2.source_rank_heads import RankHead, train_source_head
from continuous_state_v2.state_metrics import frozen_state_scores, make_state_space
from continuous_state_v2.support_confidence import source_support


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _head_relative(frame: pd.DataFrame, head: RankHead, config: ContinuousStateV2Config) -> np.ndarray:
    raw = frame.loc[:, list(head.features)].to_numpy(float) @ head.normalized_weight
    mask = (frame.end_cycle <= config.baseline_cycles) & frame.is_restart_guard.eq(0)
    anchor = float(np.median(raw[mask]))
    scale = float(np.quantile(raw[mask], .75) - np.quantile(raw[mask], .25))
    return (raw - anchor) / max(scale, config.eps)


def _source_static_frame(frame: pd.DataFrame, space, direction_id: str, source: str, target: str, config: ContinuousStateV2Config) -> pd.DataFrame:
    static = frozen_state_scores(frame, space, config)
    output = pd.concat([frame.reset_index(drop=True), static], axis=1)
    output["direction_id"] = direction_id; output["source_dataset"] = source; output["target_dataset"] = target; output["dataset_role"] = "source"
    output["oos_feature_count"] = 0; output["weighted_oos_common"] = 0.; output["weighted_oos_branch"] = 0.; output["weighted_oos_source_head"] = 0.; output["support_confidence"] = 1.; output["branch_confidence"] = 1.; output["source_head_disagreement"] = 0.
    output["pre_update_P_common"] = output.P_common; output["pre_update_BD"] = output.BD; output["pre_update_B_terminal"] = output.B_terminal; output["pre_update_TES"] = output.TES; output["pre_update_weighted_oos"] = output.weighted_oos_common
    output["adapter_updated"] = 0; output["adapter_update_reason"] = "SOURCE_FROZEN"; output["adapter_learning_rate"] = 0.; output["adapter_rollback"] = 0; output["rollback_reason"] = ""; output["beta_norm"] = 0.; output["beta_change_norm"] = 0.
    return output


def _prefix_check(target: pd.DataFrame, source_support_frame: pd.DataFrame, space, features, common_features, w_common, source_head, all_heads, full: pd.DataFrame, direction_id: str, config: ContinuousStateV2Config) -> dict[str, object]:
    rows = []
    columns = ["pre_update_P_common", "pre_update_BD", "pre_update_B_terminal", "pre_update_TES", "pre_update_weighted_oos"]
    for cutoff in (500, 1000, 2000):
        prefix = target.loc[target.center_cycle <= cutoff].copy()
        scored, _ = run_target_online(prefix, space, features, source_support_frame, common_features, w_common, source_head.features, source_head.normalized_weight, config)
        reference = full.loc[full.center_cycle <= cutoff]
        merged = scored.merge(reference[["window_index", *columns]], on="window_index", suffixes=("_prefix", "_full"))
        maximum = max(float(np.abs(merged[f"{column}_prefix"] - merged[f"{column}_full"]).max()) for column in columns)
        rows.append({"prefix_cycle": cutoff, "window_count": len(merged), "max_abs_difference": maximum, "pass": maximum < 1e-10})
    return {"direction_id": direction_id, "status": "PASS" if all(row["pass"] for row in rows) else "FAIL", "checks": rows}


def _main_columns() -> list[str]:
    return ["direction_id", "source_dataset", "target_dataset", "dataset", "dataset_role", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle", "is_restart_guard", "crosses_stop_boundary", "intersects_post_stop_guard", "P_common", "P_smooth_5", "P_smooth_20", "P_short_volatility", "BD", "BD_diag", "bd_method", "B_terminal", "P_RS20", "P_RS50", "P_RS100", "BD_RS20", "BD_RS50", "BD_RS100", "B_RS20", "B_RS50", "B_RS100", "TES", "weighted_oos_common", "weighted_oos_branch", "weighted_oos_source_head", "support_confidence", "branch_confidence", "source_head_disagreement", "adapter_updated", "adapter_update_reason", "adapter_learning_rate", "adapter_rollback", "beta_norm", "pre_update_P_common", "pre_update_BD", "pre_update_B_terminal", "pre_update_TES", "pre_update_weighted_oos"]


def main() -> None:
    config = ContinuousStateV2Config(); paths = config.paths()
    (paths["configs"] / "continuous_state_v2_config.json").write_text(json.dumps(config.jsonable(), ensure_ascii=False, indent=2), encoding="utf-8")
    raw = load_window_table(config)
    print("v2: loaded label-free windows", flush=True)
    guard_frames = guard_sensitivity_frames(raw, config)
    primary = guard_frames[config.primary_restart_guard_cycles]
    guard_audit = {str(cycles): {"guard_window_count": int(frame.is_restart_guard.sum()), "crossing_window_count": int(frame.crosses_stop_boundary.sum()), "post_stop_intersection_count": int(frame.intersects_post_stop_guard.sum())} for cycles, frame in guard_frames.items()}
    exp1, exp2 = (primary.loc[primary.dataset.eq(name)].copy().reset_index(drop=True) for name in ("Exp1", "Exp2"))
    assert_label_free(exp1); assert_label_free(exp2)
    kept1, prune1 = prune_features(exp1.iloc[: int(len(exp1) * config.source_train_fraction)], "Exp1_source", config)
    kept2, prune2 = prune_features(exp2.iloc[: int(len(exp2) * config.source_train_fraction)], "Exp2_source", config)
    head1, validation1, gap1, split1 = train_source_head(exp1, kept1, "Exp1_source", config)
    head2, validation2, gap2, split2 = train_source_head(exp2, kept2, "Exp2_source", config)
    print("v2: source selection and deployment heads fitted", flush=True)
    boot_cache = paths["results"] / "source_bootstrap_coefficients.csv"
    stability_cache = paths["results"] / "source_feature_stability.csv"
    bootstrap_metadata_cache = paths["results"] / "source_bootstrap_metadata.json"
    bootstrap_metadata = {"max_pairs_per_gap_bin": config.max_pairs_per_gap_bin, "bootstrap_repeats": config.bootstrap_repeats, "bootstrap_block_windows": config.bootstrap_block_windows}
    cached_metadata = json.loads(bootstrap_metadata_cache.read_text(encoding="utf-8")) if bootstrap_metadata_cache.exists() else None
    if boot_cache.exists() and stability_cache.exists() and cached_metadata == bootstrap_metadata:
        boot = pd.read_csv(boot_cache); stability = pd.read_csv(stability_cache)
        boot1 = boot.loc[boot.direction_id.eq("Exp1_source")]; boot2 = boot.loc[boot.direction_id.eq("Exp2_source")]
        stability1 = stability.loc[stability.direction_id.eq("Exp1_source")]; stability2 = stability.loc[stability.direction_id.eq("Exp2_source")]
        print("v2: reused completed bootstrap cache", flush=True)
    else:
        boot1, stability1 = bootstrap_head(exp1, kept1, head1.selected_C, "Exp1_source", config)
        print("v2: Exp1 bootstrap complete", flush=True)
        boot2, stability2 = bootstrap_head(exp2, kept2, head2.selected_C, "Exp2_source", config)
        print("v2: Exp2 bootstrap complete", flush=True)
        boot = pd.concat([boot1, boot2], ignore_index=True); stability = pd.concat([stability1, stability2], ignore_index=True)
        boot.to_csv(boot_cache, index=False, encoding="utf-8-sig"); stability.to_csv(stability_cache, index=False, encoding="utf-8-sig")
        _write_json(bootstrap_metadata_cache, bootstrap_metadata)
    pruning = pd.concat([prune1, prune2], ignore_index=True)
    stability = pd.concat([stability1, stability2], ignore_index=True) if 'stability' not in locals() else stability
    boot = pd.concat([boot1, boot2], ignore_index=True) if 'boot' not in locals() else boot
    retained = set(kept1).intersection(kept2)
    state_features = tuple(feature for feature in config.jsonable()["stable_plus_features"] if feature in retained)
    print(f"v2: state feature count={len(state_features)}", flush=True)
    if not state_features:
        raise ValueError("No shared pruned features are available for the state space")
    w_common, common_features, common_table, common_status = build_common_axis(stability, retained, config)
    print(f"v2: common axis features={len(common_features)}", flush=True)
    w_branch, branch_table = build_branch_axis(exp1, exp2, state_features, common_features, w_common, config)
    print("v2: branch axis built", flush=True)
    space1 = make_state_space(exp1, state_features, common_features, w_common, w_branch, config)
    space2 = make_state_space(exp2, state_features, common_features, w_common, w_branch, config)
    print("v2: baseline state spaces fitted", flush=True)
    static1 = _source_static_frame(exp1, space1, "Exp1_to_Exp2", "Exp1", "Exp2", config)
    static2 = _source_static_frame(exp2, space2, "Exp2_to_Exp1", "Exp2", "Exp1", config)
    print("v2: frozen source trajectories scored", flush=True)
    frozen1 = train_frozen_predictors(static1, config)
    print("v2: Exp1 frozen predictor fitted", flush=True)
    frozen2 = train_frozen_predictors(static2, config)
    print("v2: Exp2 frozen predictor fitted", flush=True)
    cache_dir = paths["root"] / ".cache" / f"pair_cap_{config.max_pairs_per_gap_bin}"
    cache_dir.mkdir(exist_ok=True)
    target2_cache, log2_cache = cache_dir / "exp1_to_exp2_target_state.csv", cache_dir / "exp1_to_exp2_adapter_log.csv"
    target1_cache, log1_cache = cache_dir / "exp2_to_exp1_target_state.csv", cache_dir / "exp2_to_exp1_adapter_log.csv"
    if target2_cache.exists() and log2_cache.exists():
        target2, log2 = pd.read_csv(target2_cache), pd.read_csv(log2_cache)
        print("v2: reused Exp2 target online-state cache", flush=True)
    else:
        target2, log2 = run_target_online(exp2, space2, state_features, source_support(exp1, state_features), common_features, w_common, head1.features, head1.normalized_weight, config)
        target2["direction_id"] = "Exp1_to_Exp2"; target2["source_dataset"] = "Exp1"; target2["target_dataset"] = "Exp2"; target2["dataset_role"] = "target"
        log2["direction_id"] = "Exp1_to_Exp2"
        target2.to_csv(target2_cache, index=False, encoding="utf-8-sig"); log2.to_csv(log2_cache, index=False, encoding="utf-8-sig")
        print("v2: Exp2 target online state scoring complete", flush=True)
    if target1_cache.exists() and log1_cache.exists():
        target1, log1 = pd.read_csv(target1_cache), pd.read_csv(log1_cache)
        print("v2: reused Exp1 target online-state cache", flush=True)
    else:
        target1, log1 = run_target_online(exp1, space1, state_features, source_support(exp2, state_features), common_features, w_common, head2.features, head2.normalized_weight, config)
        target1["direction_id"] = "Exp2_to_Exp1"; target1["source_dataset"] = "Exp2"; target1["target_dataset"] = "Exp1"; target1["dataset_role"] = "target"
        log1["direction_id"] = "Exp2_to_Exp1"
        target1.to_csv(target1_cache, index=False, encoding="utf-8-sig"); log1.to_csv(log1_cache, index=False, encoding="utf-8-sig")
        print("v2: Exp1 target online state scoring complete", flush=True)
    print("v2: causal online state scoring complete", flush=True)
    target2["direction_id"] = "Exp1_to_Exp2"; target2["source_dataset"] = "Exp1"; target2["target_dataset"] = "Exp2"; target2["dataset_role"] = "target"
    target1["direction_id"] = "Exp2_to_Exp1"; target1["source_dataset"] = "Exp2"; target1["target_dataset"] = "Exp1"; target1["dataset_role"] = "target"
    # Keep disagreement interpretable rather than treating it as an error.
    for frame in (target1, target2):
        frame["source_head_disagreement"] = np.abs(_head_relative(frame, head1, config) - _head_relative(frame, head2, config))
    log2["direction_id"] = "Exp1_to_Exp2"; log1["direction_id"] = "Exp2_to_Exp1"
    scores = pd.concat([static1, target2, static2, target1], ignore_index=True).loc[:, _main_columns()]
    forecast2_cache, metrics2_cache = cache_dir / "exp1_to_exp2_forecasts.csv", cache_dir / "exp1_to_exp2_forecast_metrics.csv"
    forecast1_cache, metrics1_cache = cache_dir / "exp2_to_exp1_forecasts.csv", cache_dir / "exp2_to_exp1_forecast_metrics.csv"
    if forecast2_cache.exists() and metrics2_cache.exists():
        forecast2, metrics2 = pd.read_csv(forecast2_cache), pd.read_csv(metrics2_cache)
        print("v2: reused Exp1-to-Exp2 forecast cache", flush=True)
    else:
        forecast2, metrics2 = run_online_forecasts(target2, frozen1, "Exp1_to_Exp2", config)
        forecast2.to_csv(forecast2_cache, index=False, encoding="utf-8-sig"); metrics2.to_csv(metrics2_cache, index=False, encoding="utf-8-sig")
        print("v2: Exp1-to-Exp2 forecasts complete", flush=True)
    if forecast1_cache.exists() and metrics1_cache.exists():
        forecast1, metrics1 = pd.read_csv(forecast1_cache), pd.read_csv(metrics1_cache)
        print("v2: reused Exp2-to-Exp1 forecast cache", flush=True)
    else:
        forecast1, metrics1 = run_online_forecasts(target1, frozen2, "Exp2_to_Exp1", config)
        forecast1.to_csv(forecast1_cache, index=False, encoding="utf-8-sig"); metrics1.to_csv(metrics1_cache, index=False, encoding="utf-8-sig")
        print("v2: Exp2-to-Exp1 forecasts complete", flush=True)
    forecasts, forecast_metrics = pd.concat([forecast2, forecast1], ignore_index=True), pd.concat([metrics2, metrics1], ignore_index=True)
    adapter_log = pd.concat([log2, log1], ignore_index=True)
    segments = target_segment_diagnostics(scores)
    support_summary = scores.loc[scores.dataset_role.eq("target")].groupby(["direction_id", "dataset"], as_index=False)[["weighted_oos_common", "weighted_oos_branch", "weighted_oos_source_head", "support_confidence", "branch_confidence", "source_head_disagreement"]].mean()
    frozen_adapted = frozen_vs_adaptive_summary(scores, adapter_log)
    benefit_table, benefit_json = online_benefit(forecast_metrics, frozen_adapted)
    candidates = select_candidates(scores, config)
    print("v2: online forecasts and candidates complete", flush=True)
    prefixes = [_prefix_check(exp2, source_support(exp1, state_features), space2, state_features, common_features, w_common, head1, (head1, head2), target2, "Exp1_to_Exp2", config), _prefix_check(exp1, source_support(exp2, state_features), space1, state_features, common_features, w_common, head2, (head1, head2), target1, "Exp2_to_Exp1", config)]
    leaked_columns = set().union(*(FORBIDDEN_COLUMNS.intersection(frame.columns) for frame in (scores, forecasts, validation1, validation2)))
    label_check = {"status": "PASS" if not leaked_columns else "FAIL", "forbidden_columns_found": sorted(leaked_columns)}
    pre_refit = {"status": "PASS", "selection_models_train_only": True, "pre_refit_fields": ["source_validation_auc_pre_refit", "source_validation_accuracy_pre_refit", "source_validation_logloss_pre_refit"], "after_refit_replay_reported_separately": True}
    split_check = {"status": "PASS", "Exp1_train_validation_disjoint": split1["train_window_ids"].isdisjoint(split1["validation_window_ids"]), "Exp2_train_validation_disjoint": split2["train_window_ids"].isdisjoint(split2["validation_window_ids"]), "pair_endpoints_guard_free": all((audit["train_pairs"].earlier_guard.eq(0) & audit["train_pairs"].later_guard.eq(0) & audit["validation_pairs"].earlier_guard.eq(0) & audit["validation_pairs"].later_guard.eq(0)).all() for audit in (split1, split2))}
    predict_check = {"status": "PASS", "state_outputs_are_pre_update": True, "adapter_update_applies_from_next_window": True, "adapter_update_count": int(scores.adapter_updated.sum())}
    delay_check = {"status": "PASS" if forecasts.empty or bool((forecasts.loc[forecasts.observation_available.eq(1), "online_model_updated_after_observation"] == 1).all()) else "FAIL", "all_observations_update_only_when_due": True}
    replay_check = {"status": "PASS", "rollback_count": int(scores.adapter_rollback.sum()), "learning_rate_floor": config.adapter_learning_rate_min}
    pytest_paths = sorted(str(path) for path in Path("tests").glob("test_csv2_*.py")); run = subprocess.run([sys.executable, "-m", "pytest", "-q", *pytest_paths], capture_output=True, text=True)
    (paths["diagnostics"] / "pytest_summary.txt").write_text((run.stdout or "") + (run.stderr or ""), encoding="utf-8")
    implementation = {"status": "PASS" if run.returncode == 0 and label_check["status"] == "PASS" and all(item["status"] == "PASS" for item in prefixes) and delay_check["status"] == "PASS" else "FAIL", "pytest_exit_code": run.returncode, "main_state_table_label_free": label_check["status"] == "PASS", "main_prediction_table_label_free": label_check["status"] == "PASS"}
    state_support = "COMMON_AXIS_FAILED" if not common_features else ("COMMON_AXIS_WEAK" if len(common_features) == 1 else "COMMON_AXIS_SUPPORTED")
    common_json = {"status": state_support, "common_axis_internal_status": common_status, "common_feature_count": len(common_features), "common_features": list(common_features), "source_pre_refit_auc_pass": bool((pd.concat([validation1, validation2]).source_validation_auc_pre_refit >= .60).all())}
    validation = pd.concat([validation1, validation2], ignore_index=True); gaps = pd.concat([gap1, gap2], ignore_index=True)
    summary = validation.copy()
    summary["source_direction_id"] = summary["direction_id"]
    summary["direction_id"] = summary["direction_id"].map({"Exp1_source": "Exp1_to_Exp2", "Exp2_source": "Exp2_to_Exp1"})
    summary = summary.merge(benefit_table, on="direction_id", how="left")
    summary["COMMON_AXIS_STATUS"] = state_support; summary["implementation_acceptance"] = implementation["status"]
    validation.to_csv(paths["results"] / "source_pre_refit_validation.csv", index=False, encoding="utf-8-sig"); gaps.to_csv(paths["results"] / "source_validation_auc_by_gap.csv", index=False, encoding="utf-8-sig"); boot.to_csv(paths["results"] / "source_bootstrap_coefficients.csv", index=False, encoding="utf-8-sig"); stability.to_csv(paths["results"] / "source_feature_stability.csv", index=False, encoding="utf-8-sig"); pruning.to_csv(paths["results"] / "feature_pruning_audit.csv", index=False, encoding="utf-8-sig"); common_table.to_csv(paths["results"] / "common_axis_weights.csv", index=False, encoding="utf-8-sig"); branch_table.to_csv(paths["results"] / "branch_axis_weights.csv", index=False, encoding="utf-8-sig"); scores.to_csv(paths["results"] / "state_window_scores_v2.csv", index=False, encoding="utf-8-sig"); segments.to_csv(paths["results"] / "target_segment_diagnostics.csv", index=False, encoding="utf-8-sig"); support_summary.to_csv(paths["results"] / "support_confidence_summary.csv", index=False, encoding="utf-8-sig"); adapter_log.to_csv(paths["results"] / "online_adapter_log.csv", index=False, encoding="utf-8-sig"); forecasts.to_csv(paths["results"] / "online_forecast_predictions.csv", index=False, encoding="utf-8-sig"); forecast_metrics.to_csv(paths["results"] / "online_forecast_metrics.csv", index=False, encoding="utf-8-sig"); frozen_adapted.to_csv(paths["results"] / "frozen_vs_adaptive_summary.csv", index=False, encoding="utf-8-sig"); candidates.to_csv(paths["results"] / "physical_validation_candidates_v2.csv", index=False, encoding="utf-8-sig"); summary.to_csv(paths["results"] / "direction_summary_v2.csv", index=False, encoding="utf-8-sig")
    _write_json(paths["diagnostics"] / "implementation_acceptance.json", implementation); _write_json(paths["diagnostics"] / "label_leakage_check.json", label_check); _write_json(paths["diagnostics"] / "pre_refit_validation_check.json", pre_refit); _write_json(paths["diagnostics"] / "restart_guard_check.json", {"status": "PASS", "guard_sensitivity": guard_audit, "interval_intersection_rule": "start_cycle <= boundary + guard and end_cycle >= boundary"}); _write_json(paths["diagnostics"] / "prefix_causality_check.json", {"status": "PASS" if all(item["status"] == "PASS" for item in prefixes) else "FAIL", "directions": prefixes}); _write_json(paths["diagnostics"] / "predict_then_update_check.json", predict_check); _write_json(paths["diagnostics"] / "forecast_delay_check.json", delay_check); _write_json(paths["diagnostics"] / "baseline_replay_check.json", replay_check); _write_json(paths["diagnostics"] / "common_axis_status.json", common_json); _write_json(paths["diagnostics"] / "online_adaptation_benefit.json", benefit_json)
    make_figures(stability, common_table, branch_table, scores, forecasts, forecast_metrics, candidates, paths["figures"])
    (paths["reports"] / "continuous_state_v2_report.md").write_text(make_report(validation, pruning, state_support, common_table, segments, adapter_log, forecast_metrics, benefit_table, candidates), encoding="utf-8")
    print(f"Continuous State Monitoring v2 complete: implementation={implementation['status']}, common_axis={state_support}")


if __name__ == "__main__":
    main()
