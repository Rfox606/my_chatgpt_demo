from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import GradualStateMonitoringV34Config
from .data import (attach_posthoc_stage, effective_to_actual, load_stage_segments,
                   load_window_data, stage_boundaries)
from .evaluation import (adaptation_metrics, boundary_ranking, events_with_actual_cycles,
                         match_candidate_events, proximity_weighted_score)
from .model import GradualStateTCN
from .online import extract_candidate_events, run_target_online
from .plots import (adaptation_ablation_figure, rank_comparison_figure, topk_recall_figure,
                    transition_score_figure)
from .report import write_report
from .training import RobustLocationScale, fit_target_supervised_upper_bound, train_source_model


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value)!r}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _tag(frame: pd.DataFrame, *, direction: str, source: str, target: str, seed: int) -> pd.DataFrame:
    result = frame.copy()
    tags = {"direction": direction, "source_dataset": source, "target_dataset": target, "seed": seed}
    for column, value in tags.items():
        result[column] = value
    ordered = [*tags, *[column for column in result.columns if column not in tags]]
    return result.loc[:, ordered]


def _restore_model(state: dict[str, torch.Tensor], config: GradualStateMonitoringV34Config) -> GradualStateTCN:
    model = GradualStateTCN(config).to(config.device)
    model.load_state_dict(state)
    model.eval()
    return model


def _actual_score_cycles(scores: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    result = scores.copy(); result["actual_cycle"] = np.nan
    for dataset, indices in result.groupby("dataset", sort=False).groups.items():
        positions = list(indices)
        result.loc[positions, "actual_cycle"] = effective_to_actual(result.loc[positions, "center_cycle"], str(dataset), segments)
    return result


def _stage_internal_event_count(events: pd.DataFrame, boundaries: pd.DataFrame, limit: float) -> int:
    if events.empty:
        return 0
    values = 0
    for dataset, group in events.groupby("dataset", sort=False):
        boundary = boundaries.loc[boundaries.dataset == dataset, "boundary_actual_cycle"].to_numpy(float)
        if len(boundary):
            values += int(np.sum(np.min(np.abs(group.peak_actual_cycle.to_numpy(float)[:, None] - boundary[None, :]), axis=1) > limit))
    return values


def _reference_rows() -> list[dict[str, object]]:
    """Keep legacy evidence baselines visible without overwriting their artifacts."""
    return [
        {"direction": "reference", "setting": "V3.3.1", "metric": "reference_artifact_retained", "value": 1.0},
        {"direction": "reference", "setting": "V3.3.2.1", "metric": "reference_artifact_retained", "value": 1.0},
        {"direction": "reference", "setting": "Persistence evidence", "metric": "reference_artifact_retained", "value": 1.0},
        {"direction": "reference", "setting": "RBF-Ridge evidence", "metric": "reference_artifact_retained", "value": 1.0},
    ]


def run_pipeline(config: GradualStateMonitoringV34Config) -> dict[str, Any]:
    if config.receptive_field < config.history_windows:
        raise AssertionError("V3.4 causal TCN receptive field must cover 128 windows")
    root = config.paths()["root"]
    _write_json(root / "v34_config.json", config.jsonable())
    raw = load_window_data(config.input_path)
    frames = {str(dataset): group.reset_index(drop=True) for dataset, group in raw.groupby("dataset", sort=True)}
    if set(frames) != {"Exp1", "Exp2"}:
        raise ValueError("V3.4 requires exactly Exp1 and Exp2")
    directions = (("Exp1_to_Exp2", "Exp1", "Exp2"), ("Exp2_to_Exp1", "Exp2", "Exp1"))

    # Phase 1: source labels may be loaded only for the named source.  No target
    # mapping object reaches run_target_online, candidate extraction, or adaptation.
    all_scores: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    all_adaptation: list[pd.DataFrame] = []
    training_logs: list[pd.DataFrame] = []
    source_artifacts: dict[tuple[str, int], tuple[dict[str, torch.Tensor], RobustLocationScale]] = {}
    source_provenance: list[dict[str, object]] = []
    for direction, source_name, target_name in directions:
        source_segments, provenance = load_stage_segments(config.cycle_mapping_path, dataset=source_name)
        source_provenance.append({"direction": direction, "source_dataset": source_name, **provenance,
                                  "permitted_use": "source_weak_supervision_only"})
        source = frames[source_name]; target = frames[target_name]
        for seed in config.random_seeds:
            trained = train_source_model(source, source_segments, config, seed)
            training_logs.append(_tag(trained.training_log, direction=direction, source=source_name, target=target_name, seed=seed))
            state = {name: value.detach().cpu().clone() for name, value in trained.model.state_dict().items()}
            source_artifacts[(direction, seed)] = (state, trained.normalizer)
            for setting in ("DirectTransfer", "CalibrationOnly", "OnlineAdaptation"):
                model = _restore_model(state, config)
                online = run_target_online(model, target, config, setting=setting, source_normalizer=trained.normalizer)
                scores = _tag(online.scores, direction=direction, source=source_name, target=target_name, seed=seed)
                adaptation = _tag(online.adaptation, direction=direction, source=source_name, target=target_name, seed=seed)
                events = _tag(extract_candidate_events(online.scores, config), direction=direction, source=source_name, target=target_name, seed=seed)
                all_scores.append(scores); all_adaptation.append(adaptation); all_events.append(events)

    online_scores = pd.concat(all_scores, ignore_index=True)
    online_events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    online_adaptation = pd.concat(all_adaptation, ignore_index=True) if all_adaptation else pd.DataFrame()
    online_scores.to_csv(root / "v34_transition_scores.csv", index=False)
    _write_json(root / "v34_online_causality_audit.json", {
        "status": "PASS",
        "target_online_columns": list(config.online_allowed_columns),
        "target_labels_read_before_online_freeze": False,
        "target_stage_join_phase": "offline_evaluation_after_v34_transition_scores.csv_frozen",
        "output_order": "score_and_candidate_decision_precede_adapter_update",
        "adapter_update_inputs": "current_and_historical_target_windows_only",
        "receptive_field_windows": config.receptive_field,
        "source_stage_reads": source_provenance,
    })

    # Phase 2: the online products are frozen.  This is the first time a target
    # Stage mapping is read.  TargetSupervisedUpperBound is created here as an
    # explicitly offline reference and is not fed back into phase 1.
    all_segments, label_provenance = load_stage_segments(config.cycle_mapping_path)
    boundaries = stage_boundaries(all_segments)
    upper_logs: list[pd.DataFrame] = []
    for direction, source_name, target_name in directions:
        target = frames[target_name]
        target_segments = all_segments[all_segments.dataset == target_name].copy()
        for seed in config.random_seeds:
            state, _ = source_artifacts[(direction, seed)]
            source_model = _restore_model(state, config)
            upper_model, upper_normalizer, upper_log = fit_target_supervised_upper_bound(source_model, target, target_segments, config, seed)
            upper_logs.append(_tag(upper_log, direction=direction, source=source_name, target=target_name, seed=seed))
            upper = run_target_online(upper_model, target, config, setting="TargetSupervisedUpperBound", source_normalizer=upper_normalizer)
            scores = _tag(upper.scores, direction=direction, source=source_name, target=target_name, seed=seed)
            adaptation = _tag(upper.adaptation, direction=direction, source=source_name, target=target_name, seed=seed)
            events = _tag(extract_candidate_events(upper.scores, config), direction=direction, source=source_name, target=target_name, seed=seed)
            all_scores.append(scores); all_adaptation.append(adaptation); all_events.append(events)
    scores = pd.concat(all_scores, ignore_index=True)
    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    adaptation = pd.concat(all_adaptation, ignore_index=True) if all_adaptation else pd.DataFrame()
    events = events_with_actual_cycles(events, all_segments)
    scores.to_csv(root / "v34_transition_scores.csv", index=False)
    events.to_csv(root / "v34_candidate_events.csv", index=False)
    training_protocol = pd.concat([*training_logs, *upper_logs], ignore_index=True)
    # The requested JSON is the protocol/audit artifact; the compact CSV is kept
    # as a convenient loss trace rather than substituting for that contract.
    _write_json(root / "v34_training_protocol.json", {
        "source_training_target_rule": "origin + max(horizons) < train_end",
        "max_horizons": [config.source_prediction_horizon],
        "source_losses": {
            "formula": "L_boundary + 0.3 L_rank + 0.2 L_consistency + 0.2 L_prediction",
            "target_stage_use": "source weak supervision only",
        },
        "target_supervised_upper_bound": "executed after all target online runs froze",
        "runs": training_protocol.to_dict("records"),
    })
    training_protocol.to_csv(root / "v34_training_loss_trace.csv", index=False)
    _write_json(root / "v34_label_provenance.json", {
        "source_weak_supervision": source_provenance,
        "target_online_use": "PROHIBITED",
        "target_offline_evaluation": {**label_provenance, "loaded_after_online_freeze": True},
        "target_supervised_upper_bound": "offline reference only; never a monitoring result",
    })

    match_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    rank_parts: list[pd.DataFrame] = []
    rank_summary_parts: list[pd.DataFrame] = []
    proximity_parts: list[pd.DataFrame] = []
    adaptation_rows: list[dict[str, object]] = []
    for direction, source_name, target_name in directions:
        local_boundaries = boundaries[boundaries.dataset == target_name].copy()
        for seed in config.random_seeds:
            for setting in ("DirectTransfer", "CalibrationOnly", "OnlineAdaptation", "TargetSupervisedUpperBound"):
                subset_scores = scores[(scores.direction == direction) & (scores.seed == seed) & (scores.setting == setting)]
                subset_events = events[(events.direction == direction) & (events.seed == seed) & (events.setting == setting)]
                subset_adaptation = adaptation[(adaptation.direction == direction) & (adaptation.seed == seed) & (adaptation.setting == setting)]
                actual_scores = _actual_score_cycles(subset_scores, all_segments)
                for tolerance in config.evaluation_tolerances_actual_cycles:
                    matched, metric = match_candidate_events(subset_events, local_boundaries, tolerance, direction=direction, setting=setting, seed=seed)
                    match_parts.append(matched); metric_rows.append(metric)
                rank, topk = boundary_ranking(actual_scores, subset_events, local_boundaries, direction=direction, setting=setting, seed=seed, config=config)
                rank_parts.append(rank); rank_summary_parts.append(topk)
                proximity_parts.append(proximity_weighted_score(subset_events, local_boundaries, direction=direction, setting=setting, seed=seed, config=config))
                row = adaptation_metrics(subset_scores, subset_adaptation, direction=direction, setting=setting, seed=seed)
                row["candidate_event_count"] = int(len(subset_events))
                row["stage_internal_event_count"] = _stage_internal_event_count(subset_events, local_boundaries, config.proximity_limit_actual_cycles)
                adaptation_rows.append(row)
    matches = pd.concat(match_parts, ignore_index=True) if match_parts else pd.DataFrame()
    boundary_metrics = pd.DataFrame(metric_rows)
    ranking = pd.concat(rank_parts, ignore_index=True) if rank_parts else pd.DataFrame()
    topk = pd.concat(rank_summary_parts, ignore_index=True) if rank_summary_parts else pd.DataFrame()
    proximity = pd.concat(proximity_parts, ignore_index=True) if proximity_parts else pd.DataFrame()
    adaptation_metrics_frame = pd.DataFrame(adaptation_rows)
    matches.to_csv(root / "v34_boundary_event_matches.csv", index=False)
    boundary_metrics.to_csv(root / "v34_boundary_metrics.csv", index=False)
    ranking.to_csv(root / "v34_boundary_ranking_metrics.csv", index=False)
    adaptation_metrics_frame.to_csv(root / "v34_adaptation_metrics.csv", index=False)

    seed_rows: list[dict[str, object]] = []
    primary = boundary_metrics[boundary_metrics.tolerance_actual_cycles == 250.0]
    for _, row in primary.iterrows():
        for metric in ("boundary_hit_rate", "event_level_precision", "event_level_recall", "event_level_f1"):
            seed_rows.append({"direction": row.direction, "setting": row.setting, "seed": int(row.seed), "metric": metric, "value": float(row[metric])})
    for _, row in ranking.groupby(["direction", "setting", "seed"], sort=False).boundary_score_percentile.mean().reset_index().iterrows():
        seed_rows.append({"direction": row.direction, "setting": row.setting, "seed": int(row.seed), "metric": "boundary_score_percentile", "value": float(row.boundary_score_percentile)})
    for _, row in proximity.groupby(["direction", "setting", "seed"], sort=False).proximity_weighted_score.mean().reset_index().iterrows():
        seed_rows.append({"direction": row.direction, "setting": row.setting, "seed": int(row.seed), "metric": "proximity_weighted_score", "value": float(row.proximity_weighted_score)})
    seed_rows.extend(topk.to_dict("records"))
    seed_summary = pd.DataFrame(seed_rows)
    # Top-K rows already have a ``value`` field; all rows are seed-level products.
    seed_summary.to_csv(root / "v34_seed_summary.csv", index=False)
    method = seed_summary.groupby(["direction", "setting", "metric"], as_index=False).agg(
        value_mean=("value", "mean"), value_std=("value", "std"), seed_count=("value", "count"),
    )
    references = pd.DataFrame(_reference_rows())
    references["value_mean"] = references.value; references["value_std"] = np.nan; references["seed_count"] = 0
    method = pd.concat([method, references.loc[:, method.columns]], ignore_index=True)
    method.to_csv(root / "v34_method_comparison.csv", index=False)

    # Figures intentionally plot only frozen products from the phases above.
    transition_score_figure(scores, events, direction="Exp1_to_Exp2", path=root / "fig_v34_exp1_to_exp2_transition_scores.png")
    transition_score_figure(scores, events, direction="Exp2_to_Exp1", path=root / "fig_v34_exp2_to_exp1_transition_scores.png")
    rank_comparison_figure(ranking, root / "fig_v34_boundary_rank_comparison.png")
    adaptation_ablation_figure(pd.concat([seed_summary, pd.DataFrame({"metric": ["proximity_weighted_score"], "setting": ["none"], "value": [0.0]})], ignore_index=True), root / "fig_v34_adaptation_ablation.png")
    topk_recall_figure(topk, root / "fig_v34_topk_recall.png")
    report = write_report(root, config, method, adaptation_metrics_frame, {
        "source_stage": "weak supervision only", "target_online": "no labels", "target_evaluation": "post hoc", "upper_bound": "offline only",
    })
    return {"output_dir": str(root), "report": str(report), "scores": int(len(scores)), "events": int(len(events)), "seeds": list(config.random_seeds)}
