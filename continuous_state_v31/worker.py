from __future__ import annotations

"""Memory-isolated v3.1 execution stages used by the top-level runner."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import ContinuousStateV31Config
from .data import assert_label_free, load_window_table
from .feature_pruning import prune_features
from .forecast import _metrics_from_predictions, _regret, _rolling_metrics, _segment_metrics, run_online_forecasts, train_frozen_models
from .guards import add_restart_guard
from .source_prior import SourceProtocolModel, build_source_model
from .state_engine import PlateauPrior, derive_plateau_prior, run_target_state


class FrozenLinear:
    """Serializable F0 ridge coefficients with the tiny sklearn prediction surface used online."""

    def __init__(self, coefficient: list[float]) -> None:
        self.coef_ = np.asarray(coefficient, dtype=float)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) @ self.coef_


def _work(config: ContinuousStateV31Config) -> Path:
    path = config.paths()["root"] / "work_csv31"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _spec(protocol: str) -> tuple[str, str, bool]:
    if protocol == "A_Exp1_to_Exp2":
        return "Exp1", "Exp2", False
    if protocol == "B_Exp2_to_Exp1":
        return "Exp2", "Exp1", True
    raise ValueError(f"Unknown v3.1 protocol: {protocol}")


def _raw(config: ContinuousStateV31Config) -> dict[str, pd.DataFrame]:
    frame = add_restart_guard(load_window_table(config), config)
    result = {dataset: frame.loc[frame.dataset.eq(dataset)].reset_index(drop=True) for dataset in ("Exp1", "Exp2")}
    for item in result.values():
        assert_label_free(item)
    return result


def _strength(audit: pd.DataFrame) -> dict[str, float]:
    kept = audit.loc[audit.kept.eq(1)]
    return {str(row.feature_name): float(row.direction_free_auc) if np.isfinite(row.direction_free_auc) else 1.0 for _, row in kept.iterrows()}


def _model_path(work: Path, protocol: str) -> Path:
    return work / f"{protocol}_model.json"


def _load_model(work: Path, protocol: str, source_states: pd.DataFrame) -> SourceProtocolModel:
    payload = json.loads(_model_path(work, protocol).read_text(encoding="utf-8"))
    return SourceProtocolModel(
        protocol_id=protocol,
        source_dataset=str(payload["source_dataset"]),
        features=tuple(payload["features"]),
        feature_strength={str(key): float(value) for key, value in payload["feature_strength"].items()},
        plateau_prior=PlateauPrior(**payload["plateau_prior"]),
        source_states=source_states,
        severe_direction=None if payload["severe_direction"] is None else np.asarray(payload["severe_direction"], dtype=float),
        source_exit_cycle=float(payload["source_exit_cycle"]),
    )


def source_stage(protocol: str, config: ContinuousStateV31Config) -> None:
    work = _work(config); source_dataset, _, allow = _spec(protocol)
    source = _raw(config)[source_dataset]
    features, audit = prune_features(source, protocol, config)
    model, plateau, severe = build_source_model(source, features, _strength(audit), protocol, allow, config)
    model.source_states.to_csv(work / f"{protocol}_source_states.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(work / f"{protocol}_feature_audit.csv", index=False, encoding="utf-8-sig")
    plateau.to_csv(work / f"{protocol}_plateau_prior.csv", index=False, encoding="utf-8-sig")
    severe.to_csv(work / f"{protocol}_severe_prior.csv", index=False, encoding="utf-8-sig")
    payload = {"source_dataset": model.source_dataset, "features": list(model.features), "feature_strength": model.feature_strength,
               "plateau_prior": asdict(model.plateau_prior), "severe_direction": None if model.severe_direction is None else model.severe_direction.tolist(),
               "source_exit_cycle": model.source_exit_cycle}
    _model_path(work, protocol).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"v3.1 worker source complete: {protocol}", flush=True)


def target_stage(protocol: str, config: ContinuousStateV31Config) -> None:
    work = _work(config); source_dataset, target_dataset, _ = _spec(protocol); data = _raw(config)
    model = _load_model(work, protocol, data[source_dataset])
    states, plateau, exits, updates, metadata = run_target_state(data[target_dataset], data[source_dataset], model.features, model.feature_strength,
                                                                   model.plateau_prior, model.severe_direction, protocol, config)
    states.to_csv(work / f"{protocol}_target_states.csv", index=False, encoding="utf-8-sig")
    plateau.to_csv(work / f"{protocol}_plateau_events.csv", index=False, encoding="utf-8-sig")
    exits.to_csv(work / f"{protocol}_exit_events.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(work / f"{protocol}_updates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metadata["severe_events"]).to_csv(work / f"{protocol}_severe_events.csv", index=False, encoding="utf-8-sig")
    (work / f"{protocol}_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"v3.1 worker target complete: {protocol}", flush=True)


def forecast_stage(protocol: str, config: ContinuousStateV31Config) -> None:
    work = _work(config)
    target = pd.read_csv(work / f"{protocol}_target_states.csv")
    payload = json.loads((work / f"{protocol}_frozen_models.json").read_text(encoding="utf-8"))
    models = {(target_name, int(horizon)): None if coefficient is None else FrozenLinear(coefficient)
              for target_name, horizons in payload.items() for horizon, coefficient in horizons.items()}
    result = run_online_forecasts(target, models, protocol, config, include_evaluation=False)
    names = ("predictions", "metrics", "segments", "rolling", "regret", "state_log", "episodes", "weights")
    for name, frame in zip(names, result, strict=True):
        frame.to_csv(work / f"{protocol}_{name}.csv", index=False, encoding="utf-8-sig")
    print(f"v3.1 worker forecast complete: {protocol}", flush=True)


def evaluate_stage(protocol: str, config: ContinuousStateV31Config) -> None:
    work = _work(config)
    predictions = pd.read_csv(work / f"{protocol}_predictions.csv")
    _metrics_from_predictions(predictions, protocol).to_csv(work / f"{protocol}_metrics.csv", index=False, encoding="utf-8-sig")
    _segment_metrics(predictions, protocol).to_csv(work / f"{protocol}_segments.csv", index=False, encoding="utf-8-sig")
    _rolling_metrics(predictions, protocol, config.rolling_metric_observations, config.rolling_metric_export_stride).to_csv(work / f"{protocol}_rolling.csv", index=False, encoding="utf-8-sig")
    _regret(predictions, protocol, config.rolling_metric_export_stride).to_csv(work / f"{protocol}_regret.csv", index=False, encoding="utf-8-sig")
    print(f"v3.1 worker evaluation complete: {protocol}", flush=True)


def train_stage(protocol: str, config: ContinuousStateV31Config) -> None:
    work = _work(config)
    source = pd.read_csv(work / f"{protocol}_source_states.csv")
    trained = train_frozen_models(source, config)
    payload: dict[str, dict[str, list[float] | None]] = {}
    for (target, horizon), model in trained.items():
        payload.setdefault(target, {})[str(horizon)] = None if model is None else np.asarray(model.coef_, dtype=float).reshape(-1).tolist()
    (work / f"{protocol}_frozen_models.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"v3.1 worker frozen-model training complete: {protocol}", flush=True)


def sensitivity_stage(protocol: str, quantile: float, config: ContinuousStateV31Config) -> None:
    work = _work(config); source_dataset, target_dataset, _ = _spec(protocol); data = _raw(config)
    model = _load_model(work, protocol, data[source_dataset])
    prior = derive_plateau_prior(data[source_dataset], model.features, config, quantile)
    states, _, _, _, metadata = run_target_state(data[target_dataset], data[source_dataset], model.features, model.feature_strength,
                                                   prior, model.severe_direction, protocol, config)
    locked = states.loc[states.plateau_locked.eq(1), "center_cycle"]; exited = states.loc[states.plateau_exit_confirmed.eq(1), "center_cycle"]
    pd.DataFrame([{"protocol_id": protocol, "target_dataset": target_dataset, "quantile": quantile,
                   "plateau_lock_detected": int(not locked.empty), "plateau_lock_cycle": float(locked.min()) if not locked.empty else np.nan,
                   "plateau_exit_detected": int(not exited.empty), "plateau_exit_cycle": float(exited.min()) if not exited.empty else np.nan,
                   "plateau_condition_rate": float(states.plateau_condition.mean()), "plateau_valid_cycles_final": float(states.plateau_valid_cycles.iloc[-1]),
                   "exit_valid_cycles_final": float(states.exit_valid_cycles.iloc[-1]), "threshold_quantile_pre_registered": True,
                   "metadata_exit_cycle": metadata["exit_cycle"]}]).to_csv(work / f"{protocol}_sensitivity_{quantile:.2f}.csv", index=False, encoding="utf-8-sig")
    print(f"v3.1 worker sensitivity complete: {protocol} q{quantile:.2f}", flush=True)


def prefix_stage(protocol: str, cutoff: float, config: ContinuousStateV31Config) -> None:
    work = _work(config); source_dataset, target_dataset, _ = _spec(protocol); data = _raw(config)
    full = pd.read_csv(work / f"{protocol}_target_states.csv")
    model = _load_model(work, protocol, data[source_dataset])
    target = data[target_dataset]; prefix = target.loc[target.center_cycle <= cutoff]
    columns = ("D_state", "V50_norm", "A_smooth_20", "instability_score", "S_severe_candidate", "plateau_valid_cycles", "exit_valid_cycles")
    if prefix.empty:
        result = {"prefix_cycle": cutoff, "window_count": 0, "max_abs_difference": 0.0, "pass": True}
    else:
        scored, _, _, _, _ = run_target_state(prefix, data[source_dataset], model.features, model.feature_strength,
                                                model.plateau_prior, model.severe_direction, protocol, config)
        reference = full.loc[full.center_cycle <= cutoff, ["window_index", *columns]]
        merged = scored.merge(reference, on="window_index", suffixes=("_prefix", "_full"))
        maxima = []
        for column in columns:
            difference = np.abs(merged[f"{column}_prefix"].to_numpy(float) - merged[f"{column}_full"].to_numpy(float))
            difference = difference[np.isfinite(difference)]
            maxima.append(float(difference.max()) if len(difference) else 0.0)
        result = {"prefix_cycle": cutoff, "window_count": int(len(merged)), "max_abs_difference": max(maxima, default=0.0),
                  "pass": max(maxima, default=0.0) < 1e-10}
    (work / f"{protocol}_prefix_{int(cutoff)}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"v3.1 worker prefix complete: {protocol} {int(cutoff)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("source", "target", "train", "forecast", "evaluate", "sensitivity", "prefix"))
    parser.add_argument("protocol", choices=("A_Exp1_to_Exp2", "B_Exp2_to_Exp1"))
    parser.add_argument("--quantile", type=float)
    parser.add_argument("--cutoff", type=float)
    args = parser.parse_args()
    config = ContinuousStateV31Config()
    if args.stage == "source": source_stage(args.protocol, config)
    elif args.stage == "target": target_stage(args.protocol, config)
    elif args.stage == "train": train_stage(args.protocol, config)
    elif args.stage == "forecast": forecast_stage(args.protocol, config)
    elif args.stage == "evaluate": evaluate_stage(args.protocol, config)
    elif args.stage == "sensitivity":
        if args.quantile is None: parser.error("--quantile is required for sensitivity")
        sensitivity_stage(args.protocol, args.quantile, config)
    elif args.stage == "prefix":
        if args.cutoff is None: parser.error("--cutoff is required for prefix")
        prefix_stage(args.protocol, args.cutoff, config)


if __name__ == "__main__":
    main()
