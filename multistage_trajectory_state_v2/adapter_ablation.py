from __future__ import annotations

import hashlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from .config import FEATURE_CONFIGS, MultiStageTrajectoryConfig
from .data import robust_scale


@dataclass(frozen=True)
class FrozenSourceRanker:
    location: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    ood_threshold: float
    score_location: float
    score_scale: float
    frozen_hash: str

    def z(self, frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
        return (frame.loc[:, list(features)].to_numpy(float) - self.location) / self.scale

    def score(self, frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
        raw = self.z(frame, features) @ self.coefficient
        return expit((raw - self.score_location) / self.score_scale)


def _time_pairs(cycles: np.ndarray, maximum: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed); earlier: list[int] = []; later: list[int] = []
    for right in range(1, len(cycles)):
        candidates = np.flatnonzero((cycles[:right] >= cycles[right] - 5000.0) & (cycles[:right] <= cycles[right] - 500.0))
        if len(candidates):
            earlier.append(int(rng.choice(candidates))); later.append(right)
    if len(later) > maximum:
        chosen = np.sort(rng.choice(np.arange(len(later)), size=maximum, replace=False)); earlier = list(np.asarray(earlier)[chosen]); later = list(np.asarray(later)[chosen])
    return np.asarray(earlier, dtype=int), np.asarray(later, dtype=int)


def fit_ranker(frame: pd.DataFrame, features: tuple[str, ...], maximum_pairs: int, seed: int) -> FrozenSourceRanker:
    values = frame.loc[:, list(features)].to_numpy(float); location, scale = robust_scale(values); z = (values - location) / scale
    early, late = _time_pairs(frame.center_cycle.to_numpy(float), maximum_pairs, seed)
    delta = z[late] - z[early]
    model = LogisticRegression(C=.2, penalty="l2", fit_intercept=False, max_iter=500, solver="lbfgs", random_state=seed)
    model.fit(np.vstack((delta, -delta)), np.concatenate((np.ones(len(delta)), np.zeros(len(delta)))))
    coefficient = model.coef_.reshape(-1)
    threshold = float(np.quantile(np.sqrt(np.mean(z ** 2, axis=1)), .95)); raw = z @ coefficient
    digest = hashlib.sha256(np.asarray(coefficient, dtype=np.float64).tobytes()).hexdigest()
    return FrozenSourceRanker(location, scale, coefficient, threshold, float(np.median(raw)), max(float(np.std(raw)), 1e-9), digest)


def _score_metrics(scores: np.ndarray, cycles: np.ndarray, local_scores: np.ndarray, source_scores: np.ndarray, ood: np.ndarray, config: MultiStageTrajectoryConfig) -> dict[str, float]:
    early, late = _time_pairs(cycles, 800, config.random_seed + 55)
    auc = float(np.mean(scores[late] > scores[early]) + .5 * np.mean(scores[late] == scores[early])) if len(late) else np.nan
    differences = np.diff(scores); signs = np.sign(differences); signs = signs[signs != 0]
    reversals = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    agreement = float(pd.Series(scores).corr(pd.Series(local_scores), method="spearman")) if len(scores) > 2 else np.nan
    return {
        "future_time_pair_auc": auc,
        "future_score_total_variation": float(np.abs(differences).sum()),
        "future_score_saturation_ratio": float(np.mean((scores <= .01) | (scores >= .99))),
        "future_score_reversal_count": reversals,
        "prior_adapted_disagreement": float(np.mean(np.abs(scores - source_scores))),
        "future_target_local_spearman": agreement,
        "future_ood_ratio_mean": float(np.mean(ood)),
        "future_ood_fraction": float(np.mean(ood > 1.0)),
    }


def _adapter_fit(
    z: np.ndarray, cycles: np.ndarray, source_coefficient: np.ndarray, config: MultiStageTrajectoryConfig, mode: str, seed: int,
) -> tuple[np.ndarray, list[dict[str, object]], bool, str]:
    l2 = config.adapter_l2 * (config.adapter_weak_l2_multiplier if mode == "Unbounded_WeakL2" else 1.0)
    residual = np.zeros(z.shape[1], dtype=float); trace: list[dict[str, object]] = []
    early, late = _time_pairs(cycles, 1200, seed); aborted = False; reason = "COMPLETED"
    for update, (left, right) in enumerate(zip(early, late)):
        delta = z[right] - z[left]; margin = float(np.dot(source_coefficient + residual, delta)); prediction = float(expit(margin))
        loss = float(-np.log(max(prediction, 1e-12)) + .5 * l2 * np.dot(residual, residual))
        step = -config.adapter_learning_rate * ((prediction - 1.0) * delta + l2 * residual)
        if mode == "Bounded_Baseline":
            step_norm = float(np.linalg.norm(step))
            if step_norm > config.adapter_bounded_step:
                step *= config.adapter_bounded_step / step_norm
        residual = residual + step
        if mode == "Bounded_Baseline":
            norm = float(np.linalg.norm(residual))
            if norm > config.adapter_bounded_norm:
                residual *= config.adapter_bounded_norm / norm
        norm = float(np.linalg.norm(residual))
        if not np.isfinite(norm) or not np.isfinite(loss):
            aborted, reason = True, "NUMERIC_NONFINITE"; break
        if mode != "Bounded_Baseline" and norm > config.adapter_unbounded_abort_norm:
            aborted, reason = True, "NUMERIC_NORM_ABORT"; break
        if update % 25 == 0 or update == len(early) - 1:
            trace.append({"update_index": update, "pair_earlier_cycle": float(cycles[left]), "pair_later_cycle": float(cycles[right]), "adapter_parameter_norm": norm, "adapter_step_norm": float(np.linalg.norm(step)), "objective_loss": loss, "l2": l2, "numeric_abort": 0})
    if aborted:
        trace.append({"update_index": len(trace), "adapter_parameter_norm": float(np.linalg.norm(residual)), "adapter_step_norm": np.nan, "objective_loss": np.nan, "l2": l2, "numeric_abort": 1, "abort_reason": reason})
    return residual, trace, aborted, reason


def run_adapter_ablation(frame: pd.DataFrame, config: MultiStageTrajectoryConfig, figures: object) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    features = FEATURE_CONFIGS[config.primary_feature_config]; summaries: list[dict[str, object]] = []; traces: list[dict[str, object]] = []; scores_long: list[dict[str, object]] = []
    datasets = {name: group.sort_values(["center_cycle", "window_index"]).reset_index(drop=True) for name, group in frame.groupby("dataset", sort=True)}
    directions = (("Exp1_to_Exp2", "Exp1", "Exp2", config.exp2_entry_cycles), ("Exp2_to_Exp1", "Exp2", "Exp1", config.exp1_entry_cycles))
    for direction, source_name, target_name, entries in directions:
        source = datasets[source_name]; target = datasets[target_name]; source_ranker = fit_ranker(source, features, 2400, config.random_seed + 1)
        source_z_target = source_ranker.z(target, features)
        source_raw = source_z_target @ source_ranker.coefficient
        source_score_all = source_ranker.score(target, features)
        source_ood_all = np.sqrt(np.mean(source_z_target ** 2, axis=1)) / max(source_ranker.ood_threshold, 1e-9)
        for entry in entries:
            available = target.loc[target.center_cycle >= entry].reset_index(drop=True)
            if len(available) < 80:
                continue
            available_indices = target.index[target.center_cycle >= entry].to_numpy(int)
            target_z = source_z_target[available_indices]; target_source_score = source_score_all[available_indices]; target_ood = source_ood_all[available_indices]; target_cycles = available.center_cycle.to_numpy(float)
            for prefix_fraction in config.adapter_prefix_fractions:
                prefix_count = max(40, int(np.floor(len(available) * prefix_fraction)))
                prefix = available.iloc[:prefix_count].reset_index(drop=True); future = available.iloc[prefix_count:].reset_index(drop=True)
                if len(future) < 20:
                    continue
                local_ranker = fit_ranker(prefix, features, 1600, config.random_seed + int(entry) + int(prefix_fraction * 100))
                local_score = local_ranker.score(future, features)
                future_z = target_z[prefix_count:]; future_cycles = target_cycles[prefix_count:]; future_source = target_source_score[prefix_count:]; future_ood = target_ood[prefix_count:]
                methods: dict[str, np.ndarray] = {
                    "Source_Static": future_source,
                    "Target_Local": local_score,
                    "Elapsed_Time_Since_Entry": (future_cycles - entry) / max(float(target_cycles[-1] - entry), 1e-9),
                }
                for method in ("Bounded_Baseline", "Unbounded_L2", "Unbounded_WeakL2"):
                    residual, path, aborted, reason = _adapter_fit(target_z[:prefix_count], target_cycles[:prefix_count], source_ranker.coefficient, config, method, config.random_seed + int(entry) + int(prefix_fraction * 1000) + len(method))
                    for item in path:
                        traces.append({"direction": direction, "source_dataset": source_name, "target_dataset": target_name, "entry_cycle": entry, "prefix_fraction": prefix_fraction, "adapter": method, "source_model_frozen_hash": source_ranker.frozen_hash, **item})
                    adapted_raw = future_z @ (source_ranker.coefficient + residual)
                    adapted = expit((adapted_raw - source_ranker.score_location) / source_ranker.score_scale)
                    methods[method] = adapted
                    adapter_norm = float(np.linalg.norm(residual))
                    metrics = _score_metrics(adapted, future_cycles, local_score, future_source, future_ood, config)
                    summaries.append({"direction": direction, "source_dataset": source_name, "target_dataset": target_name, "entry_cycle": entry, "prefix_fraction": prefix_fraction, "model": method, "future_window_count": int(len(future)), "adapter_aborted": int(aborted), "adapter_abort_reason": reason, "adapter_parameter_norm": adapter_norm, "source_model_frozen_hash": source_ranker.frozen_hash, **metrics})
                    for cycle, score in zip(future_cycles, adapted): scores_long.append({"direction": direction, "entry_cycle": entry, "prefix_fraction": prefix_fraction, "model": method, "center_cycle": float(cycle), "score": float(score), "future_start_cycle": float(future_cycles[0])})
                for method in ("Source_Static", "Target_Local", "Elapsed_Time_Since_Entry"):
                    score = methods[method]; metrics = _score_metrics(score, future_cycles, local_score, future_source, future_ood, config)
                    summaries.append({"direction": direction, "source_dataset": source_name, "target_dataset": target_name, "entry_cycle": entry, "prefix_fraction": prefix_fraction, "model": method, "future_window_count": int(len(future)), "adapter_aborted": 0, "adapter_abort_reason": "NOT_APPLICABLE", "adapter_parameter_norm": 0.0, "source_model_frozen_hash": source_ranker.frozen_hash, **metrics})
                    for cycle, value in zip(future_cycles, score): scores_long.append({"direction": direction, "entry_cycle": entry, "prefix_fraction": prefix_fraction, "model": method, "center_cycle": float(cycle), "score": float(value), "future_start_cycle": float(future_cycles[0])})
    summary = pd.DataFrame(summaries); parameter_path = pd.DataFrame(traces); score_paths = pd.DataFrame(scores_long)
    # Same final 20% future points are available under all prefixes and expose convergence instead of time-only fit.
    convergence_rows: list[dict[str, object]] = []
    for keys, group in score_paths.groupby(["direction", "entry_cycle", "model"], sort=True):
        cutoff = group.center_cycle.quantile(.80); tail = group.loc[group.center_cycle >= cutoff]
        pivot = tail.pivot_table(index="center_cycle", columns="prefix_fraction", values="score", aggfunc="first")
        convergence_rows.append({"direction": keys[0], "entry_cycle": keys[1], "model": keys[2], "common_future_window_count": int(len(pivot.dropna())), "common_future_prefix_score_std": float(pivot.std(axis=1).mean()) if not pivot.empty else np.nan})
    convergence = pd.DataFrame(convergence_rows)
    summary = summary.merge(convergence, on=["direction", "entry_cycle", "model"], how="left")
    decision: dict[str, object] = {"status": "FAIL", "directions": {}, "criteria": {"auc_improvement_minimum": .01, "saturation_maximum": .20, "requires_agreement_non_decrease": True, "requires_total_variation_non_increase": True, "requires_common_future_prefix_std_decrease": True}}
    for direction, group in summary.groupby("direction", sort=True):
        bounded = group.loc[group.model.eq("Bounded_Baseline")]; baseline_std = float(bounded.common_future_prefix_score_std.mean())
        direction_result: dict[str, object] = {"unbounded_groups": {}}
        for method in ("Unbounded_L2", "Unbounded_WeakL2"):
            candidate = group.loc[group.model.eq(method)]
            criteria = {
                "auc_improved": bool(candidate.future_time_pair_auc.mean() >= bounded.future_time_pair_auc.mean() + .01),
                "target_local_agreement_non_decrease": bool(candidate.future_target_local_spearman.mean() >= bounded.future_target_local_spearman.mean()),
                "total_variation_non_increase": bool(candidate.future_score_total_variation.mean() <= bounded.future_score_total_variation.mean()),
                "no_numeric_abort": bool((candidate.adapter_aborted == 0).all()),
                "not_saturated": bool((candidate.future_score_saturation_ratio <= .20).all()),
                "common_future_prefix_std_decreased": bool(candidate.common_future_prefix_score_std.mean() < baseline_std),
            }
            criteria["passed"] = bool(all(criteria.values())); direction_result["unbounded_groups"][method] = criteria
        direction_result["passed"] = bool(any(item["passed"] for item in direction_result["unbounded_groups"].values())); decision["directions"][direction] = direction_result
    decision["status"] = "PASS" if any(item["passed"] for item in decision["directions"].values()) else "FAIL"
    if not parameter_path.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for method, group in parameter_path.groupby("adapter"):
            ax.plot(group.update_index, group.adapter_parameter_norm, marker=".", linestyle="none", alpha=.25, label=method)
        ax.axhline(config.adapter_bounded_norm, color="black", linestyle="--", label="bounded norm")
        ax.set(title="Adapter parameter paths", xlabel="update index", ylabel="parameter norm"); ax.legend(); fig.tight_layout(); fig.savefig(getattr(figures, "__truediv__")("adapter_norm_paths_v2.png"), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); means = summary.groupby("model", sort=False).future_time_pair_auc.mean().sort_values()
    means.plot.barh(ax=ax, color="slateblue"); ax.set(title="Future-frozen time-pair AUC (secondary diagnostic)", xlabel="mean AUC"); fig.tight_layout(); fig.savefig(getattr(figures, "__truediv__")("future_frozen_comparison_v2.png"), dpi=150); plt.close(fig)
    return summary, parameter_path, score_paths, decision
