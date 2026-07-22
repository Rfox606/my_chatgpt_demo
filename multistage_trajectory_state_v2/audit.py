from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.neighbors import NearestNeighbors

from .config import FEATURE_CONFIGS, MultiStageTrajectoryConfig
from .data import equal_time_sample, robust_scale


def _window_positions(length: int, fraction: float, step_fraction: float) -> list[tuple[int, int]]:
    width = max(8, int(round(length * fraction)))
    step = max(1, int(round(length * step_fraction)))
    starts = list(range(0, max(1, length - width + 1), step))
    final_start = max(0, length - width)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return [(start, min(length, start + width)) for start in sorted(set(starts))]


def _block_resample_indices(length: int, fraction: float, rng: np.random.Generator) -> np.ndarray:
    block = max(3, int(np.ceil(length * fraction)))
    starts = np.arange(max(1, length - block + 1))
    selected: list[np.ndarray] = []
    while sum(len(part) for part in selected) < length:
        start = int(rng.choice(starts))
        selected.append(np.arange(start, min(length, start + block), dtype=int))
    return np.sort(np.concatenate(selected)[:length])


def _safe_slope(cycles: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 3 or np.ptp(cycles) <= 0 or np.ptp(values) <= 1e-12:
        return 0.0
    # Huber regression is a robust local slope but, unlike all-pairs Theil--Sen,
    # remains tractable for the prescribed 10--30% windows of the real trace.
    x_location = float(np.median(cycles)); x_scale = max(float(np.ptp(cycles)), 1e-12)
    model = HuberRegressor(epsilon=1.35, alpha=1e-8, max_iter=100)
    model.fit(((cycles - x_location) / x_scale).reshape(-1, 1), values)
    return float(model.coef_[0] / x_scale)


def _fast_bootstrap_slope(cycles: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 3 or np.ptp(cycles) <= 0 or np.ptp(values) <= 1e-12:
        return 0.0
    return float(np.polyfit(cycles, values, 1)[0])


def _bootstrap_sign_stability(cycles: np.ndarray, values: np.ndarray, config: MultiStageTrajectoryConfig, seed: int) -> float:
    observed = int(np.sign(_safe_slope(cycles, values)))
    if observed == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = []
    for _ in range(config.local_bootstrap_replicates):
        index = _block_resample_indices(len(values), config.bootstrap_block_fraction, rng)
        # The bootstrap tests direction stability; the reported point estimate
        # remains the robust Huber slope above.
        signs.append(int(np.sign(_fast_bootstrap_slope(cycles[index], values[index]))))
    return float(np.mean(np.asarray(signs) == observed))


def offline_preprocess(frame: pd.DataFrame, config: MultiStageTrajectoryConfig) -> pd.DataFrame:
    """Create a clearly marked, audit-only centred smoother and its derivatives."""
    features = tuple(dict.fromkeys(feature for values in FEATURE_CONFIGS.values() for feature in values))
    rows: list[pd.DataFrame] = []
    for dataset, group in frame.groupby("dataset", sort=True):
        group = group.sort_values(["center_cycle", "window_index"]).copy()
        n = len(group)
        width = max(5, int(round(n * config.offline_smoothing_fraction)))
        if width % 2 == 0:
            width += 1
        cycles = group.center_cycle.to_numpy(float)
        result = group.loc[:, ["dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle", *features]].copy()
        result["offline_diagnostic_only"] = True
        result["offline_smoothing_window_rows"] = width
        for feature in features:
            values = group[feature].to_numpy(float)
            smooth = pd.Series(values).rolling(width, center=True, min_periods=max(3, width // 3)).median().bfill().ffill().to_numpy(float)
            derivative = np.gradient(smooth, cycles)
            curvature = np.gradient(derivative, cycles)
            volatility = pd.Series(values).rolling(width, center=True, min_periods=max(3, width // 3)).std().fillna(0.0).to_numpy(float)
            result[f"offline_smooth__{feature}"] = smooth
            result[f"offline_d1__{feature}"] = derivative
            result[f"offline_d2__{feature}"] = curvature
            result[f"offline_vol__{feature}"] = volatility
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def local_monotonicity_audit(frame: pd.DataFrame, config: MultiStageTrajectoryConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    features = tuple(dict.fromkeys(feature for values in FEATURE_CONFIGS.values() for feature in values))
    for dataset, group in frame.groupby("dataset", sort=True):
        group = group.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
        cycles = group.center_cycle.to_numpy(float); total_span = max(float(np.ptp(cycles)), 1e-12)
        for feature_index, feature in enumerate(features):
            values = group[feature].to_numpy(float)
            for fraction in config.audit_window_fractions:
                window_rows: list[dict[str, object]] = []
                for ordinal, (start, end) in enumerate(_window_positions(len(group), fraction, config.audit_step_fraction)):
                    x, y = cycles[start:end], values[start:end]
                    spearman = stats.spearmanr(x, y)
                    kendall = stats.kendalltau(x, y)
                    slope = _safe_slope(x, y)
                    sign = int(np.sign(slope))
                    stable = _bootstrap_sign_stability(x, y, config, config.random_seed + feature_index * 100_000 + ordinal)
                    row = {
                        "row_type": "window", "dataset": dataset, "feature": feature, "window_fraction": fraction,
                        "window_ordinal": ordinal, "window_start_cycle": float(x[0]), "window_end_cycle": float(x[-1]),
                        "window_count": int(len(x)), "local_spearman": float(spearman.statistic) if np.isfinite(spearman.statistic) else 0.0,
                        "local_spearman_pvalue": float(spearman.pvalue) if np.isfinite(spearman.pvalue) else 1.0,
                        "local_kendall": float(kendall.statistic) if np.isfinite(kendall.statistic) else 0.0,
                        "local_kendall_pvalue": float(kendall.pvalue) if np.isfinite(kendall.pvalue) else 1.0,
                        "robust_local_slope": slope, "slope_sign": sign, "bootstrap_sign_stability": stable,
                    }
                    window_rows.append(row); rows.append(row)
                windows = pd.DataFrame(window_rows)
                positive = (windows.local_spearman_pvalue < config.local_audit_alpha) & (windows.robust_local_slope > 0)
                negative = (windows.local_spearman_pvalue < config.local_audit_alpha) & (windows.robust_local_slope < 0)
                def longest_span(mask: np.ndarray) -> float:
                    best = 0.0; start_cycle: float | None = None; last_end: float | None = None
                    for keep, item in zip(mask, window_rows):
                        if keep:
                            if start_cycle is None:
                                start_cycle = float(item["window_start_cycle"])
                            last_end = float(item["window_end_cycle"])
                        elif start_cycle is not None:
                            best = max(best, float(last_end - start_cycle)); start_cycle = None; last_end = None
                    if start_cycle is not None and last_end is not None:
                        best = max(best, float(last_end - start_cycle))
                    return best / total_span
                pos_span = longest_span(positive.to_numpy(bool)); neg_span = longest_span(negative.to_numpy(bool))
                passed = bool(positive.mean() >= .20 and negative.mean() >= .20 and pos_span > .05 and neg_span > .05)
                rows.append({
                    "row_type": "summary", "dataset": dataset, "feature": feature, "window_fraction": fraction,
                    "positive_significant_fraction": float(positive.mean()), "negative_significant_fraction": float(negative.mean()),
                    "positive_longest_span_fraction": pos_span, "negative_longest_span_fraction": neg_span,
                    "mean_bootstrap_sign_stability": float(windows.bootstrap_sign_stability.mean()),
                    "PERSISTENT_DIRECTION_REVERSAL": "PASS" if passed else "FAIL", "passed": int(passed),
                })
    return pd.DataFrame(rows)


def _ranker_pairs(values: np.ndarray, maximum: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    later = np.arange(1, len(values), dtype=int)
    earlier = np.asarray([rng.integers(0, value) for value in later], dtype=int)
    if len(later) > maximum:
        chosen = np.sort(rng.choice(np.arange(len(later)), size=maximum, replace=False)); later, earlier = later[chosen], earlier[chosen]
    delta = values[later] - values[earlier]
    return np.vstack((delta, -delta)), np.concatenate((np.ones(len(delta), dtype=int), np.zeros(len(delta), dtype=int)))


def _coefficient_runs(signs: Iterable[int], minimum: int) -> bool:
    values = list(signs); pos = neg = 0; run_sign = 0; run = 0
    for sign in values:
        if sign and sign == run_sign:
            run += 1
        elif sign:
            run_sign, run = sign, 1
        else:
            run_sign, run = 0, 0
        if run >= minimum:
            pos = max(pos, int(run_sign > 0)); neg = max(neg, int(run_sign < 0))
    return bool(pos and neg)


def rolling_ranker_direction_audit(frame: pd.DataFrame, config: MultiStageTrajectoryConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in frame.groupby("dataset", sort=True):
        group = group.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
        for config_index, (feature_config, features) in enumerate(FEATURE_CONFIGS.items()):
            blocks: list[dict[str, object]] = []
            for block_id, (start, end) in enumerate(_window_positions(len(group), config.rolling_ranker_block_fraction, config.rolling_ranker_step_fraction)):
                raw = group.loc[start:end - 1, list(features)].to_numpy(float)
                location, scale = robust_scale(raw); z = (raw - location) / scale
                design, labels = _ranker_pairs(z, config.rolling_ranker_max_pairs, np.random.default_rng(config.random_seed + 1_000 * config_index + block_id))
                model = LogisticRegression(C=config.rolling_ranker_c, penalty="l2", fit_intercept=False, max_iter=500, solver="lbfgs", random_state=config.random_seed)
                model.fit(design, labels); coefficients = model.coef_.reshape(-1)
                row = {
                    "row_type": "block", "dataset": dataset, "feature_configuration": feature_config, "block_id": block_id,
                    "block_start_cycle": float(group.center_cycle.iloc[start]), "block_end_cycle": float(group.center_cycle.iloc[end - 1]),
                    "block_windows": int(end - start), "pair_count": int(len(labels) // 2), "coefficients_json": json.dumps(coefficients.tolist()),
                }
                for name, coefficient in zip(features, coefficients):
                    row[f"coef__{name}"] = float(coefficient)
                blocks.append(row); rows.append(row)
            adjacent_cosines: list[float] = []
            distant_cosines: list[float] = []
            for left in range(len(blocks)):
                a = np.asarray(json.loads(str(blocks[left]["coefficients_json"])), dtype=float)
                for right in range(left + 1, len(blocks)):
                    b = np.asarray(json.loads(str(blocks[right]["coefficients_json"])), dtype=float)
                    cosine = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
                    kind = "adjacent" if right == left + 1 else "distant"
                    rows.append({"row_type": "cosine", "dataset": dataset, "feature_configuration": feature_config, "left_block": left, "right_block": right, "comparison": kind, "coefficient_cosine": cosine})
                    (adjacent_cosines if kind == "adjacent" else distant_cosines).append(cosine)
            stable_reversal = any(_coefficient_runs([int(np.sign(float(item[f"coef__{feature}"]))) for item in blocks], config.rolling_ranker_persistent_blocks) for feature in features)
            adjacent_median = float(np.median(adjacent_cosines)) if adjacent_cosines else 1.0
            adjacent_negative = float(np.mean(np.asarray(adjacent_cosines) < 0)) if adjacent_cosines else 0.0
            passed = bool(adjacent_median < .5 or adjacent_negative >= .2 or stable_reversal)
            rows.append({
                "row_type": "summary", "dataset": dataset, "feature_configuration": feature_config, "adjacent_cosine_median": adjacent_median,
                "adjacent_negative_fraction": adjacent_negative, "distant_cosine_median": float(np.median(distant_cosines)) if distant_cosines else np.nan,
                "stable_major_coefficient_reversal": int(stable_reversal), "RANK_DIRECTION_INSTABILITY": "PASS" if passed else "FAIL", "passed": int(passed),
            })
    return pd.DataFrame(rows)


def _recurrence_ratio(values: np.ndarray, cycles: np.ndarray, neighbours: int, exclusion: float) -> tuple[float, list[tuple[int, int]]]:
    n = len(values); k = min(neighbours + 1, n)
    model = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(values)
    indices = model.kneighbors(values, return_distance=False)
    far_pairs: list[tuple[int, int]] = []
    for index, row in enumerate(indices):
        for neighbour in row:
            if neighbour != index and abs(cycles[index] - cycles[neighbour]) >= exclusion:
                far_pairs.append((index, int(neighbour))); break
    return float(len(far_pairs) / max(n, 1)), far_pairs


def _block_permuted_cycles(cycles: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    block = max(1, int(np.ceil(len(cycles) * fraction)))
    chunks = [np.arange(start, min(len(cycles), start + block)) for start in range(0, len(cycles), block)]
    order = rng.permutation(len(chunks))
    return cycles[np.concatenate([chunks[int(index)] for index in order])]


def _random_walk(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    difference = np.diff(values, axis=0)
    covariance = np.cov(difference.T) if len(difference) > 2 else np.eye(values.shape[1])
    covariance = np.atleast_2d(covariance) + np.eye(values.shape[1]) * 1e-6
    steps = rng.multivariate_normal(np.zeros(values.shape[1]), covariance, size=len(values))
    return np.cumsum(steps, axis=0)


def trajectory_recurrence_audit(frame: pd.DataFrame, config: MultiStageTrajectoryConfig, figures: object) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    features = FEATURE_CONFIGS[config.primary_feature_config]
    for dataset, group in frame.groupby("dataset", sort=True):
        sampled = equal_time_sample(group, config.recurrence_max_samples)
        values = sampled.loc[:, list(features)].to_numpy(float); location, scale = robust_scale(values); z = (values - location) / scale
        cycles = sampled.center_cycle.to_numpy(float); span = max(float(np.ptp(cycles)), 1e-12); exclusion = span * config.recurrence_exclusion_fraction
        pca = PCA(n_components=2, random_state=config.random_seed).fit_transform(z)
        observed, pairs = _recurrence_ratio(z, cycles, config.recurrence_neighbours, exclusion)
        rng = np.random.default_rng(config.random_seed + (1 if dataset == "Exp1" else 2))
        permutation_null = [_recurrence_ratio(z, _block_permuted_cycles(cycles, config.recurrence_exclusion_fraction, rng), config.recurrence_neighbours, exclusion)[0] for _ in range(config.recurrence_null_replicates)]
        walk_null = [_recurrence_ratio(_random_walk(z, rng), cycles, config.recurrence_neighbours, exclusion)[0] for _ in range(config.recurrence_null_replicates)]
        p_perm = float(np.quantile(permutation_null, .95)); p_walk = float(np.quantile(walk_null, .95)); passed = bool(observed > p_perm and observed > p_walk)
        rows.append({"row_type": "summary", "dataset": dataset, "feature_configuration": config.primary_feature_config, "sample_count": int(len(sampled)), "far_time_exclusion_cycles": exclusion, "observed_far_time_neighbour_ratio": observed, "time_block_permutation_null_q95": p_perm, "random_walk_null_q95": p_walk, "RECURRENT_OBSERVATION_STATE": "PASS" if passed else "FAIL", "passed": int(passed)})
        for name, values_null in (("time_block_permutation", permutation_null), ("random_walk", walk_null)):
            for replicate, value in enumerate(values_null):
                rows.append({"row_type": "null", "dataset": dataset, "null_type": name, "replicate": replicate, "far_time_neighbour_ratio": float(value)})
        fig, ax = plt.subplots(figsize=(8, 6)); scatter = ax.scatter(pca[:, 0], pca[:, 1], c=cycles, s=8, cmap="viridis")
        stride = max(1, len(pca) // 40); ax.quiver(pca[::stride, 0], pca[::stride, 1], np.gradient(pca[:, 0])[::stride], np.gradient(pca[:, 1])[::stride], angles="xy", scale_units="xy", scale=1, width=.002, alpha=.45)
        fig.colorbar(scatter, ax=ax, label="cycle"); ax.set(title=f"{dataset} PCA trajectory (offline audit)", xlabel="PC1", ylabel="PC2"); fig.tight_layout(); fig.savefig(getattr(figures, "__truediv__")(f"pca_trajectory_{dataset.lower()}_v2.png"), dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 6)); ax.scatter(pca[:, 0], pca[:, 1], c="lightgray", s=7)
        for left, right in pairs[:80]: ax.plot([pca[left, 0], pca[right, 0]], [pca[left, 1], pca[right, 1]], color="tab:red", alpha=.2, linewidth=.6)
        ax.set(title=f"{dataset} far-time nearest-neighbour examples", xlabel="PC1", ylabel="PC2"); fig.tight_layout(); fig.savefig(getattr(figures, "__truediv__")(f"trajectory_recurrence_examples_{dataset.lower()}_v2.png"), dpi=150); plt.close(fig)
    # Required aggregate example figure also has the task-book filename.
    exp1 = getattr(figures, "__truediv__")("trajectory_recurrence_examples_exp1_v2.png")
    if exp1.exists():
        import shutil
        shutil.copyfile(exp1, getattr(figures, "__truediv__")("trajectory_recurrence_examples_v2.png"))
    return pd.DataFrame(rows)


def _segment_bic(values: np.ndarray, bkps: list[int]) -> float:
    start = 0; sse = 0.0
    for end in bkps:
        segment = values[start:end]
        if len(segment): sse += float(((segment - segment.mean(axis=0)) ** 2).sum())
        start = end
    n, dimensions = values.shape; parameters = max(1, (len(bkps) - 1) * dimensions)
    return float(n * np.log(max(sse / max(n * dimensions, 1), 1e-12)) + parameters * np.log(max(n, 2)))


def _fit_change_points(values: np.ndarray, config: MultiStageTrajectoryConfig, method: str) -> tuple[list[int], float, object]:
    min_size = max(3, int(np.ceil(len(values) * config.cp_min_segment_fraction)))
    choices: list[tuple[float, list[int], object]] = []
    if method == "Pelt_rbf":
        fitted = rpt.Pelt(model="rbf", min_size=min_size).fit(values)
        for penalty in config.cp_pelt_penalties:
            try:
                bkps = list(map(int, fitted.predict(pen=float(penalty))))
            except Exception:
                continue
            choices.append((_segment_bic(values, bkps), bkps, float(penalty)))
    elif method == "Binseg_l2":
        fitted = rpt.Binseg(model="l2", min_size=min_size).fit(values)
        for count in config.cp_binseg_counts:
            try:
                bkps = list(map(int, fitted.predict(n_bkps=int(count))))
            except Exception:
                continue
            choices.append((_segment_bic(values, bkps), bkps, int(count)))
    else:
        raise ValueError(method)
    if not choices:
        return [len(values)], float("inf"), None
    best = min(choices, key=lambda item: item[0])
    return best[1], best[0], best[2]


def _clusters(cycles: list[float], tolerance: float) -> list[list[float]]:
    result: list[list[float]] = []
    for cycle in sorted(cycles):
        if not result or cycle - np.mean(result[-1]) > tolerance:
            result.append([cycle])
        else:
            result[-1].append(cycle)
    return result


def change_point_audit(preprocessed: pd.DataFrame, config: MultiStageTrajectoryConfig, figures: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates: list[dict[str, object]] = []
    for dataset, group in preprocessed.groupby("dataset", sort=True):
        for feature_index, (feature_config, features) in enumerate(FEATURE_CONFIGS.items()):
            selected = equal_time_sample(group, config.cp_max_samples)
            cycles = selected.center_cycle.to_numpy(float); raw = selected.loc[:, [f"offline_smooth__{feature}" for feature in features]].to_numpy(float)
            location, scale = robust_scale(raw); values = (raw - location) / scale
            for method_index, method in enumerate(("Pelt_rbf", "Binseg_l2")):
                bkps, bic, setting = _fit_change_points(values, config, method)
                for bkp in bkps[:-1]:
                    candidates.append({"row_type": "base", "dataset": dataset, "feature_configuration": feature_config, "method": method, "selection_setting": setting, "selection_bic": bic, "sample_index": int(bkp), "center_cycle": float(cycles[min(bkp, len(cycles) - 1)]), "offline_diagnostic_only": True})
                seed = config.random_seed + (1 if dataset == "Exp1" else 2) * 10_000 + feature_index * 100 + method_index
                rng = np.random.default_rng(seed)
                for replicate in range(config.cp_bootstrap_replicates):
                    index = _block_resample_indices(len(values), config.bootstrap_block_fraction, rng)
                    boot_values = values[index]
                    boot_cycles = cycles[index]
                    try:
                        boot_bkps, boot_bic, boot_setting = _fit_change_points(boot_values, config, method)
                    except Exception as exc:
                        candidates.append({"row_type": "bootstrap_error", "dataset": dataset, "feature_configuration": feature_config, "method": method, "replicate": replicate, "error": type(exc).__name__, "offline_diagnostic_only": True}); continue
                    for bkp in boot_bkps[:-1]:
                        candidates.append({"row_type": "bootstrap", "dataset": dataset, "feature_configuration": feature_config, "method": method, "replicate": replicate, "selection_setting": boot_setting, "selection_bic": boot_bic, "sample_index": int(bkp), "center_cycle": float(boot_cycles[min(bkp, len(boot_cycles) - 1)]), "offline_diagnostic_only": True})
    candidate_frame = pd.DataFrame(candidates)
    consensus_rows: list[dict[str, object]] = []
    for dataset, base in candidate_frame.loc[candidate_frame.row_type.eq("base")].groupby("dataset", sort=True):
        span = float(preprocessed.loc[preprocessed.dataset.eq(dataset), "center_cycle"].max() - preprocessed.loc[preprocessed.dataset.eq(dataset), "center_cycle"].min())
        tolerance = max(span * config.cp_consensus_tolerance_fraction, 1e-12)
        for cluster_id, cluster in enumerate(_clusters(base.center_cycle.astype(float).tolist(), tolerance)):
            centre = float(np.median(cluster)); part = base.loc[np.abs(base.center_cycle.astype(float) - centre) <= tolerance]
            bootstrap = candidate_frame.loc[(candidate_frame.dataset.eq(dataset)) & candidate_frame.row_type.eq("bootstrap")]
            combinations = [(method, feature) for method in ("Pelt_rbf", "Binseg_l2") for feature in FEATURE_CONFIGS]
            hits = 0
            for method, feature in combinations:
                subset = bootstrap.loc[(bootstrap.method.eq(method)) & (bootstrap.feature_configuration.eq(feature))]
                for replicate in range(config.cp_bootstrap_replicates):
                    one = subset.loc[subset.replicate.eq(replicate), "center_cycle"].to_numpy(float)
                    hits += int(len(one) > 0 and np.any(np.abs(one - centre) <= tolerance))
            support = hits / max(len(combinations) * config.cp_bootstrap_replicates, 1)
            position_error = float((np.quantile(cluster, .75) - np.quantile(cluster, .25)) / 2 / max(span, 1e-12)) if len(cluster) > 1 else 0.0
            methods = int(part.method.nunique()); feature_count = int(part.feature_configuration.nunique())
            passed = bool(methods >= 2 and feature_count >= 2 and support >= .60 and position_error <= .03)
            consensus_rows.append({"dataset": dataset, "consensus_id": cluster_id, "center_cycle": centre, "method_count": methods, "feature_configuration_count": feature_count, "bootstrap_support_rate": support, "position_error_fraction": position_error, "tolerance_fraction": config.cp_consensus_tolerance_fraction, "REPRODUCIBLE_CHANGE_POINT": "PASS" if passed else "FAIL", "passed": int(passed)})
        fig, ax = plt.subplots(figsize=(10, 4)); primary = equal_time_sample(preprocessed.loc[preprocessed.dataset.eq(dataset)], config.cp_max_samples)
        values = primary[[f"offline_smooth__{name}" for name in FEATURE_CONFIGS[config.primary_feature_config]]].to_numpy(float); values = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-9)
        trace = PCA(n_components=1, random_state=config.random_seed).fit_transform(values).reshape(-1)
        ax.plot(primary.center_cycle, trace, linewidth=.8, color="steelblue", label="offline diagnostic PC1")
        for row in [row for row in consensus_rows if row["dataset"] == dataset]: ax.axvline(float(row["center_cycle"]), color="tab:red" if row["passed"] else "tab:orange", alpha=.8)
        ax.set(title=f"{dataset} change-point consensus (offline audit)", xlabel="cycle", ylabel="PC1"); fig.tight_layout(); fig.savefig(getattr(figures, "__truediv__")(f"change_point_consensus_{dataset.lower()}_v2.png"), dpi=150); plt.close(fig)
    return candidate_frame, pd.DataFrame(consensus_rows)


def multistage_decision(local: pd.DataFrame, ranker: pd.DataFrame, recurrence: pd.DataFrame, consensus: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {"evidence": {}, "decision_rule": {"PASS": "3/4 or more", "QUALIFIED PASS": "2/4", "FAIL": "fewer than 2/4"}}
    datasets = sorted(set(local.dataset.dropna()) | set(ranker.dataset.dropna()) | set(recurrence.dataset.dropna()) | set(consensus.dataset.dropna()))
    for dataset in datasets:
        reversal = bool((local.loc[(local.dataset.eq(dataset)) & local.row_type.eq("summary"), "passed"].fillna(0).astype(int) > 0).any())
        instability = bool((ranker.loc[(ranker.dataset.eq(dataset)) & ranker.row_type.eq("summary"), "passed"].fillna(0).astype(int) > 0).any())
        recurrent = bool((recurrence.loc[(recurrence.dataset.eq(dataset)) & recurrence.row_type.eq("summary"), "passed"].fillna(0).astype(int) > 0).any())
        change = bool((consensus.loc[consensus.dataset.eq(dataset), "passed"].fillna(0).astype(int) > 0).any()) if not consensus.empty else False
        count = sum((reversal, instability, recurrent, change)); status = "PASS" if count >= 3 else "QUALIFIED PASS" if count == 2 else "FAIL"
        result["evidence"][str(dataset)] = {"PERSISTENT_DIRECTION_REVERSAL": "PASS" if reversal else "FAIL", "RANK_DIRECTION_INSTABILITY": "PASS" if instability else "FAIL", "RECURRENT_OBSERVATION_STATE": "PASS" if recurrent else "FAIL", "REPRODUCIBLE_CHANGE_POINT": "PASS" if change else "FAIL", "evidence_pass_count": count, "status": status}
    return result
