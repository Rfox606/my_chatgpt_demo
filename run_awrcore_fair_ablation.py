from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_weighted_awrcore_models import (
    STABLE_PLUS_FEATURES,
    WeightedAWRConfig,
    average_precision,
    bootstrap_direction_stability,
    cliffs_delta,
    cohen_d,
    finite_median,
    finite_quantile,
    roc_auc,
    save_figure,
    source_split_by_stage,
    spearman_corr,
)


@dataclass
class FairAblationConfig:
    output_dir: str = "outputs_awrcore_fair_ablation_v1"
    weighted_dir: str = "outputs_weighted_awrcore_v1"
    v2_dir: str = "outputs_aux_state_metrics_v2"
    baseline_cycles: int = 500
    source_gap_windows: int = 4
    high_awr_threshold_percentile: float = 95.0
    bootstrap_n: int = 500
    bootstrap_block_size: int = 20
    redundancy_threshold: float = 0.95
    redundancy_downweight: float = 0.5
    random_seed: int = 20260707
    bd_default_metric: str = "BDall_xy_v2"


FORMULATION_LABELS = {
    "mean_z": "mean-z",
    "direction_mean_z": "direction-corrected mean-z",
    "weighted_direction": "weighted direction-corrected",
}
MODEL_COLORS = {
    "mean_z": "#7a7a7a",
    "direction_mean_z": "#3b6ea8",
    "weighted_direction": "#46a67a",
}
REGION_COLORS = {
    "low_AWR_low_BD": "#7c8a99",
    "low_AWR_high_BD": "#d4a72c",
    "high_AWR_high_BD": "#b64b5a",
    "high_AWR_low_BD": "#5a6fb0",
}


def setup_dirs(config: FairAblationConfig) -> Dict[str, Path]:
    root = Path(config.output_dir)
    dirs = {
        "root": root,
        "results": root / "results",
        "figures": root / "figures",
        "reports": root / "reports",
        "configs": root / "configs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(dirs: Dict[str, Path]) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(dirs["root"] / "fair_ablation_run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def dataframe_to_markdown(frame: pd.DataFrame, max_rows: int = 60) -> str:
    if frame.empty:
        return "No rows."
    shown = frame.head(max_rows).copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4g}")
        else:
            shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else str(value))
    lines = [
        "| " + " | ".join(map(str, shown.columns)) + " |",
        "| " + " | ".join(["---"] * len(shown.columns)) + " |",
    ]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    if len(frame) > max_rows:
        lines.append(f"\nShowing first {max_rows} of {len(frame)} rows.")
    return "\n".join(lines)


def finite_values(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def load_z_wide(config: FairAblationConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(config.weighted_dir) / "results" / "window_feature_z_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing z feature table: {path}")
    logging.info("Reading z feature table: %s", path)
    z_long = pd.read_csv(path)
    id_cols = [
        "dataset",
        "window_id",
        "window_index",
        "start_cycle",
        "end_cycle",
        "center_cycle",
        "stage",
        "stage_label",
        "baseline_window",
    ]
    meta = (
        z_long[["feature_name", "channel", "feature_family", "physical_meaning"]]
        .drop_duplicates("feature_name")
        .sort_values("feature_name")
        .reset_index(drop=True)
    )
    z_wide = (
        z_long.pivot_table(index=id_cols, columns="feature_name", values="z_value", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
        .sort_values(["dataset", "window_index"])
        .reset_index(drop=True)
    )
    return z_wide, meta


def build_feature_groups(meta: pd.DataFrame) -> Dict[str, List[str]]:
    features = meta["feature_name"].astype(str).tolist()
    by_channel = {
        channel: sorted(meta.loc[meta["channel"].astype(str) == channel, "feature_name"].astype(str).tolist())
        for channel in ("rx", "ry", "rs")
    }
    stable_no_corr = [f for f in STABLE_PLUS_FEATURES if f not in {"rs_corrdist_base", "rx_corrdist_base"}]
    return {
        "stable_plus": list(STABLE_PLUS_FEATURES),
        "stable_plus_no_corrdist": stable_no_corr,
        "rx_only": by_channel["rx"],
        "ry_only": by_channel["ry"],
        "rs_only": by_channel["rs"],
        "rx_ry": by_channel["rx"] + by_channel["ry"],
        "rx_ry_rs": features,
    }


def direction_protocol(
    direction_id: str,
    source_dataset: str,
    target_dataset: str,
    z_wide: pd.DataFrame,
    meta: pd.DataFrame,
    config: FairAblationConfig,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    source = z_wide[z_wide["dataset"].astype(str) == source_dataset].sort_values("window_index").reset_index(drop=True)
    features = meta["feature_name"].astype(str).tolist()
    stages = source["stage"].to_numpy(dtype=int)
    train_mask, val_mask = source_split_by_stage(stages, config.source_gap_windows)
    early_mask = train_mask & (stages == 1)
    late_mask = train_mask & (stages == 5)
    if int(early_mask.sum()) < 20:
        early_mask = train_mask & (stages <= 1)
    if int(late_mask.sum()) < 20:
        late_mask = train_mask & (stages >= 4)

    early = source.loc[early_mask, features].to_numpy(dtype=float)
    late = source.loc[late_mask, features].to_numpy(dtype=float)
    rows = []
    signs = []
    for pos, feature in enumerate(features):
        early_values = early[:, pos]
        late_values = late[:, pos]
        early_med = finite_median(early_values, np.nan)
        late_med = finite_median(late_values, np.nan)
        delta = late_med - early_med
        sign = 0 if (not np.isfinite(delta) or delta == 0.0) else int(np.sign(delta))
        signs.append(sign)
        c_d = cohen_d(early_values, late_values)
        c_delta = cliffs_delta(early_values, late_values)
        abs_effect = abs(c_delta) if np.isfinite(c_delta) else abs(c_d)
        mrow = meta[meta["feature_name"].astype(str) == feature].iloc[0]
        rows.append(
            {
                "direction_id": direction_id,
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "feature_name": feature,
                "channel": mrow["channel"],
                "feature_family": mrow["feature_family"],
                "early_median": early_med,
                "late_median": late_med,
                "delta_median": float(delta) if np.isfinite(delta) else np.nan,
                "direction_sign": int(sign),
                "cohen_d": float(c_d) if np.isfinite(c_d) else np.nan,
                "cliffs_delta": float(c_delta) if np.isfinite(c_delta) else np.nan,
                "abs_effect_size": float(abs_effect) if np.isfinite(abs_effect) else 0.0,
                "physical_meaning": mrow["physical_meaning"],
            }
        )
    weighted_cfg = WeightedAWRConfig(
        bootstrap_n=config.bootstrap_n,
        bootstrap_block_size=config.bootstrap_block_size,
        random_seed=config.random_seed,
    )
    stability = bootstrap_direction_stability(early, late, np.asarray(signs, dtype=int), weighted_cfg)
    table = pd.DataFrame(rows)
    table["direction_stability"] = stability
    table["enter_model"] = np.where(table["direction_sign"].astype(int) != 0, "direction_and_weighted", "mean_z_only")
    return table, train_mask, val_mask


def compute_redundancy(
    source: pd.DataFrame,
    features: List[str],
    direction_table: pd.DataFrame,
    train_mask: np.ndarray,
    config: FairAblationConfig,
) -> Dict[str, Tuple[float, str]]:
    if not features:
        return {}
    sub = direction_table.set_index("feature_name").loc[features]
    rank = (sub["abs_effect_size"].astype(float) * sub["direction_stability"].astype(float)).sort_values(ascending=False)
    signed = {}
    for feature in rank.index.astype(str).tolist():
        sign = int(sub.loc[feature, "direction_sign"])
        signed[feature] = sign * source.loc[train_mask, feature].to_numpy(dtype=float)
    factors = {feature: 1.0 for feature in features}
    notes = {feature: "" for feature in features}
    kept: List[str] = []
    for feature in rank.index.astype(str).tolist():
        for prev in kept:
            x = signed[feature]
            y = signed[prev]
            ok = np.isfinite(x) & np.isfinite(y)
            corr = 0.0
            if int(ok.sum()) > 5 and np.nanstd(x[ok]) > 1e-12 and np.nanstd(y[ok]) > 1e-12:
                corr = float(np.corrcoef(x[ok], y[ok])[0, 1])
            if abs(corr) > config.redundancy_threshold:
                factors[feature] *= config.redundancy_downweight
                note = f"downweighted_with_{prev}_r={corr:.3f}"
                notes[feature] = note if not notes[feature] else notes[feature] + ";" + note
        kept.append(feature)
    return {feature: (float(factors[feature]), notes[feature]) for feature in features}


def model_names_for_group(group_name: str) -> Dict[str, str]:
    if group_name == "stable_plus":
        return {
            "mean_z": "M0_stable",
            "direction_mean_z": "M1_stable",
            "weighted_direction": "M2_stable",
        }
    if group_name == "stable_plus_no_corrdist":
        return {
            "mean_z": "M0_stable_no_corrdist",
            "direction_mean_z": "M1_stable_no_corrdist",
            "weighted_direction": "M2_stable_no_corrdist",
        }
    return {
        "mean_z": f"{group_name}_mean_z",
        "direction_mean_z": f"{group_name}_direction_mean_z",
        "weighted_direction": f"{group_name}_weighted_direction",
    }


def model_scores(
    z: pd.DataFrame,
    features: List[str],
    direction_table: pd.DataFrame,
    redundancy: Dict[str, Tuple[float, str]],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    n = len(features)
    if n == 0:
        zero = np.zeros(len(z), dtype=float)
        return {"mean_z": zero, "direction_mean_z": zero, "weighted_direction": zero}, pd.DataFrame()
    d = direction_table.set_index("feature_name").loc[features]
    X = z[features].to_numpy(dtype=float)
    signs = d["direction_sign"].to_numpy(dtype=float)
    signs_for_mean = np.ones(n, dtype=float)
    equal_weights = np.full(n, 1.0 / n, dtype=float)
    raw = (
        d["abs_effect_size"].to_numpy(dtype=float)
        * d["direction_stability"].to_numpy(dtype=float)
        * np.asarray([redundancy.get(feature, (1.0, ""))[0] for feature in features], dtype=float)
        * (signs != 0).astype(float)
    )
    if np.isfinite(raw).any() and float(np.nansum(raw)) > 0:
        weighted = raw / float(np.nansum(raw))
    else:
        weighted = np.zeros(n, dtype=float)
    signed_X = X * signs.reshape(1, -1)
    scores = {
        "mean_z": np.nansum(X * equal_weights.reshape(1, -1), axis=1),
        "direction_mean_z": np.nansum(signed_X * equal_weights.reshape(1, -1), axis=1),
        "weighted_direction": np.nansum(signed_X * weighted.reshape(1, -1), axis=1),
    }
    weight_rows = []
    for feature, sign, equal_w, raw_w, norm_w in zip(features, signs, equal_weights, raw, weighted):
        red_factor, red_note = redundancy.get(feature, (1.0, ""))
        weight_rows.append(
            {
                "feature_name": feature,
                "direction_sign": int(sign),
                "equal_weight": float(equal_w),
                "raw_weight": float(raw_w),
                "normalized_weight": float(norm_w),
                "redundancy_factor": float(red_factor),
                "redundancy_notes": red_note,
            }
        )
    return scores, pd.DataFrame(weight_rows)


def evaluate_model(
    direction_id: str,
    source_dataset: str,
    target_dataset: str,
    model_name: str,
    feature_group: str,
    formulation: str,
    scores: pd.DataFrame,
    val_mask: np.ndarray,
    state_v2: pd.DataFrame,
    bd_thresholds: pd.DataFrame,
    config: FairAblationConfig,
) -> Dict[str, object]:
    source = scores[scores["dataset"].astype(str) == source_dataset].sort_values("window_index").reset_index(drop=True)
    target = scores[scores["dataset"].astype(str) == target_dataset].sort_values("window_index").reset_index(drop=True)
    if len(val_mask) != len(source) or int(val_mask.sum()) < 20:
        val_mask = np.ones(len(source), dtype=bool)
    threshold = finite_quantile(source.loc[val_mask, model_name], config.high_awr_threshold_percentile, default=np.nan)
    if not np.isfinite(threshold):
        threshold = finite_quantile(source[model_name], config.high_awr_threshold_percentile, default=0.0)

    target_eval = target[["dataset", "window_index", "stage", model_name]].copy()
    bd_cols = ["dataset", "window_index", config.bd_default_metric]
    target_eval = target_eval.merge(state_v2[bd_cols], on=["dataset", "window_index"], how="left")
    default_bd = bd_thresholds[bd_thresholds["bd_metric"].astype(str) == config.bd_default_metric]
    bd_lookup = {str(row.dataset): float(row.BD_major_threshold) for row in default_bd.itertuples(index=False)}
    score = target_eval[model_name].to_numpy(dtype=float)
    stage = target_eval["stage"].to_numpy(dtype=int)
    y = (stage == 5).astype(int)
    high = score >= float(threshold)
    stage5 = stage == 5
    bd_high = target_eval[config.bd_default_metric].to_numpy(dtype=float) >= bd_lookup.get(target_dataset, np.inf)
    stage1_values = score[stage == 1]
    stage5_values = score[stage == 5]
    return {
        "model_name": model_name,
        "feature_group": feature_group,
        "formulation": formulation,
        "direction_id": direction_id,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "target_AUROC": roc_auc(y, score),
        "target_AUPRC": average_precision(y, score),
        "AUPRC_baseline": float(np.nanmean(y)),
        "Spearman_stage_AWR": spearman_corr(stage, score),
        "Stage1_median_AWR": finite_median(stage1_values, np.nan),
        "Stage5_median_AWR": finite_median(stage5_values, np.nan),
        "ScoreGap": finite_median(stage5_values, np.nan) - finite_median(stage1_values, np.nan),
        "source_threshold_P95": float(threshold),
        "target_Stage5_high_AWR_rate": float(np.mean(high[stage5])) if int(stage5.sum()) else np.nan,
        "target_Stage5_high_AWR_high_BD_occupancy": float(np.mean(high[stage5] & bd_high[stage5])) if int(stage5.sum()) else np.nan,
        "notes": "Stage5 is used only as a late-state proxy label.",
    }


def add_summary_stats(rows: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for model, group in rows.groupby("model_name", sort=False):
        first = group.iloc[0]
        out_rows.append(
            {
                "model_name": model,
                "feature_group": first["feature_group"],
                "formulation": first["formulation"],
                "mean_AUROC": float(np.nanmean(group["target_AUROC"])),
                "worst_AUROC": float(np.nanmin(group["target_AUROC"])),
                "mean_AUPRC": float(np.nanmean(group["target_AUPRC"])),
                "worst_AUPRC": float(np.nanmin(group["target_AUPRC"])),
                "mean_Spearman": float(np.nanmean(group["Spearman_stage_AWR"])),
                "worst_Spearman": float(np.nanmin(group["Spearman_stage_AWR"])),
                "mean_ScoreGap": float(np.nanmean(group["ScoreGap"])),
                "worst_ScoreGap": float(np.nanmin(group["ScoreGap"])),
                "mean_Stage5_high_AWR_high_BD_occupancy": float(
                    np.nanmean(group["target_Stage5_high_AWR_high_BD_occupancy"])
                ),
            }
        )
    return pd.DataFrame(out_rows).sort_values(["worst_AUROC", "worst_AUPRC"], ascending=False).reset_index(drop=True)


def compute_all_ablation_scores(
    z_wide: pd.DataFrame,
    meta: pd.DataFrame,
    state_v2: pd.DataFrame,
    bd_thresholds: pd.DataFrame,
    config: FairAblationConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    groups = build_feature_groups(meta)
    directions = [("Exp1_to_Exp2", "Exp1", "Exp2"), ("Exp2_to_Exp1", "Exp2", "Exp1")]
    score_frames = []
    eval_rows = []
    direction_tables = []
    weight_rows_all = []
    split_masks: Dict[str, np.ndarray] = {}

    for direction_id, source_dataset, target_dataset in directions:
        direction_table, train_mask, val_mask = direction_protocol(
            direction_id, source_dataset, target_dataset, z_wide, meta, config
        )
        direction_tables.append(direction_table)
        split_masks[direction_id] = val_mask
        source_z = z_wide[z_wide["dataset"].astype(str) == source_dataset].sort_values("window_index").reset_index(drop=True)

        direction_score_frames = []
        for dataset in (source_dataset, target_dataset):
            z = z_wide[z_wide["dataset"].astype(str) == dataset].sort_values("window_index").reset_index(drop=True)
            base = z[
                [
                    "dataset",
                    "window_id",
                    "window_index",
                    "start_cycle",
                    "end_cycle",
                    "center_cycle",
                    "stage",
                    "stage_label",
                ]
            ].copy()
            base["direction_id"] = direction_id
            base["source_dataset"] = source_dataset
            base["target_dataset"] = target_dataset

            for group_name, features in groups.items():
                model_map = model_names_for_group(group_name)
                redundancy = compute_redundancy(source_z, features, direction_table, train_mask, config)
                group_scores, weights = model_scores(z, features, direction_table, redundancy)
                for formulation, values in group_scores.items():
                    base[model_map[formulation]] = values
                if dataset == source_dataset:
                    for formulation, model_name in model_map.items():
                        weights_for_form = weights.copy()
                        if weights_for_form.empty:
                            continue
                        weights_for_form["direction_id"] = direction_id
                        weights_for_form["source_dataset"] = source_dataset
                        weights_for_form["target_dataset"] = target_dataset
                        weights_for_form["feature_group"] = group_name
                        weights_for_form["formulation"] = formulation
                        weights_for_form["model_name"] = model_name
                        d_sub = direction_table[
                            [
                                "feature_name",
                                "channel",
                                "feature_family",
                                "abs_effect_size",
                                "direction_stability",
                                "early_median",
                                "late_median",
                                "physical_meaning",
                            ]
                        ]
                        weights_for_form = weights_for_form.merge(d_sub, on="feature_name", how="left")
                        if formulation == "mean_z":
                            weights_for_form["normalized_weight"] = weights_for_form["equal_weight"]
                            weights_for_form["direction_sign"] = 1
                        elif formulation == "direction_mean_z":
                            weights_for_form["normalized_weight"] = weights_for_form["equal_weight"]
                        weights_for_form["contribution_mean_stage1"] = (
                            weights_for_form["normalized_weight"]
                            * weights_for_form["direction_sign"].astype(float)
                            * weights_for_form["early_median"].astype(float)
                        )
                        weights_for_form["contribution_mean_stage5"] = (
                            weights_for_form["normalized_weight"]
                            * weights_for_form["direction_sign"].astype(float)
                            * weights_for_form["late_median"].astype(float)
                        )
                        weights_for_form["contribution_gap"] = (
                            weights_for_form["contribution_mean_stage5"]
                            - weights_for_form["contribution_mean_stage1"]
                        )
                        weight_rows_all.append(weights_for_form)
            direction_score_frames.append(base)

        direction_scores = pd.concat(direction_score_frames, ignore_index=True)
        score_frames.append(direction_scores)
        model_columns = [
            col
            for col in direction_scores.columns
            if col not in {
                "dataset",
                "window_id",
                "window_index",
                "start_cycle",
                "end_cycle",
                "center_cycle",
                "stage",
                "stage_label",
                "direction_id",
                "source_dataset",
                "target_dataset",
            }
        ]
        for group_name in groups:
            model_map = model_names_for_group(group_name)
            for formulation, model_name in model_map.items():
                if model_name in model_columns:
                    eval_rows.append(
                        evaluate_model(
                            direction_id,
                            source_dataset,
                            target_dataset,
                            model_name,
                            group_name,
                            formulation,
                            direction_scores,
                            val_mask,
                            state_v2,
                            bd_thresholds,
                            config,
                        )
                    )
        logging.info("%s complete: %d models", direction_id, len(model_columns))

    scores_all = pd.concat(score_frames, ignore_index=True)
    eval_df = pd.DataFrame(eval_rows)
    direction_all = pd.concat(direction_tables, ignore_index=True)
    if weight_rows_all:
        weight_all = pd.concat(weight_rows_all, ignore_index=True)
        keep_cols = [
            "direction_id",
            "source_dataset",
            "target_dataset",
            "feature_group",
            "formulation",
            "model_name",
            "feature_name",
            "channel",
            "feature_family",
            "direction_sign",
            "abs_effect_size",
            "direction_stability",
            "redundancy_factor",
            "normalized_weight",
            "contribution_mean_stage1",
            "contribution_mean_stage5",
            "contribution_gap",
            "redundancy_notes",
            "physical_meaning",
        ]
        weight_all = weight_all[keep_cols]
    else:
        weight_all = pd.DataFrame()
    return scores_all, eval_df, direction_all, weight_all, split_masks


def corrdist_dependency(comparison: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    comp = comparison.set_index("model_name")
    metrics = {}
    for form, all_name, no_name in [
        ("mean_z", "M0_stable", "M0_stable_no_corrdist"),
        ("direction_mean_z", "M1_stable", "M1_stable_no_corrdist"),
        ("weighted_direction", "M2_stable", "M2_stable_no_corrdist"),
    ]:
        if all_name in comp.index and no_name in comp.index:
            metrics[f"{form}_worst_AUROC_drop"] = float(comp.loc[all_name, "worst_AUROC"] - comp.loc[no_name, "worst_AUROC"])
            metrics[f"{form}_worst_AUPRC_drop"] = float(comp.loc[all_name, "worst_AUPRC"] - comp.loc[no_name, "worst_AUPRC"])
    max_drop = max([abs(v) for v in metrics.values()], default=0.0)
    if max_drop >= 0.08:
        level = "strong"
    elif max_drop >= 0.03:
        level = "moderate"
    else:
        level = "weak"
    return level, metrics


def make_decision(eval_df: pd.DataFrame, comparison: pd.DataFrame, weight_table: pd.DataFrame) -> Dict[str, object]:
    comp = comparison.set_index("model_name")
    stable_models = ["M0_stable", "M1_stable", "M2_stable"]
    best_stable = max(stable_models, key=lambda model: (comp.loc[model, "worst_AUROC"], comp.loc[model, "worst_AUPRC"]))
    channel_models = comparison[comparison["feature_group"].isin(["rx_only", "ry_only", "rs_only", "rx_ry", "rx_ry_rs", "stable_plus"])]
    best_channel = channel_models.sort_values(["worst_AUROC", "worst_AUPRC"], ascending=False).iloc[0]["model_name"]
    corr_level, corr_metrics = corrdist_dependency(comparison)
    m0_best = bool(best_stable == "M0_stable")
    m1_close = bool(comp.loc["M1_stable", "worst_AUROC"] >= comp.loc["M0_stable", "worst_AUROC"] - 0.03)
    m2_gain = bool(comp.loc["M2_stable", "worst_AUROC"] > comp.loc["M1_stable", "worst_AUROC"] + 0.02)
    no_corr_drop = corr_metrics.get("mean_z_worst_AUROC_drop", np.nan)
    channel_group_rank = (
        comparison[comparison["formulation"] == "mean_z"]
        .sort_values(["worst_AUROC", "worst_AUPRC"], ascending=False)[["model_name", "feature_group", "worst_AUROC", "worst_AUPRC"]]
        .to_dict(orient="records")
    )
    top_contrib = (
        weight_table[weight_table["model_name"] == "M0_stable"]
        .sort_values("contribution_gap", ascending=False)
        .head(8)[["direction_id", "feature_name", "channel", "contribution_gap"]]
        .to_dict(orient="records")
        if not weight_table.empty
        else []
    )
    if m0_best and (not m1_close) and (not m2_gain):
        recommendation = "Use M0_stable as the main AWR score, with direction and channel decompositions as auxiliary interpretation."
    elif best_stable == "M1_stable":
        recommendation = "Use M1_stable if transparent direction correction is prioritized and worst-direction metrics remain acceptable."
    elif best_stable == "M2_stable":
        recommendation = "Use M2_stable only if weighted gains remain stable under physical-loop review."
    else:
        recommendation = "Keep stable_plus as the main score family and use channel/family ablations for interpretation."
    return {
        "M0_stable_is_best_stable_plus": m0_best,
        "best_stable_plus_model": best_stable,
        "M1_stable_close_to_M0": m1_close,
        "M2_stable_has_incremental_gain": m2_gain,
        "corrdist_dependency": corr_level,
        "corrdist_metric_deltas": corr_metrics,
        "no_corrdist_worst_AUROC_drop_from_M0_stable": float(no_corr_drop) if np.isfinite(no_corr_drop) else None,
        "best_channel_family_model": str(best_channel),
        "channel_family_rank_mean_z": channel_group_rank,
        "top_M0_stable_contribution_gaps": top_contrib,
        "recommended_AWR_structure": recommendation,
        "physical_loop_focus": [
            "Windows where M0_stable is high while BDall_xy_v2 is also high.",
            "Contributions from rs_corrdist_base and rx_corrdist_base if corrdist dependency is not weak.",
            "Channel-family intervals where rx/ry/rs rankings disagree.",
        ],
    }


def add_boundaries(ax: plt.Axes, boundaries: pd.DataFrame, dataset: str) -> None:
    if boundaries.empty:
        return
    ds = boundaries[boundaries["dataset"].astype(str) == str(dataset)]
    for row in ds.itertuples(index=False):
        ax.axvline(float(getattr(row, "boundary_cycle")), color="#999999", linestyle="--", linewidth=0.8, alpha=0.55)


def plot_stable_timeseries(scores: pd.DataFrame, boundaries: pd.DataFrame, figures_dir: Path) -> None:
    targets = {"Exp1": "Exp2_to_Exp1", "Exp2": "Exp1_to_Exp2"}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=False)
    for ax, (dataset, direction_id) in zip(axes, targets.items()):
        sub = scores[(scores["dataset"].astype(str) == dataset) & (scores["direction_id"].astype(str) == direction_id)].sort_values("window_index")
        for col, color in [("M0_stable", "#7a7a7a"), ("M1_stable", "#3b6ea8"), ("M2_stable", "#46a67a")]:
            ax.plot(sub["center_cycle"], sub[col], label=col, color=color, linewidth=1.0)
        add_boundaries(ax, boundaries, dataset)
        ax.set_title(f"{dataset}: stable_plus ablation")
        ax.set_ylabel("AWR score")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Cycle")
    save_figure(fig, figures_dir / "fig_stable_plus_ablation_timeseries")


def plot_stable_model_comparison(eval_df: pd.DataFrame, figures_dir: Path) -> None:
    models = ["M0_stable", "M1_stable", "M2_stable"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True)
    x = np.arange(len(models))
    for ax, metric, title in zip(axes, ["target_AUROC", "target_AUPRC"], ["AUROC", "AUPRC"]):
        means = [float(eval_df[eval_df["model_name"] == m][metric].mean()) for m in models]
        worst = [float(eval_df[eval_df["model_name"] == m][metric].min()) for m in models]
        ax.bar(x - 0.18, means, width=0.35, color="#5a84b8", label="Mean")
        ax.bar(x + 0.18, worst, width=0.35, color="#c76d2a", label="Worst direction")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    save_figure(fig, figures_dir / "fig_stable_plus_model_comparison")


def plot_corrdist_ablation(comparison: pd.DataFrame, figures_dir: Path) -> None:
    pairs = [
        ("M0_stable", "M0_stable_no_corrdist"),
        ("M1_stable", "M1_stable_no_corrdist"),
        ("M2_stable", "M2_stable_no_corrdist"),
    ]
    comp = comparison.set_index("model_name")
    labels = ["M0", "M1", "M2"]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, metric, title in zip(axes, ["worst_AUROC", "worst_AUPRC"], ["Worst AUROC", "Worst AUPRC"]):
        all_vals = [float(comp.loc[a, metric]) for a, _ in pairs]
        no_vals = [float(comp.loc[n, metric]) for _, n in pairs]
        ax.bar(x - 0.18, all_vals, width=0.35, color="#5a84b8", label="stable_plus")
        ax.bar(x + 0.18, no_vals, width=0.35, color="#c76d2a", label="no corrdist")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    save_figure(fig, figures_dir / "fig_corrdist_ablation")


def plot_corrdist_contribution(scores: pd.DataFrame, z_wide: pd.DataFrame, figures_dir: Path) -> None:
    targets = {"Exp1": "Exp2_to_Exp1", "Exp2": "Exp1_to_Exp2"}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), sharex=False)
    weight = 1.0 / len(STABLE_PLUS_FEATURES)
    for ax, (dataset, direction_id) in zip(axes, targets.items()):
        sub = scores[(scores["dataset"].astype(str) == dataset) & (scores["direction_id"].astype(str) == direction_id)].sort_values("window_index")
        zw = z_wide[z_wide["dataset"].astype(str) == dataset].sort_values("window_index")
        merged = sub[["window_index", "center_cycle"]].merge(
            zw[["window_index", "rs_corrdist_base", "rx_corrdist_base"]], on="window_index", how="left"
        )
        ax.plot(merged["center_cycle"], weight * merged["rs_corrdist_base"], label="rs_corrdist_base contribution", color="#5a6fb0")
        ax.plot(merged["center_cycle"], weight * merged["rx_corrdist_base"], label="rx_corrdist_base contribution", color="#c76d2a")
        ax.set_title(f"{dataset}: corrdist contribution to M0_stable")
        ax.set_ylabel("Contribution")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Cycle")
    save_figure(fig, figures_dir / "fig_corrdist_contribution_timeseries")


def plot_channel_family_ablation(comparison: pd.DataFrame, figures_dir: Path) -> None:
    comp = comparison[comparison["feature_group"].isin(["rx_only", "ry_only", "rs_only", "rx_ry", "rx_ry_rs", "stable_plus"])].copy()
    order = ["rx_only", "ry_only", "rs_only", "rx_ry", "rx_ry_rs", "stable_plus"]
    forms = ["mean_z", "direction_mean_z", "weighted_direction"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    width = 0.24
    for pos, form in enumerate(forms):
        values = []
        for group in order:
            sub = comp[(comp["feature_group"] == group) & (comp["formulation"] == form)]
            values.append(float(sub["worst_AUROC"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (pos - 1) * width, values, width=width, label=FORMULATION_LABELS[form], color=MODEL_COLORS[form])
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("Worst-direction AUROC")
    ax.set_title("Channel/family ablation")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, figures_dir / "fig_channel_family_ablation")


def plot_feature_contribution_top(weight_table: pd.DataFrame, figures_dir: Path) -> None:
    sub = weight_table[weight_table["model_name"].isin(["M0_stable", "M2_stable"])].copy()
    sub["abs_gap"] = sub["contribution_gap"].abs()
    top = sub.sort_values("abs_gap", ascending=False).head(18)
    labels = top["direction_id"].astype(str) + "\n" + top["model_name"].astype(str) + "\n" + top["feature_name"].astype(str)
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    colors = np.where(top["contribution_gap"].to_numpy(dtype=float) >= 0, "#46a67a", "#b64b5a")
    ax.bar(np.arange(len(top)), top["contribution_gap"], color=colors)
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Stage5 - Stage1 contribution")
    ax.set_title("Top feature contribution gaps")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "fig_feature_contribution_top")


def plot_state_map_selected(
    scores: pd.DataFrame,
    state_v2: pd.DataFrame,
    bd_thresholds: pd.DataFrame,
    eval_df: pd.DataFrame,
    selected_model: str,
    figures_dir: Path,
    config: FairAblationConfig,
) -> None:
    targets = {"Exp1": "Exp2_to_Exp1", "Exp2": "Exp1_to_Exp2"}
    bd_default = bd_thresholds[bd_thresholds["bd_metric"].astype(str) == config.bd_default_metric]
    bd_lookup = {str(row.dataset): float(row.BD_major_threshold) for row in bd_default.itertuples(index=False)}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, (dataset, direction_id) in zip(axes, targets.items()):
        sub = scores[(scores["dataset"].astype(str) == dataset) & (scores["direction_id"].astype(str) == direction_id)].sort_values("window_index")
        sub = sub[["dataset", "window_index", "stage", selected_model]].merge(
            state_v2[["dataset", "window_index", config.bd_default_metric]], on=["dataset", "window_index"], how="left"
        )
        th = float(
            eval_df[
                (eval_df["model_name"] == selected_model)
                & (eval_df["direction_id"] == direction_id)
            ]["source_threshold_P95"].iloc[0]
        )
        awr_high = sub[selected_model].to_numpy(dtype=float) >= th
        bd_high = sub[config.bd_default_metric].to_numpy(dtype=float) >= bd_lookup.get(dataset, np.inf)
        regions = np.where(
            awr_high & bd_high,
            "high_AWR_high_BD",
            np.where(awr_high, "high_AWR_low_BD", np.where(bd_high, "low_AWR_high_BD", "low_AWR_low_BD")),
        )
        colors = pd.Series(regions).map(REGION_COLORS).to_numpy()
        ax.scatter(sub[config.bd_default_metric], sub[selected_model], c=colors, s=8, alpha=0.6, linewidths=0)
        ax.set_title(f"{dataset}: selected fair AWR state map")
        ax.set_xlabel(config.bd_default_metric)
        ax.set_ylabel(selected_model)
        ax.grid(alpha=0.25)
    save_figure(fig, figures_dir / "fig_AWR_BD_state_map_fair_selected")


def plot_fair_summary(comparison: pd.DataFrame, figures_dir: Path) -> None:
    picked = [
        "M0_stable",
        "M1_stable",
        "M2_stable",
        "M0_stable_no_corrdist",
        "rx_only_mean_z",
        "ry_only_mean_z",
        "rs_only_mean_z",
    ]
    comp = comparison.set_index("model_name")
    labels = [m for m in picked if m in comp.index]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, metric, title in zip(axes, ["worst_AUROC", "worst_AUPRC"], ["Worst AUROC", "Worst AUPRC"]):
        values = [float(comp.loc[m, metric]) for m in labels]
        ax.bar(np.arange(len(labels)), values, color="#5a84b8")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "fig_fair_ablation_summary")


def generate_figures(
    scores: pd.DataFrame,
    z_wide: pd.DataFrame,
    eval_df: pd.DataFrame,
    comparison: pd.DataFrame,
    weight_table: pd.DataFrame,
    state_v2: pd.DataFrame,
    bd_thresholds: pd.DataFrame,
    boundaries: pd.DataFrame,
    decision: Dict[str, object],
    dirs: Dict[str, Path],
    config: FairAblationConfig,
) -> None:
    selected_model = str(decision["best_stable_plus_model"])
    plot_stable_timeseries(scores, boundaries, dirs["figures"])
    plot_stable_model_comparison(eval_df, dirs["figures"])
    plot_corrdist_ablation(comparison, dirs["figures"])
    plot_corrdist_contribution(scores, z_wide, dirs["figures"])
    plot_channel_family_ablation(comparison, dirs["figures"])
    plot_feature_contribution_top(weight_table, dirs["figures"])
    plot_state_map_selected(scores, state_v2, bd_thresholds, eval_df, selected_model, dirs["figures"], config)
    plot_fair_summary(comparison, dirs["figures"])


def write_report(
    path: Path,
    decision: Dict[str, object],
    stable_summary: pd.DataFrame,
    corrdist_summary: pd.DataFrame,
    channel_summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best_stable = decision["best_stable_plus_model"]
    best_channel = decision["best_channel_family_model"]
    corr_level = decision["corrdist_dependency"]
    recommended = decision["recommended_AWR_structure"]
    text = f"""# Fair AWR ablation interpretation

## Purpose

This run performs a fair ablation of the AWR score. It does not rebuild Stage1-Stage5 classification and does not change the research target. Stage5 is used only as a late-state proxy label for target-side evaluation.

## Main answers

1. Best stable_plus model: **{best_stable}**.
2. M0_stable remains the best stable_plus model: **{decision["M0_stable_is_best_stable_plus"]}**.
3. M1_stable is close to M0_stable under the worst-direction guardrail: **{decision["M1_stable_close_to_M0"]}**.
4. M2_stable shows incremental gain over M1_stable: **{decision["M2_stable_has_incremental_gain"]}**.
5. Corrdist dependency: **{corr_level}**.
6. Best channel/family model: **{best_channel}**.
7. Recommended AWR structure: {recommended}

## Stable Plus Ablation

{dataframe_to_markdown(stable_summary)}

## Corrdist Ablation

{dataframe_to_markdown(corrdist_summary)}

## Channel / Feature Family Ablation

{dataframe_to_markdown(channel_summary)}

## Overall Model Comparison

{dataframe_to_markdown(comparison)}

## Interpretation Boundary

AUROC/AUPRC describe target-side ranking ability with Stage5 as a late-state proxy label. The high_AWR_high_BD occupancy describes how a source-only high-AWR state threshold transfers into the target dataset when combined with BD v2. These two outputs should be interpreted separately.

BD is reused from v2 as the AWR-independent baseline deviation layer. The ablation results are signal-layer evidence only; FEM/contact morphology/debris evidence remains the next physical closed-loop validation layer.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    config = FairAblationConfig()
    dirs = setup_dirs(config)
    setup_logging(dirs)
    logging.info("Starting fair AWR ablation")

    z_wide, meta = load_z_wide(config)
    state_v2 = pd.read_csv(Path(config.v2_dir) / "window_state_scores_v2.csv")
    bd_thresholds = pd.read_csv(Path(config.v2_dir) / "bd_thresholds_v2.csv")
    boundaries_path = Path(config.v2_dir) / "stage_boundaries_v2.csv"
    boundaries = pd.read_csv(boundaries_path) if boundaries_path.exists() else pd.DataFrame()

    scores, eval_df, direction_table, weight_table, _ = compute_all_ablation_scores(
        z_wide, meta, state_v2, bd_thresholds, config
    )
    comparison = add_summary_stats(eval_df)
    decision = make_decision(eval_df, comparison, weight_table)

    stable_summary = eval_df[eval_df["model_name"].isin(["M0_stable", "M1_stable", "M2_stable"])].copy()
    corrdist_summary = eval_df[
        eval_df["model_name"].isin(
            [
                "M0_stable",
                "M1_stable",
                "M2_stable",
                "M0_stable_no_corrdist",
                "M1_stable_no_corrdist",
                "M2_stable_no_corrdist",
            ]
        )
    ].copy()
    channel_summary = eval_df[eval_df["feature_group"].isin(["rx_only", "ry_only", "rs_only", "rx_ry", "rx_ry_rs", "stable_plus"])].copy()

    stable_summary.to_csv(dirs["results"] / "stable_plus_ablation_summary.csv", index=False, encoding="utf-8-sig")
    corrdist_summary.to_csv(dirs["results"] / "corrdist_ablation_summary.csv", index=False, encoding="utf-8-sig")
    channel_summary.to_csv(dirs["results"] / "channel_family_ablation_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(dirs["results"] / "fair_ablation_model_comparison.csv", index=False, encoding="utf-8-sig")
    direction_table.to_csv(dirs["results"] / "fair_ablation_feature_direction_table.csv", index=False, encoding="utf-8-sig")
    weight_table.to_csv(dirs["results"] / "fair_ablation_feature_weight_table.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(dirs["results"] / "fair_ablation_window_scores.csv", index=False, encoding="utf-8-sig")
    (dirs["results"] / "fair_ablation_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (dirs["configs"] / "fair_ablation_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        dirs["reports"] / "fair_ablation_interpretation.md",
        decision,
        stable_summary,
        corrdist_summary,
        channel_summary,
        comparison,
    )
    generate_figures(
        scores,
        z_wide,
        eval_df,
        comparison,
        weight_table,
        state_v2,
        bd_thresholds,
        boundaries,
        decision,
        dirs,
        config,
    )

    print("Fair AWR ablation complete.")
    print(f"Best stable_plus model: {decision['best_stable_plus_model']}")
    print(f"Best channel/family model: {decision['best_channel_family_model']}")
    print(f"Corrdist dependency: {decision['corrdist_dependency']}")
    print(f"Recommended AWR structure: {decision['recommended_AWR_structure']}")
    print(f"Output directory: {config.output_dir}/")


if __name__ == "__main__":
    main()
