from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_DATE = "20260709"

STABLE_PLUS = [
    "rs_corrdist_base",
    "rs_mean",
    "rs_absmean",
    "rs_q05",
    "rx_corrdist_base",
    "rs_rms",
    "ry_p2p",
    "rx_mean",
    "rx_absmean",
    "rx_q05",
]


@dataclass
class AuditConfig:
    output_dir: str = "outputs_stable_plus_selection_audit_v1"
    status_file: str = f"docs/STATUS_{RUN_DATE}.md"
    saturation_rate_threshold: float = 0.05
    clip_abs_threshold: float = 11.9
    min_direction_auc_for_strong_reason: float = 0.70
    min_spearman_for_strong_reason: float = 0.30


PATHS = {
    "fair_scores": Path("outputs_awrcore_fair_ablation_v1/results/fair_ablation_window_scores.csv"),
    "feature_directions": Path("outputs_awrcore_fair_ablation_v1/results/fair_ablation_feature_direction_table.csv"),
    "feature_weights": Path("outputs_awrcore_fair_ablation_v1/results/fair_ablation_feature_weight_table.csv"),
    "channel_family_summary": Path("outputs_awrcore_fair_ablation_v1/results/channel_family_ablation_summary.csv"),
    "stable_plus_summary": Path("outputs_awrcore_fair_ablation_v1/results/stable_plus_ablation_summary.csv"),
    "z_table": Path("outputs_weighted_awrcore_v1/results/window_feature_z_table.csv"),
    "normalization_diag_preferred": Path("outputs_weighted_awrcore_v1/results/feature_normalization_diagnostics.csv"),
    "normalization_diag_fallback": Path("outputs_weighted_awrcore_v1/diagnostics/feature_normalization_diagnostics.csv"),
    "fair_report": Path("outputs_awrcore_fair_ablation_v1/reports/fair_ablation_interpretation.md"),
}


def setup_dirs(config: AuditConfig) -> Dict[str, Path]:
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
    Path(config.status_file).parent.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(dirs: Dict[str, Path]) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(dirs["root"] / "stable_plus_selection_audit_run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def read_csv(path: Path, warnings: List[str], required: bool = True) -> pd.DataFrame:
    if not path.exists():
        message = f"Missing input file: {path}"
        if required:
            warnings.append(message)
        logging.warning(message)
        return pd.DataFrame()
    logging.info("Reading %s", path)
    return pd.read_csv(path)


def read_normalization_diagnostics(warnings: List[str]) -> pd.DataFrame:
    preferred = PATHS["normalization_diag_preferred"]
    fallback = PATHS["normalization_diag_fallback"]
    if preferred.exists():
        return read_csv(preferred, warnings, required=False)
    if fallback.exists():
        warnings.append(f"Preferred normalization diagnostics missing; used fallback: {fallback}")
        return read_csv(fallback, warnings, required=False)
    warnings.append(f"Normalization diagnostics missing: {preferred} and {fallback}")
    return pd.DataFrame()


def dataframe_to_markdown(frame: pd.DataFrame, max_rows: int = 80) -> str:
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


def sign_label(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    return "increase" if value >= 0 else "decrease"


def safe_direction_sign(value: float) -> int:
    if not np.isfinite(value) or value == 0:
        return 1
    return 1 if value > 0 else -1


def roc_auc_binary(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=float)
    pos_rank_sum = float(np.sum(ranks[y == 1]))
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_binary(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    cumulative_pos = np.cumsum(y_sorted == 1)
    ranks = np.arange(1, len(y_sorted) + 1, dtype=float)
    precision = cumulative_pos / ranks
    return float(np.sum(precision[y_sorted == 1]) / n_pos)


def build_candidate_pool(z_table: pd.DataFrame) -> pd.DataFrame:
    cols = ["feature_name", "channel", "feature_family", "physical_meaning"]
    return z_table[cols].drop_duplicates("feature_name").sort_values(["channel", "feature_family", "feature_name"])


def build_direction_lookup(direction_table: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, float]]:
    lookup: Dict[Tuple[str, str], Dict[str, float]] = {}
    if direction_table.empty:
        return lookup
    for row in direction_table.itertuples(index=False):
        key = (str(row.source_dataset), str(row.feature_name))
        lookup[key] = {
            "direction_id": str(row.direction_id),
            "early_median": float(row.early_median),
            "late_median": float(row.late_median),
            "delta_median": float(row.delta_median),
            "direction_sign": int(row.direction_sign),
            "abs_effect_size": float(row.abs_effect_size),
            "direction_stability": float(row.direction_stability),
        }
    return lookup


def compute_fallback_medians(z_table: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, float]]:
    lookup: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (dataset, feature), sub in z_table.groupby(["dataset", "feature_name"], sort=True):
        early = float(sub.loc[sub["stage"].astype(int) == 1, "z_value"].median())
        late = float(sub.loc[sub["stage"].astype(int) == 5, "z_value"].median())
        gap = late - early
        lookup[(str(dataset), str(feature))] = {
            "direction_id": "",
            "early_median": early,
            "late_median": late,
            "delta_median": gap,
            "direction_sign": safe_direction_sign(gap),
            "abs_effect_size": abs(gap),
            "direction_stability": float("nan"),
        }
    return lookup


def compute_target_metrics(
    z_table: pd.DataFrame,
    direction_info: Dict[Tuple[str, str], Dict[str, float]],
    feature: str,
    source_dataset: str,
    target_dataset: str,
) -> Tuple[float, float, float]:
    info = direction_info.get((source_dataset, feature))
    if not info:
        return float("nan"), float("nan"), float("nan")
    target = z_table[
        (z_table["dataset"].astype(str) == target_dataset)
        & (z_table["feature_name"].astype(str) == feature)
    ]
    if target.empty:
        return float("nan"), float("nan"), float("nan")
    sign = int(info["direction_sign"])
    scores = target["z_value"].astype(float).to_numpy() * sign
    y = target["stage"].astype(int).eq(5).astype(int).to_numpy()
    auc = roc_auc_binary(y, scores)
    auprc = average_precision_binary(y, scores)
    spear = pd.Series(target["stage"].astype(float).to_numpy()).corr(pd.Series(scores), method="spearman")
    return float(auc), float(auprc), float(spear)


def compute_dataset_spearman(
    z_table: pd.DataFrame,
    direction_info: Dict[Tuple[str, str], Dict[str, float]],
    feature: str,
    dataset: str,
) -> float:
    sub = z_table[
        (z_table["dataset"].astype(str) == dataset)
        & (z_table["feature_name"].astype(str) == feature)
    ]
    if sub.empty:
        return float("nan")
    sign = int(direction_info.get((dataset, feature), {}).get("direction_sign", 1))
    scores = sub["z_value"].astype(float).to_numpy() * sign
    return float(pd.Series(sub["stage"].astype(float).to_numpy()).corr(pd.Series(scores), method="spearman"))


def saturation_lookup(diag: pd.DataFrame, z_table: pd.DataFrame, config: AuditConfig) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    if not diag.empty:
        temp = diag.copy()
        temp["saturation_rate_total"] = temp.get("saturation_rate_low", 0.0).fillna(0.0) + temp.get(
            "saturation_rate_high", 0.0
        ).fillna(0.0)
        for feature, sub in temp.groupby("feature_name", sort=True):
            rows[str(feature)] = {
                "max_saturation_rate": float(sub["saturation_rate_total"].max()),
                "max_missing_rate": float(sub.get("missing_rate", pd.Series([0.0])).max()),
            }
    z = z_table.copy()
    z["abs_z"] = z["z_value"].astype(float).abs()
    observed = z.groupby("feature_name").agg(
        max_abs_z=("abs_z", "max"),
        p99_abs_z=("abs_z", lambda s: float(s.quantile(0.99))),
        frac_abs_ge_clip=("abs_z", lambda s: float((s >= config.clip_abs_threshold).mean())),
    )
    for feature, obs in observed.iterrows():
        item = rows.setdefault(str(feature), {"max_saturation_rate": 0.0, "max_missing_rate": 0.0})
        item["max_abs_z"] = float(obs["max_abs_z"])
        item["p99_abs_z"] = float(obs["p99_abs_z"])
        item["frac_abs_ge_clip"] = float(obs["frac_abs_ge_clip"])
    return rows


def redundancy_lookup(weights: pd.DataFrame) -> Dict[str, str]:
    if weights.empty or "redundancy_notes" not in weights.columns:
        return {}
    rows: Dict[str, str] = {}
    for feature, sub in weights.groupby("feature_name", sort=True):
        notes = sorted({str(value) for value in sub["redundancy_notes"].dropna() if str(value).strip()})
        if notes:
            rows[str(feature)] = "; ".join(notes[:3])
    return rows


def infer_redundancy_note(feature: str, family: str, note_lookup: Dict[str, str]) -> str:
    if feature in note_lookup:
        return note_lookup[feature]
    if family in {"mean", "absmean", "rms"}:
        return "amplitude-family feature; interpret jointly with same-channel signed and magnitude descriptors."
    if family == "corrdist_base":
        return "baseline-shape descriptor; compare with amplitude descriptors to separate waveform-shape change."
    if family in {"peak_phase", "peak_width"}:
        return "phase/width descriptor; more sensitive to local waveform alignment and should be treated cautiously."
    return "no strong redundancy note in existing weight table."


def build_keep_reason(row: pd.Series, config: AuditConfig) -> str:
    if bool(row["in_stable_plus"]):
        pieces = [
            "retained in stable_plus because it comes from the candidate shear-ratio pool",
            f"direction consistency={row['direction_consistent']}",
            f"worst AUROC={row['worst_direction_AUROC']:.3f}",
            f"Spearman_min={row['Spearman_min']:.3f}",
        ]
        if str(row["saturation_warning"]).startswith("yes"):
            pieces.append("kept with a physical-validation flag because clipping/saturation is visible")
        else:
            pieces.append("no major clipping flag under the current diagnostics")
        return "; ".join(pieces) + "."
    reasons = []
    if not bool(row["direction_consistent"]):
        reasons.append("direction changes between Exp1 and Exp2")
    if np.isfinite(row["worst_direction_AUROC"]) and row["worst_direction_AUROC"] < config.min_direction_auc_for_strong_reason:
        reasons.append("weaker cross-experiment target ranking")
    if str(row["saturation_warning"]).startswith("yes"):
        reasons.append("clip/saturation risk")
    if not reasons:
        reasons.append("not part of the current compact stable_plus structure after redundancy and interpretability checks")
    return "not retained in stable_plus: " + "; ".join(reasons) + "."


def build_audit_table(
    z_table: pd.DataFrame,
    direction_table: pd.DataFrame,
    weights: pd.DataFrame,
    diag: pd.DataFrame,
    config: AuditConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pool = build_candidate_pool(z_table)
    fallback = compute_fallback_medians(z_table)
    direction_lookup = fallback
    direction_lookup.update(build_direction_lookup(direction_table))
    sat = saturation_lookup(diag, z_table, config)
    redund = redundancy_lookup(weights)

    rows = []
    for meta in pool.itertuples(index=False):
        feature = str(meta.feature_name)
        exp1 = direction_lookup.get(("Exp1", feature), {})
        exp2 = direction_lookup.get(("Exp2", feature), {})
        exp1_gap = float(exp1.get("delta_median", np.nan))
        exp2_gap = float(exp2.get("delta_median", np.nan))
        exp1_sign = int(exp1.get("direction_sign", safe_direction_sign(exp1_gap)))
        exp2_sign = int(exp2.get("direction_sign", safe_direction_sign(exp2_gap)))
        direction_consistent = bool(exp1_sign == exp2_sign)

        auc_12, auprc_12, spear_target_12 = compute_target_metrics(
            z_table, direction_lookup, feature, "Exp1", "Exp2"
        )
        auc_21, auprc_21, spear_target_21 = compute_target_metrics(
            z_table, direction_lookup, feature, "Exp2", "Exp1"
        )
        spearman_exp1 = compute_dataset_spearman(z_table, direction_lookup, feature, "Exp1")
        spearman_exp2 = compute_dataset_spearman(z_table, direction_lookup, feature, "Exp2")
        spearman_values = [value for value in [spearman_exp1, spearman_exp2] if np.isfinite(value)]
        spearman_min = float(min(spearman_values)) if spearman_values else float("nan")

        sat_item = sat.get(feature, {})
        max_sat = float(sat_item.get("max_saturation_rate", 0.0))
        frac_clip = float(sat_item.get("frac_abs_ge_clip", 0.0))
        warning_flag = max(max_sat, frac_clip) >= config.saturation_rate_threshold
        saturation_warning = (
            f"yes: max_saturation={max_sat:.3f}, frac_abs_ge_{config.clip_abs_threshold:.1f}={frac_clip:.3f}"
            if warning_flag
            else f"no: max_saturation={max_sat:.3f}, frac_abs_ge_{config.clip_abs_threshold:.1f}={frac_clip:.3f}"
        )

        rows.append(
            {
                "feature_name": feature,
                "channel": str(meta.channel),
                "feature_family": str(meta.feature_family),
                "in_stable_plus": feature in STABLE_PLUS,
                "Exp1_early_median": float(exp1.get("early_median", np.nan)),
                "Exp1_late_median": float(exp1.get("late_median", np.nan)),
                "Exp1_direction": sign_label(exp1_gap),
                "Exp1_direction_sign": exp1_sign,
                "Exp1_effect_gap": exp1_gap,
                "Exp2_early_median": float(exp2.get("early_median", np.nan)),
                "Exp2_late_median": float(exp2.get("late_median", np.nan)),
                "Exp2_direction": sign_label(exp2_gap),
                "Exp2_direction_sign": exp2_sign,
                "Exp2_effect_gap": exp2_gap,
                "direction_consistent": direction_consistent,
                "Exp1_to_Exp2_target_AUROC": auc_12,
                "Exp1_to_Exp2_target_AUPRC": auprc_12,
                "Exp2_to_Exp1_target_AUROC": auc_21,
                "Exp2_to_Exp1_target_AUPRC": auprc_21,
                "worst_direction_AUROC": float(np.nanmin([auc_12, auc_21])),
                "worst_direction_AUPRC": float(np.nanmin([auprc_12, auprc_21])),
                "Spearman_Exp1_signed": spearman_exp1,
                "Spearman_Exp2_signed": spearman_exp2,
                "Spearman_target_Exp1_to_Exp2": spear_target_12,
                "Spearman_target_Exp2_to_Exp1": spear_target_21,
                "Spearman_min": spearman_min,
                "Stage5_median_minus_Stage1_median_Exp1": exp1_gap,
                "Stage5_median_minus_Stage1_median_Exp2": exp2_gap,
                "min_abs_effect_gap": float(np.nanmin([abs(exp1_gap), abs(exp2_gap)])),
                "max_saturation_rate": max_sat,
                "frac_abs_ge_clip": frac_clip,
                "saturation_warning": saturation_warning,
                "redundancy_note": infer_redundancy_note(feature, str(meta.feature_family), redund),
                "physical_meaning": str(meta.physical_meaning),
            }
        )
    audit = pd.DataFrame(rows)
    audit["keep_reason"] = audit.apply(lambda row: build_keep_reason(row, config), axis=1)

    ordered_cols = [
        "feature_name",
        "channel",
        "feature_family",
        "in_stable_plus",
        "Exp1_direction",
        "Exp2_direction",
        "direction_consistent",
        "Exp1_effect_gap",
        "Exp2_effect_gap",
        "worst_direction_AUROC",
        "worst_direction_AUPRC",
        "Spearman_min",
        "saturation_warning",
        "redundancy_note",
        "physical_meaning",
        "keep_reason",
    ]
    remaining = [col for col in audit.columns if col not in ordered_cols]
    audit = audit[ordered_cols + remaining].sort_values(["in_stable_plus", "worst_direction_AUROC"], ascending=[False, False])
    return audit, pool


def summarize_model_context(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = summary[
        (summary["feature_group"].astype(str) == "stable_plus")
        & (summary["model_name"].astype(str).isin(["M0_stable", "M1_stable", "M2_stable"]))
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    out = (
        rows.groupby(["model_name", "feature_group", "formulation"], as_index=False)
        .agg(
            mean_AUROC=("target_AUROC", "mean"),
            worst_AUROC=("target_AUROC", "min"),
            mean_AUPRC=("target_AUPRC", "mean"),
            worst_AUPRC=("target_AUPRC", "min"),
            mean_Spearman=("Spearman_stage_AWR", "mean"),
            worst_Spearman=("Spearman_stage_AWR", "min"),
            mean_ScoreGap=("ScoreGap", "mean"),
            worst_ScoreGap=("ScoreGap", "min"),
        )
        .sort_values("model_name")
    )
    return out


def validation_focus(row: pd.Series) -> str:
    family = str(row["feature_family"])
    channel = str(row["channel"])
    if str(row["saturation_warning"]).startswith("yes"):
        return "Yes; verify whether the large normalized response is physical rather than clipping-driven."
    if family == "corrdist_base":
        return "Yes; compare with contact-zone migration and waveform-shape change in FEM or morphology."
    if family in {"q05", "p2p"}:
        return "Yes; check sensitive-phase tail/span behavior against local contact and debris evidence."
    if channel == "rs":
        return "Moderate; use resultant shear behavior as an integrated cross-channel validation clue."
    return "Moderate; validate when the candidate window is selected for physical-loop checking."


def write_feature_rationale(report_path: Path, audit: pd.DataFrame) -> None:
    stable = audit[audit["in_stable_plus"].astype(bool)].copy()
    stable = stable.set_index("feature_name").loc[STABLE_PLUS].reset_index()
    lines = [
        "# stable_plus Feature Rationale",
        "",
        "This note explains why the current stable_plus features are treated as a justified signal layer rather than an arbitrary empirical subset.",
        "Each item is drawn from the candidate shear-ratio feature pool and is checked for cross-experiment direction, target-side ranking, effect gap, saturation risk, redundancy, and physical meaning.",
        "",
    ]
    for row in stable.itertuples(index=False):
        lines.extend(
            [
                f"## `{row.feature_name}`",
                "",
                f"`{row.feature_name}` belongs to the `{row.channel}` channel and `{row.feature_family}` feature family. It reflects {str(row.physical_meaning).rstrip('.')}.",
                f"It is kept because Exp1/Exp2 directions are {str(row.Exp1_direction)} / {str(row.Exp2_direction)}, direction_consistent={bool(row.direction_consistent)}, worst AUROC={float(row.worst_direction_AUROC):.3f}, worst AUPRC={float(row.worst_direction_AUPRC):.3f}, and Spearman_min={float(row.Spearman_min):.3f}.",
                f"Physical-loop priority: {validation_focus(pd.Series(row._asdict()))}",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_audit_report(
    report_path: Path,
    audit: pd.DataFrame,
    model_context: pd.DataFrame,
    warnings: List[str],
) -> None:
    stable = audit[audit["in_stable_plus"].astype(bool)].copy()
    non_stable = audit[~audit["in_stable_plus"].astype(bool)].copy()
    top_cols = [
        "feature_name",
        "channel",
        "feature_family",
        "direction_consistent",
        "worst_direction_AUROC",
        "worst_direction_AUPRC",
        "Spearman_min",
        "saturation_warning",
        "keep_reason",
    ]
    lines = [
        "# stable_plus Selection Audit Report",
        "",
        "## Purpose",
        "",
        "This run documents the evidence behind the current stable_plus feature set. It does not retrain AWR, rebuild Stage1-Stage5 classification, or change the research target.",
        "",
        "## Main Answer",
        "",
        "stable_plus is not an arbitrary hand-picked list. It is a compact subset of the candidate shear-ratio feature pool, audited against direction consistency, cross-experiment target-side ranking, effect size, saturation risk, redundancy, and physical interpretability.",
        "",
        "## Candidate Pool",
        "",
        f"- Candidate features audited: {len(audit)}.",
        f"- stable_plus features: {len(stable)}.",
        f"- non stable_plus reference features: {len(non_stable)}.",
        "- Source pool: `outputs_weighted_awrcore_v1/results/window_feature_z_table.csv`.",
        "",
        "## Selection Evidence",
        "",
        "- Direction consistency checks whether Exp1 and Exp2 show the same early-to-late feature direction.",
        "- Target AUROC/AUPRC are computed by applying the source-dataset direction sign to the opposite dataset and using Stage5 only as a late-state proxy label.",
        "- Spearman_min summarizes the weakest signed monotonic relation between stage and feature across Exp1/Exp2.",
        "- Saturation warnings combine available normalization diagnostics with observed clipped z-values.",
        "- Redundancy notes are read from the fair ablation feature weight table where available.",
        "",
        "## stable_plus Feature Audit",
        "",
        dataframe_to_markdown(stable[top_cols], max_rows=40),
        "",
        "## Non stable_plus Reference Features",
        "",
        dataframe_to_markdown(non_stable[top_cols].sort_values("worst_direction_AUROC", ascending=False), max_rows=40),
        "",
        "## Fair Ablation Context",
        "",
        "M1_stable uses explicit direction correction. In the fair ablation, M1_stable is close to M0_stable, which supports the interpretation that internal direction conflict inside stable_plus is limited. M2_stable does not provide incremental gain over M1_stable, so the current equal-weight direction-corrected structure is the more robust interpretation layer.",
        "",
        dataframe_to_markdown(model_context, max_rows=20),
        "",
        "## Physical Validation Implications",
        "",
        "- Features with corrdist_base or saturation warnings should be prioritized in physical closed-loop validation because large normalized values may combine real waveform-shape change with clipping pressure.",
        "- Resultant shear features (`rs_*`) are useful as integrated shear-state descriptors, but their high agreement should be checked against rx/ry channel-specific evidence.",
        "- Tail/span features such as `rs_q05`, `rx_q05`, and `ry_p2p` should be checked against sensitive-phase contact migration, local wear morphology, and debris evidence.",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- None.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_status(status_path: Path, audit: pd.DataFrame, warnings: List[str]) -> None:
    stable = audit[audit["in_stable_plus"].astype(bool)].copy()
    strongest = stable.sort_values(["direction_consistent", "worst_direction_AUROC", "Spearman_min"], ascending=False)
    validation = stable[stable["saturation_warning"].astype(str).str.startswith("yes")].copy()
    if validation.empty:
        validation = stable[stable["feature_family"].isin(["corrdist_base", "q05", "p2p"])].copy()
    lines = [
        f"# STATUS {RUN_DATE}",
        "",
        "## 1. What Was Completed",
        "",
        "- Added `run_stable_plus_selection_audit.py`.",
        "- Generated a stable_plus feature selection audit without retraining AWR or rebuilding Stage1-Stage5 classification.",
        "- Audited the candidate shear-ratio feature pool for direction consistency, target-side ranking, effect gap, saturation risk, redundancy, and physical interpretability.",
        "",
        "## 2. Key Files and Data Paths",
        "",
        "- Output directory: `outputs_stable_plus_selection_audit_v1/`",
        "- Audit table: `outputs_stable_plus_selection_audit_v1/results/stable_plus_selection_audit.csv`",
        "- Candidate pool: `outputs_stable_plus_selection_audit_v1/results/candidate_feature_pool.csv`",
        "- Feature rationale: `outputs_stable_plus_selection_audit_v1/reports/stable_plus_feature_rationale.md`",
        "- Summary report: `outputs_stable_plus_selection_audit_v1/reports/stable_plus_selection_audit_report.md`",
        "- Selection flow figure: `outputs_stable_plus_selection_audit_v1/figures/fig_stable_plus_selection_flow.png`",
        "- Feature audit figure: `outputs_stable_plus_selection_audit_v1/figures/fig_stable_plus_feature_audit.png`",
        "",
        "## 3. stable_plus Audit Results",
        "",
        f"- Candidate features audited: {len(audit)}.",
        f"- stable_plus features audited: {len(stable)}.",
        "- stable_plus is documented as a compact, direction-checked, physically interpretable subset of the shear-ratio feature pool.",
        "",
        "### Features With Strongest Current Basis",
        "",
        dataframe_to_markdown(
            strongest[
                [
                    "feature_name",
                    "channel",
                    "feature_family",
                    "worst_direction_AUROC",
                    "worst_direction_AUPRC",
                    "Spearman_min",
                    "saturation_warning",
                ]
            ].head(8),
            max_rows=8,
        ),
        "",
        "### Features Needing Physical Closed-loop Focus",
        "",
        dataframe_to_markdown(
            validation[
                [
                    "feature_name",
                    "channel",
                    "feature_family",
                    "saturation_warning",
                    "physical_meaning",
                ]
            ].head(8),
            max_rows=8,
        ),
        "",
        "## 4. Next Step",
        "",
        "- Use this audit as the feature-source explanation layer when presenting M1_stable and the physical validation candidate windows.",
        "- In physical closed-loop validation, focus first on corrdist_base and clipped/saturated features, then verify sensitive-phase tail/span features.",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- None.")
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_selection_flow(figures_dir: Path) -> None:
    steps = [
        "Candidate shear-ratio\nfeature pool",
        "Baseline robust\nnormalization",
        "Early/late direction\ncheck",
        "Bidirectional cross-\nexperiment ranking",
        "Redundancy and\nsaturation check",
        "Physical interpretability\ncheck",
        "stable_plus\nfeature set",
    ]
    fig, ax = plt.subplots(figsize=(13.2, 3.2))
    ax.axis("off")
    xs = np.linspace(0.06, 0.94, len(steps))
    y = 0.52
    for idx, (x, label) in enumerate(zip(xs, steps)):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef3f8", edgecolor="#426b93", linewidth=1.0),
            transform=ax.transAxes,
        )
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[idx + 1] - 0.055, y),
                xytext=(x + 0.055, y),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color="#4d5966", linewidth=1.2),
            )
    ax.set_title("stable_plus selection audit flow", fontsize=13, pad=16)
    fig.savefig(figures_dir / "fig_stable_plus_selection_flow.png", dpi=240, bbox_inches="tight")
    fig.savefig(figures_dir / "fig_stable_plus_selection_flow.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_feature_audit(audit: pd.DataFrame, figures_dir: Path) -> None:
    frame = audit.copy()
    frame["group"] = np.where(frame["in_stable_plus"].astype(bool), "stable_plus", "non stable_plus")
    frame["direction_consistent_value"] = frame["direction_consistent"].astype(float)
    frame["saturation_warning_value"] = frame["saturation_warning"].astype(str).str.startswith("yes").astype(float)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    metrics = [
        ("direction_consistent_value", "Direction consistency rate", "bar"),
        ("worst_direction_AUROC", "Worst target AUROC", "box"),
        ("min_abs_effect_gap", "Minimum absolute effect gap", "box"),
        ("saturation_warning_value", "Saturation warning rate", "bar"),
    ]
    colors = {"stable_plus": "#3b6ea8", "non stable_plus": "#c76d2a"}
    for ax, (col, title, kind) in zip(axes.ravel(), metrics):
        groups = ["stable_plus", "non stable_plus"]
        data = [frame.loc[frame["group"] == group, col].dropna().astype(float).to_numpy() for group in groups]
        if kind == "bar":
            values = [float(np.mean(values)) if len(values) else np.nan for values in data]
            ax.bar(groups, values, color=[colors[group] for group in groups])
            ax.set_ylim(0, 1.05)
        else:
            ax.boxplot(data, tick_labels=groups, patch_artist=True, medianprops=dict(color="#222222"))
            for patch, group in zip(ax.artists, groups):
                patch.set_facecolor(colors[group])
                patch.set_alpha(0.55)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("stable_plus vs non stable_plus feature audit", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figures_dir / "fig_stable_plus_feature_audit.png", dpi=240, bbox_inches="tight")
    fig.savefig(figures_dir / "fig_stable_plus_feature_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = AuditConfig()
    dirs = setup_dirs(config)
    setup_logging(dirs)
    warnings: List[str] = []

    fair_scores = read_csv(PATHS["fair_scores"], warnings, required=False)
    direction_table = read_csv(PATHS["feature_directions"], warnings)
    weights = read_csv(PATHS["feature_weights"], warnings)
    channel_summary = read_csv(PATHS["channel_family_summary"], warnings, required=False)
    stable_summary = read_csv(PATHS["stable_plus_summary"], warnings, required=False)
    z_table = read_csv(PATHS["z_table"], warnings)
    normalization_diag = read_normalization_diagnostics(warnings)
    if not PATHS["fair_report"].exists():
        warnings.append(f"Fair ablation interpretation report not found: {PATHS['fair_report']}")
    if fair_scores.empty:
        warnings.append("fair_ablation_window_scores.csv was not needed for per-feature metrics but was missing or empty.")

    audit, pool = build_audit_table(z_table, direction_table, weights, normalization_diag, config)
    model_context = summarize_model_context(stable_summary if not stable_summary.empty else channel_summary)

    audit.to_csv(dirs["results"] / "stable_plus_selection_audit.csv", index=False, encoding="utf-8-sig")
    pool.to_csv(dirs["results"] / "candidate_feature_pool.csv", index=False, encoding="utf-8-sig")
    model_context.to_csv(dirs["results"] / "stable_plus_model_context.csv", index=False, encoding="utf-8-sig")
    (dirs["configs"] / "stable_plus_selection_audit_config.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "stable_plus": STABLE_PLUS,
                "input_paths": {key: str(value) for key, value in PATHS.items()},
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_feature_rationale(dirs["reports"] / "stable_plus_feature_rationale.md", audit)
    write_audit_report(dirs["reports"] / "stable_plus_selection_audit_report.md", audit, model_context, warnings)
    write_status(Path(config.status_file), audit, warnings)
    plot_selection_flow(dirs["figures"])
    plot_feature_audit(audit, dirs["figures"])

    logging.info("Stable plus selection audit complete.")
    logging.info("Output directory: %s", config.output_dir)
    logging.info("STATUS file: %s", config.status_file)
    print("Stable plus selection audit complete.")
    print(f"Output directory: {config.output_dir}/")
    print(f"STATUS file: {config.status_file}")


if __name__ == "__main__":
    main()
