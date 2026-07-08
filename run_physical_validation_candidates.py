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
class CandidateConfig:
    output_dir: str = "outputs_physical_validation_candidates_v1"
    main_awr_model: str = "M1_stable"
    fallback_awr_model: str = "M0_stable"
    min_candidates_per_type: int = 3
    max_candidates_per_type: int = 5
    min_center_cycle_gap: float = 500.0
    awr_high_percentile: float = 90.0
    awr_low_percentile: float = 70.0
    bd_high_percentile: float = 90.0
    bd_low_percentile: float = 70.0
    status_file: str = "docs/STATUS_20260708.md"


PATHS = {
    "fair_scores": Path("outputs_awrcore_fair_ablation_v1/results/fair_ablation_window_scores.csv"),
    "fair_weights": Path("outputs_awrcore_fair_ablation_v1/results/fair_ablation_feature_weight_table.csv"),
    "fair_directions": Path("outputs_awrcore_fair_ablation_v1/results/fair_ablation_feature_direction_table.csv"),
    "fair_decision": Path("outputs_awrcore_fair_ablation_v1/results/fair_ablation_decision.json"),
    "state_weighted": Path("outputs_weighted_awrcore_v1/results/window_state_scores_weighted.csv"),
    "awrs_weighted": Path("outputs_weighted_awrcore_v1/results/window_awrs_M0_M1_M2_M3.csv"),
    "tes_events": Path("outputs_weighted_awrcore_v1/results/tes_events_weighted.csv"),
    "tes_eval": Path("outputs_weighted_awrcore_v1/results/tes_event_eval_summary_weighted.csv"),
    "state_v2": Path("outputs_aux_state_metrics_v2/window_state_scores_v2.csv"),
    "bd_thresholds": Path("outputs_aux_state_metrics_v2/bd_thresholds_v2.csv"),
    "boundaries": Path("outputs_aux_state_metrics_v2/stage_boundaries_v2.csv"),
    "z_table": Path("outputs_weighted_awrcore_v1/results/window_feature_z_table.csv"),
}


CYCLE_MAPPING_SEGMENTS = [
    {
        "dataset": "Exp1",
        "stage": 1,
        "effective_start": 1.0,
        "effective_end": 7575.0,
        "actual_start": 1.0,
        "actual_end": 8000.0,
        "note": "Stage 1 piecewise mapping",
    },
    {
        "dataset": "Exp1",
        "stage": 2,
        "effective_start": 7575.0,
        "effective_end": 21125.0,
        "actual_start": 8000.0,
        "actual_end": 24000.0,
        "note": "Stage 2 piecewise mapping",
    },
    {
        "dataset": "Exp1",
        "stage": 3,
        "effective_start": 21125.0,
        "effective_end": 27840.0,
        "actual_start": 24000.0,
        "actual_end": 32000.0,
        "note": "Stage 3 piecewise mapping",
    },
    {
        "dataset": "Exp1",
        "stage": 4,
        "effective_start": 27840.0,
        "effective_end": 34600.0,
        "actual_start": 32000.0,
        "actual_end": 40000.0,
        "note": "Stage 4 piecewise mapping",
    },
    {
        "dataset": "Exp1",
        "stage": 5,
        "effective_start": 34600.0,
        "effective_end": 45590.0,
        "actual_start": 40000.0,
        "actual_end": 53000.0,
        "note": "Stage 5 piecewise mapping",
    },
    {
        "dataset": "Exp2",
        "stage": 1,
        "effective_start": 1.0,
        "effective_end": 3005.0,
        "actual_start": 501.0,
        "actual_end": 5500.0,
        "note": "Stage 1 piecewise mapping; actual cycles 1-500 are NaN",
    },
    {
        "dataset": "Exp2",
        "stage": 2,
        "effective_start": 3005.0,
        "effective_end": 6005.0,
        "actual_start": 5500.0,
        "actual_end": 10500.0,
        "note": "Stage 2 piecewise mapping",
    },
    {
        "dataset": "Exp2",
        "stage": 3,
        "effective_start": 6005.0,
        "effective_end": 8705.0,
        "actual_start": 10500.0,
        "actual_end": 15000.0,
        "note": "Stage 3 piecewise mapping",
    },
    {
        "dataset": "Exp2",
        "stage": 4,
        "effective_start": 8705.0,
        "effective_end": 11705.0,
        "actual_start": 15000.0,
        "actual_end": 20000.0,
        "note": "Stage 4 piecewise mapping",
    },
    {
        "dataset": "Exp2",
        "stage": 5,
        "effective_start": 11705.0,
        "effective_end": 14100.0,
        "actual_start": 20000.0,
        "actual_end": 24000.0,
        "note": "Stage 5 piecewise mapping",
    },
]


def setup_dirs(config: CandidateConfig) -> Dict[str, Path]:
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
            logging.FileHandler(dirs["root"] / "physical_validation_candidates_run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def load_csv(name: str, warnings: List[str]) -> pd.DataFrame:
    path = PATHS[name]
    if not path.exists():
        message = f"Missing input file: {path}"
        warnings.append(message)
        logging.warning(message)
        return pd.DataFrame()
    logging.info("Reading %s: %s", name, path)
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def finite_quantile(values: Iterable[float], percentile: float, default: float = np.nan) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.nanpercentile(arr, percentile))


def percentile_rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True, method="average") * 100.0


def target_direction_for_dataset(dataset: str) -> str:
    return {"Exp1": "Exp2_to_Exp1", "Exp2": "Exp1_to_Exp2"}.get(dataset, "")


def cycle_mapping_table() -> pd.DataFrame:
    rows = []
    for item in CYCLE_MAPPING_SEGMENTS:
        row = dict(item)
        row["slope_actual_per_effective"] = (
            (row["actual_end"] - row["actual_start"])
            / (row["effective_end"] - row["effective_start"])
        )
        row["formula"] = (
            "actual_start + (effective - effective_start) * "
            "(actual_end - actual_start) / (effective_end - effective_start)"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def map_effective_to_actual(dataset: str, effective_cycle: float) -> float:
    if not np.isfinite(effective_cycle):
        return float("nan")
    ds_segments = [item for item in CYCLE_MAPPING_SEGMENTS if item["dataset"] == str(dataset)]
    if not ds_segments:
        return float(effective_cycle)
    segment = None
    for item in ds_segments:
        if float(item["effective_start"]) <= float(effective_cycle) <= float(item["effective_end"]):
            segment = item
            break
    if segment is None:
        if effective_cycle < min(item["effective_start"] for item in ds_segments):
            segment = ds_segments[0]
        else:
            segment = ds_segments[-1]
    eff0 = float(segment["effective_start"])
    eff1 = float(segment["effective_end"])
    act0 = float(segment["actual_start"])
    act1 = float(segment["actual_end"])
    return float(act0 + (float(effective_cycle) - eff0) * (act1 - act0) / (eff1 - eff0))


def add_cycle_mapping_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("start_cycle", "end_cycle", "center_cycle"):
        if col in out.columns:
            out[f"{col}_effective"] = out[col].astype(float)
            out[f"{col}_actual"] = [
                map_effective_to_actual(dataset, value)
                for dataset, value in zip(out["dataset"].astype(str), out[col].astype(float))
            ]
    return out


def add_boundary_actual_cycles(boundaries: pd.DataFrame) -> pd.DataFrame:
    if boundaries.empty or "boundary_cycle" not in boundaries.columns:
        return boundaries.copy()
    out = boundaries.copy()
    out["boundary_cycle_effective"] = out["boundary_cycle"].astype(float)
    out["boundary_cycle_actual"] = [
        map_effective_to_actual(dataset, value)
        for dataset, value in zip(out["dataset"].astype(str), out["boundary_cycle"].astype(float))
    ]
    return out


def add_event_actual_cycles(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "peak_cycle" not in events.columns:
        return events.copy()
    out = events.copy()
    out["peak_cycle_effective"] = out["peak_cycle"].astype(float)
    out["peak_cycle_actual"] = [
        map_effective_to_actual(dataset, value)
        for dataset, value in zip(out["dataset"].astype(str), out["peak_cycle"].astype(float))
    ]
    return out


def prepare_main_frame(
    fair_scores: pd.DataFrame,
    state_weighted: pd.DataFrame,
    state_v2: pd.DataFrame,
    bd_thresholds: pd.DataFrame,
    tes_events: pd.DataFrame,
    config: CandidateConfig,
    warnings: List[str],
) -> Tuple[pd.DataFrame, str]:
    if fair_scores.empty:
        warnings.append("fair_ablation_window_scores.csv is missing; no candidate frame can be built.")
        return pd.DataFrame(), config.main_awr_model

    main_model = config.main_awr_model
    if main_model not in fair_scores.columns:
        main_model = config.fallback_awr_model
        warnings.append(f"{config.main_awr_model} not found; fallback to {main_model}.")
    if main_model not in fair_scores.columns:
        warnings.append(f"Neither {config.main_awr_model} nor {config.fallback_awr_model} found.")
        return pd.DataFrame(), main_model

    frames = []
    for dataset in sorted(fair_scores["dataset"].dropna().astype(str).unique()):
        direction_id = target_direction_for_dataset(dataset)
        sub = fair_scores[
            (fair_scores["dataset"].astype(str) == dataset)
            & (fair_scores["direction_id"].astype(str) == direction_id)
        ].copy()
        if sub.empty:
            sub = fair_scores[fair_scores["dataset"].astype(str) == dataset].copy()
            if not sub.empty:
                direction_id = str(sub["direction_id"].iloc[0])
                warnings.append(f"No target-direction rows for {dataset}; using {direction_id}.")
        frames.append(sub)
    main = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if main.empty:
        warnings.append("No fair ablation rows available after target-direction filtering.")
        return main, main_model

    main["AWR_model"] = main_model
    main["AWR"] = main[main_model].astype(float)
    main = add_cycle_mapping_columns(main)

    state_cols = [
        "dataset",
        "direction_id",
        "window_index",
        "BDx_v2",
        "BDy_v2",
        "BDshape_v2",
        "BDall_xy_v2",
        "RS_trend20",
        "RS_trend50",
        "RS_trend100",
        "TES",
        "TES_smooth",
        "is_TES_event",
        "TES_event_neighborhood",
        "TES_high_confidence_neighborhood",
        "state_region",
    ]
    if not state_weighted.empty:
        available = [col for col in state_cols if col in state_weighted.columns]
        main = main.merge(state_weighted[available], on=["dataset", "direction_id", "window_index"], how="left")
    else:
        warnings.append("Weighted state score table missing; BD/RS/TES fields will use v2 fallback where possible.")

    if "BDall_xy_v2" not in main.columns or main["BDall_xy_v2"].isna().all():
        if not state_v2.empty and "BDall_xy_v2" in state_v2.columns:
            fallback_cols = [
                col
                for col in [
                    "dataset",
                    "window_index",
                    "BDx_v2",
                    "BDy_v2",
                    "BDshape_v2",
                    "BDall_xy_v2",
                    "RS_trend50_v2",
                    "RS_trend100_v2",
                    "TES_smooth_v2",
                ]
                if col in state_v2.columns
            ]
            main = main.merge(state_v2[fallback_cols], on=["dataset", "window_index"], how="left", suffixes=("", "_v2fallback"))
            for left, right in [
                ("BDx_v2", "BDx_v2_v2fallback"),
                ("BDy_v2", "BDy_v2_v2fallback"),
                ("BDshape_v2", "BDshape_v2_v2fallback"),
                ("BDall_xy_v2", "BDall_xy_v2_v2fallback"),
                ("RS_trend50", "RS_trend50_v2"),
                ("RS_trend100", "RS_trend100_v2"),
                ("TES_smooth", "TES_smooth_v2"),
            ]:
                if right in main.columns:
                    if left not in main.columns:
                        main[left] = main[right]
                    else:
                        main[left] = main[left].fillna(main[right])
        else:
            warnings.append("No BD v2 fallback available.")

    for col in ["BDall_xy_v2", "RS_trend50", "RS_trend100", "TES_smooth", "TES", "is_TES_event"]:
        if col not in main.columns:
            main[col] = np.nan if col != "is_TES_event" else 0
    main["TES_for_selection"] = main["TES_smooth"].fillna(main["TES"])
    main["AWR_delta10"] = 0.0
    for dataset, idx in main.groupby("dataset", sort=True).groups.items():
        ordered = main.loc[list(idx)].sort_values("window_index")
        delta = ordered["AWR"].diff(10).fillna(0.0)
        main.loc[ordered.index, "AWR_delta10"] = delta
        main.loc[ordered.index, "AWR_percentile_within_dataset"] = percentile_rank(ordered["AWR"])
        main.loc[ordered.index, "BD_percentile_within_dataset"] = percentile_rank(ordered["BDall_xy_v2"])
        main.loc[ordered.index, "TES_percentile_within_dataset"] = percentile_rank(ordered["TES_for_selection"].fillna(0.0))

    bd_lookup = {}
    if not bd_thresholds.empty:
        default_bd = bd_thresholds[bd_thresholds["bd_metric"].astype(str) == "BDall_xy_v2"]
        bd_lookup = {str(row.dataset): float(row.BD_major_threshold) for row in default_bd.itertuples(index=False)}

    main["AWR_high"] = main["AWR_percentile_within_dataset"] >= config.awr_high_percentile
    main["AWR_low"] = main["AWR_percentile_within_dataset"] <= config.awr_low_percentile
    main["BD_high"] = False
    for dataset, idx in main.groupby("dataset", sort=True).groups.items():
        threshold = bd_lookup.get(str(dataset), np.nan)
        if np.isfinite(threshold):
            main.loc[list(idx), "BD_high"] = main.loc[list(idx), "BDall_xy_v2"].astype(float) >= threshold
        else:
            main.loc[list(idx), "BD_high"] = (
                main.loc[list(idx), "BD_percentile_within_dataset"].astype(float) >= config.bd_high_percentile
            )
    main["BD_low"] = main["BD_percentile_within_dataset"] <= config.bd_low_percentile

    main["state_region_main"] = np.where(
        main["AWR_high"] & main["BD_high"],
        "high_AWR_high_BD",
        np.where(
            main["AWR_high"],
            "high_AWR_low_BD",
            np.where(main["BD_high"], "low_AWR_high_BD", "low_AWR_low_BD"),
        ),
    )
    main["TES_event_peak"] = 0
    main["TES_high_confidence_peak"] = 0
    if not tes_events.empty:
        for ev in tes_events.itertuples(index=False):
            dataset = str(getattr(ev, "dataset"))
            window_index = int(getattr(ev, "peak_window_index"))
            direction_id = str(getattr(ev, "direction_id"))
            mask = (
                (main["dataset"].astype(str) == dataset)
                & (main["direction_id"].astype(str) == direction_id)
                & (main["window_index"].astype(int) == window_index)
            )
            main.loc[mask, "TES_event_peak"] = 1
            if bool(getattr(ev, "high_confidence_event")):
                main.loc[mask, "TES_high_confidence_peak"] = 1
    return main, main_model


def priority_and_check(candidate_type: str) -> Tuple[str, str]:
    mapping = {
        "high_AWR_high_BD": (
            "High",
            "Check surface damage, debris increase, and FEM high-contribution contact zones.",
        ),
        "high_AWR_low_BD": (
            "Medium-High",
            "Check local late-state morphology or short sensitive-phase shape change without global baseline shift.",
        ),
        "low_AWR_high_BD": (
            "Medium",
            "Check run-in, contact reorganization, lubrication disturbance, or non-late baseline deviation.",
        ),
        "AWR_rising": (
            "Medium-High",
            "Check whether rapid AWR increase aligns with contact migration or emerging morphology change.",
        ),
        "TES_high_confidence": (
            "High",
            "Check disassembly, lubrication change, measurement disturbance, or true force-signal transition.",
        ),
        "Exp1_late_stable_candidate": (
            "Medium",
            "Check stable late wear morphology and whether debris/contact state remains relatively steady.",
        ),
        "Exp2_late_severe_candidate": (
            "High",
            "Check severe late wear, increased debris, and high-risk contact morphology.",
        ),
    }
    return mapping.get(candidate_type, ("Medium", "Review with FEM, morphology, and debris evidence."))


def selection_text(candidate_type: str, relaxed: bool) -> str:
    base = {
        "high_AWR_high_BD": "High AWR percentile and high BDall_xy_v2.",
        "high_AWR_low_BD": "High AWR percentile with comparatively low BDall_xy_v2.",
        "low_AWR_high_BD": "High BDall_xy_v2 with comparatively low AWR percentile.",
        "AWR_rising": "Large local AWR increase / RS trend.",
        "TES_high_confidence": "TES high-confidence event or high TES neighborhood.",
        "Exp1_late_stable_candidate": "Exp1 Stage5 stable late-state candidate with lower AWR volatility.",
        "Exp2_late_severe_candidate": "Exp2 Stage5 high AWR and high BD candidate.",
    }.get(candidate_type, "Selected by candidate score.")
    return base + (" Gap or threshold relaxed because strict candidates were limited." if relaxed else "")


def pick_spaced(sub: pd.DataFrame, score_col: str, n_min: int, n_max: int, gap: float) -> Tuple[pd.DataFrame, bool]:
    if sub.empty:
        return sub.copy(), False
    ordered = sub.sort_values(score_col, ascending=False).copy()
    selected = []
    relaxed = False
    for row in ordered.itertuples():
        cycle = float(getattr(row, "center_cycle_actual", getattr(row, "center_cycle")))
        if all(abs(cycle - float(prev.get("center_cycle_actual", prev["center_cycle"]))) >= gap for prev in selected):
            selected.append(row._asdict())
        if len(selected) >= n_max:
            break
    if len(selected) < min(n_min, len(ordered)):
        relaxed = True
        seen = {int(item["window_index"]) for item in selected}
        for row in ordered.itertuples():
            if int(getattr(row, "window_index")) not in seen:
                selected.append(row._asdict())
                seen.add(int(getattr(row, "window_index")))
            if len(selected) >= min(n_max, len(ordered)):
                break
    return pd.DataFrame(selected), relaxed


def build_candidates(main: pd.DataFrame, config: CandidateConfig, warnings: List[str]) -> pd.DataFrame:
    rows = []
    category_specs = [
        ("high_AWR_high_BD", lambda d: d["AWR_high"] & d["BD_high"], "score_high_high"),
        ("high_AWR_low_BD", lambda d: d["AWR_high"] & d["BD_low"], "score_high_low"),
        ("low_AWR_high_BD", lambda d: d["AWR_low"] & d["BD_high"], "score_low_high"),
        ("AWR_rising", lambda d: d["RS_trend50"].fillna(0.0).gt(0) | d["AWR_delta10"].gt(0), "score_rising"),
        (
            "TES_high_confidence",
            lambda d: d["TES_high_confidence_peak"].eq(1) | d.get("TES_high_confidence_neighborhood", pd.Series(0, index=d.index)).eq(1),
            "score_tes",
        ),
    ]
    for dataset, ds in main.groupby("dataset", sort=True):
        ds = ds.copy()
        ds["score_high_high"] = ds["AWR_percentile_within_dataset"] + ds["BD_percentile_within_dataset"]
        ds["score_high_low"] = ds["AWR_percentile_within_dataset"] - 0.4 * ds["BD_percentile_within_dataset"]
        ds["score_low_high"] = ds["BD_percentile_within_dataset"] - 0.4 * ds["AWR_percentile_within_dataset"]
        ds["score_rising"] = (
            ds["RS_trend50"].fillna(0.0) * 1000.0
            + ds["RS_trend100"].fillna(0.0) * 500.0
            + ds["AWR_delta10"].fillna(0.0)
        )
        ds["score_tes"] = (
            ds["TES_high_confidence_peak"].fillna(0).astype(float) * 1000.0
            + ds.get("TES_high_confidence_neighborhood", pd.Series(0, index=ds.index)).fillna(0).astype(float) * 100.0
            + ds["TES_percentile_within_dataset"].fillna(0.0)
        )
        specs = list(category_specs)
        if str(dataset) == "Exp1":
            ds["score_exp1_stable"] = (
                100.0
                - ds["AWR_percentile_within_dataset"].fillna(100.0)
                - ds["RS_trend50"].fillna(0.0).abs() * 100.0
                - ds["TES_percentile_within_dataset"].fillna(0.0) * 0.1
            )
            specs.append(
                (
                    "Exp1_late_stable_candidate",
                    lambda d: d["stage"].astype(int).eq(5) & d["AWR_percentile_within_dataset"].le(75),
                    "score_exp1_stable",
                )
            )
        if str(dataset) == "Exp2":
            ds["score_exp2_severe"] = (
                ds["AWR_percentile_within_dataset"].fillna(0.0) + ds["BD_percentile_within_dataset"].fillna(0.0)
            )
            specs.append(
                (
                    "Exp2_late_severe_candidate",
                    lambda d: d["stage"].astype(int).eq(5) & d["AWR_percentile_within_dataset"].ge(75) & d["BD_high"],
                    "score_exp2_severe",
                )
            )
        for candidate_type, mask_fn, score_col in specs:
            mask = mask_fn(ds)
            strict = ds[mask].copy()
            relaxed = False
            if strict.empty:
                relaxed = True
                if candidate_type == "high_AWR_high_BD":
                    strict = ds.sort_values("score_high_high", ascending=False).head(30)
                elif candidate_type == "high_AWR_low_BD":
                    strict = ds[ds["AWR_high"]].sort_values("score_high_low", ascending=False).head(30)
                elif candidate_type == "low_AWR_high_BD":
                    strict = ds[ds["BD_high"]].sort_values("score_low_high", ascending=False).head(30)
                elif candidate_type == "TES_high_confidence":
                    strict = ds.sort_values("score_tes", ascending=False).head(30)
                elif candidate_type == "Exp1_late_stable_candidate":
                    strict = ds[ds["stage"].astype(int).eq(5)].sort_values(score_col, ascending=False).head(30)
                elif candidate_type == "Exp2_late_severe_candidate":
                    strict = ds[ds["stage"].astype(int).eq(5)].sort_values(score_col, ascending=False).head(30)
                else:
                    strict = ds.sort_values(score_col, ascending=False).head(30)
            picked, gap_relaxed = pick_spaced(
                strict,
                score_col,
                config.min_candidates_per_type,
                config.max_candidates_per_type,
                config.min_center_cycle_gap,
            )
            relaxed = relaxed or gap_relaxed
            if len(picked) < config.min_candidates_per_type:
                warnings.append(
                    f"{dataset} {candidate_type}: selected {len(picked)} candidates, below requested {config.min_candidates_per_type}."
                )
            priority, check = priority_and_check(candidate_type)
            for rank, row in enumerate(picked.itertuples(index=False), start=1):
                rows.append(
                    {
                        "dataset": dataset,
                        "candidate_type": candidate_type,
                        "candidate_rank": rank,
                        "window_id": int(getattr(row, "window_id")),
                        "window_index": int(getattr(row, "window_index")),
                        "direction_id": str(getattr(row, "direction_id")),
                        "center_cycle": float(getattr(row, "center_cycle")),
                        "start_cycle_effective": float(getattr(row, "start_cycle_effective", np.nan)),
                        "start_cycle_actual": float(getattr(row, "start_cycle_actual", np.nan)),
                        "end_cycle_effective": float(getattr(row, "end_cycle_effective", np.nan)),
                        "end_cycle_actual": float(getattr(row, "end_cycle_actual", np.nan)),
                        "center_cycle_effective": float(getattr(row, "center_cycle_effective", getattr(row, "center_cycle"))),
                        "center_cycle_actual": float(getattr(row, "center_cycle_actual", getattr(row, "center_cycle"))),
                        "stage": int(getattr(row, "stage")),
                        "AWR_model": str(getattr(row, "AWR_model")),
                        "AWR": float(getattr(row, "AWR")),
                        "AWR_percentile_within_dataset": float(getattr(row, "AWR_percentile_within_dataset")),
                        "BDall_xy_v2": float(getattr(row, "BDall_xy_v2")) if np.isfinite(getattr(row, "BDall_xy_v2")) else np.nan,
                        "BD_percentile_within_dataset": float(getattr(row, "BD_percentile_within_dataset")),
                        "RS_trend50": float(getattr(row, "RS_trend50")) if np.isfinite(getattr(row, "RS_trend50")) else np.nan,
                        "TES": float(getattr(row, "TES_for_selection")) if np.isfinite(getattr(row, "TES_for_selection")) else np.nan,
                        "is_TES_event": int(getattr(row, "is_TES_event")) if np.isfinite(getattr(row, "is_TES_event")) else 0,
                        "state_region": str(getattr(row, "state_region_main")),
                        "selection_reason": selection_text(candidate_type, relaxed),
                        "physical_validation_priority": priority,
                        "suggested_physical_check": check,
                    }
                )
    return pd.DataFrame(rows)


def build_z_pivot(z_table: pd.DataFrame, warnings: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if z_table.empty:
        warnings.append("z feature table missing; feature contributions cannot be computed.")
        return pd.DataFrame(), pd.DataFrame()
    z = z_table[z_table["feature_name"].isin(STABLE_PLUS)].copy()
    meta = z[["feature_name", "channel", "feature_family", "physical_meaning"]].drop_duplicates("feature_name")
    pivot = (
        z.pivot_table(index=["dataset", "window_index"], columns="feature_name", values="z_value", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return pivot, meta


def feature_contributions(
    candidates: pd.DataFrame,
    z_pivot: pd.DataFrame,
    feature_meta: pd.DataFrame,
    directions: pd.DataFrame,
    warnings: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty or z_pivot.empty or directions.empty:
        warnings.append("Candidate feature contribution inputs incomplete.")
        return pd.DataFrame(), pd.DataFrame()
    direction_lookup = {
        (str(row.direction_id), str(row.feature_name)): int(row.direction_sign)
        for row in directions.itertuples(index=False)
        if str(row.feature_name) in STABLE_PLUS
    }
    meta_lookup = {
        str(row.feature_name): {
            "channel": str(row.channel),
            "feature_family": str(row.feature_family),
            "physical_meaning": str(row.physical_meaning),
        }
        for row in feature_meta.itertuples(index=False)
    }
    z_lookup = z_pivot.set_index(["dataset", "window_index"])
    contribution_rows = []
    summary_rows = []
    for cand in candidates.itertuples(index=False):
        key = (str(cand.dataset), int(cand.window_index))
        if key not in z_lookup.index:
            warnings.append(f"Missing z values for {key}.")
            continue
        z_row = z_lookup.loc[key]
        all_rows = []
        for feature in STABLE_PLUS:
            z_value = float(z_row[feature]) if feature in z_row.index and np.isfinite(z_row[feature]) else 0.0
            sign = direction_lookup.get((str(cand.direction_id), feature), 1)
            signed = sign * z_value
            meta = meta_lookup.get(feature, {"channel": "", "feature_family": "", "physical_meaning": ""})
            all_rows.append(
                {
                    "dataset": str(cand.dataset),
                    "candidate_type": str(cand.candidate_type),
                    "center_cycle": float(cand.center_cycle),
                    "center_cycle_effective": float(getattr(cand, "center_cycle_effective", cand.center_cycle)),
                    "center_cycle_actual": float(getattr(cand, "center_cycle_actual", cand.center_cycle)),
                    "window_index": int(cand.window_index),
                    "feature_name": feature,
                    "channel": meta["channel"],
                    "feature_family": meta["feature_family"],
                    "direction_sign": int(sign),
                    "z_value": z_value,
                    "signed_contribution": signed,
                    "physical_meaning": meta["physical_meaning"],
                }
            )
        sorted_rows = sorted(all_rows, key=lambda item: abs(item["signed_contribution"]), reverse=True)
        for rank, item in enumerate(sorted_rows[:10], start=1):
            item["abs_contribution_rank"] = rank
            contribution_rows.append(item)
        full = pd.DataFrame(all_rows)
        channel_sum = full.groupby("channel")["signed_contribution"].sum().to_dict()
        channel_abs = full.groupby("channel")["signed_contribution"].apply(lambda s: float(np.abs(s).sum())).to_dict()
        family_sum = full.groupby("feature_family")["signed_contribution"].sum().to_dict()
        family_abs = full.groupby("feature_family")["signed_contribution"].apply(lambda s: float(np.abs(s).sum())).to_dict()
        dominant_channel = max(channel_abs, key=channel_abs.get) if channel_abs else ""
        dominant_family = max(family_abs, key=family_abs.get) if family_abs else ""
        hint = f"Dominant channel {dominant_channel}; dominant family {dominant_family}."
        summary_rows.append(
            {
                "dataset": str(cand.dataset),
                "candidate_type": str(cand.candidate_type),
                "center_cycle": float(cand.center_cycle),
                "center_cycle_effective": float(getattr(cand, "center_cycle_effective", cand.center_cycle)),
                "center_cycle_actual": float(getattr(cand, "center_cycle_actual", cand.center_cycle)),
                "window_index": int(cand.window_index),
                "rx_contribution": float(channel_sum.get("rx", 0.0)),
                "ry_contribution": float(channel_sum.get("ry", 0.0)),
                "rs_contribution": float(channel_sum.get("rs", 0.0)),
                "corrdist_contribution": float(sum(v for k, v in family_sum.items() if "corrdist" in str(k))),
                "amplitude_contribution": float(sum(v for k, v in family_sum.items() if "corrdist" not in str(k))),
                "dominant_channel": dominant_channel,
                "dominant_feature_family": dominant_family,
                "interpretation_hint": hint,
            }
        )
    return pd.DataFrame(contribution_rows), pd.DataFrame(summary_rows)


def plot_candidate_map(main: pd.DataFrame, candidates: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=False)
    for ax, dataset in zip(axes, sorted(main["dataset"].astype(str).unique())):
        ds = main[main["dataset"].astype(str) == dataset]
        cand = candidates[candidates["dataset"].astype(str) == dataset]
        ax.scatter(ds["BDall_xy_v2"], ds["AWR"], s=6, color="#9aa3ad", alpha=0.35, linewidths=0)
        if not cand.empty:
            ax.scatter(cand["BDall_xy_v2"], cand["AWR"], s=38, color="#b64b5a", marker="x", linewidths=1.3)
        ax.set_title(f"{dataset}: candidate AWR-BD map")
        ax.set_xlabel("BDall_xy_v2")
        ax.set_ylabel("M1_stable AWR")
        ax.grid(alpha=0.25)
    save_figure(fig, figures_dir / "fig_candidate_AWR_BD_map")


def add_boundaries(ax: plt.Axes, boundaries: pd.DataFrame, dataset: str) -> None:
    if boundaries.empty:
        return
    ds = boundaries[boundaries["dataset"].astype(str) == dataset]
    for row in ds.itertuples(index=False):
        boundary = float(getattr(row, "boundary_cycle_actual", getattr(row, "boundary_cycle")))
        ax.axvline(boundary, color="#777777", linestyle="--", linewidth=0.8, alpha=0.45)


def plot_candidate_timeseries(main: pd.DataFrame, candidates: pd.DataFrame, events: pd.DataFrame, boundaries: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=False)
    for ax, dataset in zip(axes, sorted(main["dataset"].astype(str).unique())):
        cycle_col = "center_cycle_actual" if "center_cycle_actual" in main.columns else "center_cycle"
        ds = main[main["dataset"].astype(str) == dataset].sort_values(cycle_col)
        cand = candidates[candidates["dataset"].astype(str) == dataset]
        ax.plot(ds[cycle_col], ds["AWR"], color="#3b6ea8", linewidth=1.0, label="M1_stable AWR")
        if not cand.empty:
            cand_cycle_col = "center_cycle_actual" if "center_cycle_actual" in cand.columns else "center_cycle"
            ax.scatter(cand[cand_cycle_col], cand["AWR"], color="#b64b5a", s=30, zorder=3, label="Candidate")
        ev = events[events["dataset"].astype(str) == dataset] if not events.empty else pd.DataFrame()
        if not ev.empty:
            high = ev[ev["high_confidence_event"].astype(bool)]
            for row in high.itertuples(index=False):
                peak_cycle = float(getattr(row, "peak_cycle_actual", getattr(row, "peak_cycle")))
                ax.axvline(peak_cycle, color="#c76d2a", linestyle=":", linewidth=0.8, alpha=0.5)
        add_boundaries(ax, boundaries, dataset)
        ax.set_title(f"{dataset}: AWR trajectory with candidates")
        ax.set_ylabel("M1_stable AWR")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Actual experimental cycle")
    save_figure(fig, figures_dir / "fig_candidate_AWR_timeseries")


def plot_feature_contributions(contrib: pd.DataFrame, figures_dir: Path) -> None:
    if contrib.empty:
        return
    top = (
        contrib.groupby("feature_name")["signed_contribution"]
        .apply(lambda s: float(np.mean(np.abs(s))))
        .sort_values(ascending=False)
        .head(12)
        .reset_index(name="mean_abs_signed_contribution")
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(np.arange(len(top)), top["mean_abs_signed_contribution"], color="#5a84b8")
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(top["feature_name"], rotation=35, ha="right")
    ax.set_ylabel("Mean abs signed contribution")
    ax.set_title("Candidate top feature contributions")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "fig_candidate_feature_contributions")


def plot_channel_family_summary(summary: pd.DataFrame, figures_dir: Path) -> None:
    if summary.empty:
        return
    agg = summary.groupby("candidate_type")[
        ["rx_contribution", "ry_contribution", "rs_contribution", "corrdist_contribution", "amplitude_contribution"]
    ].apply(lambda frame: frame.abs().mean()).reset_index()
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    x = np.arange(len(agg))
    bottom = np.zeros(len(agg))
    colors = ["#3b6ea8", "#c76d2a", "#46a67a", "#5a6fb0", "#7c8a99"]
    for col, color in zip(agg.columns[1:], colors):
        values = agg[col].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=col, color=color)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(agg["candidate_type"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean abs contribution")
    ax.set_title("Candidate channel/family contributions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, figures_dir / "fig_candidate_channel_family_contributions")


def plot_exp1_vs_exp2(candidates: pd.DataFrame, contrib: pd.DataFrame, figures_dir: Path) -> None:
    focus = candidates[
        candidates["candidate_type"].isin(["Exp1_late_stable_candidate", "Exp2_late_severe_candidate"])
    ].copy()
    if focus.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    label_cycle = focus["center_cycle_actual"] if "center_cycle_actual" in focus.columns else focus["center_cycle"]
    labels = focus["dataset"].astype(str) + "\n" + focus["candidate_type"].astype(str) + "\n" + label_cycle.round(0).astype(int).astype(str)
    axes[0].bar(np.arange(len(focus)), focus["AWR_percentile_within_dataset"], color="#3b6ea8", label="AWR pct")
    axes[0].bar(np.arange(len(focus)), focus["BD_percentile_within_dataset"], bottom=focus["AWR_percentile_within_dataset"], color="#c76d2a", label="BD pct")
    axes[0].set_xticks(np.arange(len(focus)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_title("Exp1 stable vs Exp2 severe candidates")
    axes[0].set_ylabel("Stacked percentiles")
    axes[0].legend(frameon=False)
    top = contrib[
        contrib["candidate_type"].isin(["Exp1_late_stable_candidate", "Exp2_late_severe_candidate"])
    ]
    if not top.empty:
        agg = top.groupby("feature_name")["signed_contribution"].apply(lambda s: float(np.mean(np.abs(s)))).sort_values(ascending=False).head(10)
        axes[1].bar(np.arange(len(agg)), agg.to_numpy(), color="#46a67a")
        axes[1].set_xticks(np.arange(len(agg)))
        axes[1].set_xticklabels(agg.index, rotation=45, ha="right", fontsize=8)
        axes[1].set_title("Top feature contributions")
        axes[1].grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "fig_exp1_stable_vs_exp2_severe")


def write_report(
    report_path: Path,
    candidates: pd.DataFrame,
    summary: pd.DataFrame,
    warnings: List[str],
    main_model: str,
    config: CandidateConfig,
) -> None:
    counts = candidates.groupby(["dataset", "candidate_type"]).size().reset_index(name="count") if not candidates.empty else pd.DataFrame()
    priority = candidates.sort_values(["physical_validation_priority", "dataset", "candidate_type", "candidate_rank"]) if not candidates.empty else pd.DataFrame()
    priority_columns = [
        "dataset",
        "candidate_type",
        "candidate_rank",
        "center_cycle_effective",
        "center_cycle_actual",
        "stage",
        "AWR",
        "AWR_percentile_within_dataset",
        "BDall_xy_v2",
        "BD_percentile_within_dataset",
        "physical_validation_priority",
    ]
    priority_columns = [col for col in priority_columns if priority.empty or col in priority.columns]
    mapping_cols = [
        "dataset",
        "stage",
        "effective_start",
        "effective_end",
        "actual_start",
        "actual_end",
        "slope_actual_per_effective",
        "note",
    ]
    mapping = cycle_mapping_table()
    lines = [
        "# Physical validation candidate report",
        "",
        "## Purpose",
        "",
        "This run selects representative windows for FEM, surface morphology, and debris closed-loop validation. It does not tune AWR models.",
        "",
        "## Main model",
        "",
        f"- Main AWR model: `{main_model}`.",
        "- `M0_stable` remains the equal-weight baseline.",
        "- `M2_stable` is kept as weighted sensitivity context, not as the main model.",
        "",
        "## Cycle mapping",
        "",
        "Candidate windows keep the original effective-cycle fields and add actual experimental-cycle fields.",
        "Mapping uses piecewise linear interpolation inside each user-provided stage. For Exp2, actual cycles 1-500 are NaN, so effective cycle 1 maps to actual cycle 501.",
        "",
        dataframe_to_markdown(mapping[mapping_cols]),
        "",
        "## Candidate types",
        "",
        "- `high_AWR_high_BD`: high late-state AWR form with clear baseline deviation.",
        "- `high_AWR_low_BD`: local late-state form or short sensitive-phase change.",
        "- `low_AWR_high_BD`: baseline deviation without high late-state AWR form.",
        "- `AWR_rising`: rapid local AWR increase.",
        "- `TES_high_confidence`: transition event neighborhood.",
        "- `Exp1_late_stable_candidate`: Exp1 Stage5 stable late-state check.",
        "- `Exp2_late_severe_candidate`: Exp2 Stage5 severe-state check.",
        "",
        "## Candidate counts",
        "",
        dataframe_to_markdown(counts),
        "",
        "## Priority candidate list",
        "",
        dataframe_to_markdown(
            priority[priority_columns].head(40)
            if not priority.empty
            else pd.DataFrame()
        ),
        "",
        "## Channel / family contribution hints",
        "",
        dataframe_to_markdown(summary.head(40) if not summary.empty else pd.DataFrame()),
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Physical-loop checks",
            "",
            "- For high_AWR_high_BD windows, check surface damage, debris increase, and FEM high-contribution contact zones.",
            "- For low_AWR_high_BD windows, check run-in, contact reorganization, or lubrication disturbance.",
            "- For TES event windows, check whether transitions align with disassembly, lubrication change, measurement disturbance, or real force-signal jumps.",
            "- For ry-dominant windows, check lateral contact migration or eccentric loading traces.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


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


def write_status(
    status_path: Path,
    candidates: pd.DataFrame,
    warnings: List[str],
    main_model: str,
    config: CandidateConfig,
) -> None:
    counts = candidates.groupby(["dataset", "candidate_type"]).size().reset_index(name="count") if not candidates.empty else pd.DataFrame()
    mapping_cols = [
        "dataset",
        "stage",
        "effective_start",
        "effective_end",
        "actual_start",
        "actual_end",
        "slope_actual_per_effective",
        "note",
    ]
    mapping = cycle_mapping_table()[mapping_cols]
    lines = [
        "# STATUS 2026-07-08",
        "",
        "## 1. 本轮完成了什么",
        "",
        "- 更新 `run_physical_validation_candidates.py`，将物理验证候选窗口从有效周期映射回实际实验周期。",
        "- 按用户确认节点执行阶段内线性映射：EXP1 为 8000-24000-32000-40000-53000；EXP2 为 5500-10500-15000-20000-24000，且 EXP2 前 500 个实际周期为 NaN。",
        "- 候选筛选、TES 事件、阶段边界、时间序列图和报告均新增 actual-cycle 口径；原 effective-cycle 字段保留用于追溯。",
        f"- 主 AWR 结构仍使用 `{main_model}`，本轮不重新训练 AWR，也不做 Stage1-Stage5 分类。",
        "",
        "### 候选窗口数量",
        "",
        dataframe_to_markdown(counts),
        "",
        "### 周期映射关系",
        "",
        dataframe_to_markdown(mapping),
        "",
        "## 2. 本轮生成的关键文件/数据路径",
        "",
        "- 脚本：`run_physical_validation_candidates.py`",
        "- STATUS：`docs/STATUS_20260708.md`",
        "- 候选窗口：`outputs_physical_validation_candidates_v1/results/physical_validation_candidates.csv`",
        "- 候选特征贡献：`outputs_physical_validation_candidates_v1/results/candidate_feature_contributions.csv`",
        "- 通道/特征族汇总：`outputs_physical_validation_candidates_v1/results/candidate_channel_family_summary.csv`",
        "- 周期映射表：`outputs_physical_validation_candidates_v1/results/cycle_mapping_table.csv`",
        "- 周期映射配置：`outputs_physical_validation_candidates_v1/configs/cycle_mapping_config.json`",
        "- 运行配置：`outputs_physical_validation_candidates_v1/configs/physical_validation_candidate_config.json`",
        "- 阶段报告：`outputs_physical_validation_candidates_v1/reports/physical_validation_candidate_report.md`",
        "- 图件：`outputs_physical_validation_candidates_v1/figures/fig_candidate_AWR_BD_map.png`",
        "- 图件：`outputs_physical_validation_candidates_v1/figures/fig_candidate_AWR_timeseries.png`",
        "- 图件：`outputs_physical_validation_candidates_v1/figures/fig_candidate_feature_contributions.png`",
        "- 图件：`outputs_physical_validation_candidates_v1/figures/fig_candidate_channel_family_contributions.png`",
        "- 图件：`outputs_physical_validation_candidates_v1/figures/fig_exp1_stable_vs_exp2_severe.png`",
        "- 运行日志：`outputs_physical_validation_candidates_v1/physical_validation_candidates_run.log`",
        "",
        "## 3. 下一步建议做什么",
        "",
        "- 用 `center_cycle_actual` 作为物理闭环验证的主定位字段，优先回查表面形貌、FEM 接触区域和磨屑证据。",
        "- 检查 Exp1 Stage5 stable 候选与 Exp2 Stage5 severe 候选是否在实际周期尺度上对应合理的磨损阶段。",
        "- 对 TES_high_confidence 候选单独核对实验记录，区分真实状态突变、润滑/装拆扰动和测量异常。",
        "- 后续若补充物理证据，可在候选表基础上增加验证结果字段，而不是重新训练 AWR 或改成分类任务。",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- None.")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = CandidateConfig()
    dirs = setup_dirs(config)
    setup_logging(dirs)
    warnings: List[str] = []

    fair_scores = load_csv("fair_scores", warnings)
    fair_weights = load_csv("fair_weights", warnings)
    fair_directions = load_csv("fair_directions", warnings)
    state_weighted = load_csv("state_weighted", warnings)
    state_v2 = load_csv("state_v2", warnings)
    bd_thresholds = load_csv("bd_thresholds", warnings)
    boundaries = load_csv("boundaries", warnings)
    tes_events = load_csv("tes_events", warnings)
    z_table = load_csv("z_table", warnings)
    boundaries = add_boundary_actual_cycles(boundaries)
    tes_events = add_event_actual_cycles(tes_events)
    cycle_mapping = cycle_mapping_table()

    if PATHS["fair_decision"].exists():
        fair_decision = json.loads(PATHS["fair_decision"].read_text(encoding="utf-8"))
    else:
        fair_decision = {}
        warnings.append(f"Missing input file: {PATHS['fair_decision']}")

    main_frame, main_model = prepare_main_frame(
        fair_scores, state_weighted, state_v2, bd_thresholds, tes_events, config, warnings
    )
    candidates = build_candidates(main_frame, config, warnings) if not main_frame.empty else pd.DataFrame()
    z_pivot, feature_meta = build_z_pivot(z_table, warnings)
    contributions, channel_summary = feature_contributions(candidates, z_pivot, feature_meta, fair_directions, warnings)

    candidates.to_csv(dirs["results"] / "physical_validation_candidates.csv", index=False, encoding="utf-8-sig")
    contributions.to_csv(dirs["results"] / "candidate_feature_contributions.csv", index=False, encoding="utf-8-sig")
    channel_summary.to_csv(dirs["results"] / "candidate_channel_family_summary.csv", index=False, encoding="utf-8-sig")
    cycle_mapping.to_csv(dirs["results"] / "cycle_mapping_table.csv", index=False, encoding="utf-8-sig")
    cycle_mapping_config = {
        "formula": "actual = actual_start + (effective - effective_start) * (actual_end - actual_start) / (effective_end - effective_start)",
        "exp2_nan_prefix_actual_cycles": [1, 500],
        "segments": CYCLE_MAPPING_SEGMENTS,
    }
    (dirs["configs"] / "cycle_mapping_config.json").write_text(
        json.dumps(cycle_mapping_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dirs["configs"] / "physical_validation_candidate_config.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "main_model_resolved": main_model,
                "fair_decision": fair_decision,
                "cycle_mapping": cycle_mapping_config,
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_candidate_map(main_frame, candidates, dirs["figures"])
    plot_candidate_timeseries(main_frame, candidates, tes_events, boundaries, dirs["figures"])
    plot_feature_contributions(contributions, dirs["figures"])
    plot_channel_family_summary(channel_summary, dirs["figures"])
    plot_exp1_vs_exp2(candidates, contributions, dirs["figures"])

    write_report(
        dirs["reports"] / "physical_validation_candidate_report.md",
        candidates,
        channel_summary,
        warnings,
        main_model,
        config,
    )
    write_status(Path(config.status_file), candidates, warnings, main_model, config)

    logging.info("Physical validation candidate selection complete.")
    logging.info("Main AWR model: %s", main_model)
    logging.info("Output directory: %s", config.output_dir)
    logging.info("STATUS file: %s", config.status_file)
    print("Physical validation candidate selection complete.")
    print(f"Main AWR model: {main_model}")
    print(f"Output directory: {config.output_dir}/")
    print(f"STATUS file: {config.status_file}")


if __name__ == "__main__":
    main()
