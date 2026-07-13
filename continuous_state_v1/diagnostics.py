from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .config import ContinuousStateV1Config
from .data import FORBIDDEN_COLUMNS, assert_label_free
from .pair_sampling import sample_temporal_pairs


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def source_support_table(
    source: pd.DataFrame, direction_id: str, config: ContinuousStateV1Config
) -> pd.DataFrame:
    assert_label_free(source)
    usable = source.loc[source["is_restart_guard"].astype(int) == 0]
    rows = []
    for feature in config.stable_plus_features:
        values = usable[feature].to_numpy(float)
        rows.append(
            {
                "direction_id": direction_id,
                "source_dataset": str(source["dataset"].iloc[0]),
                "feature_name": feature,
                "p01": float(np.percentile(values, 1)),
                "p05": float(np.percentile(values, 5)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def add_target_support_scores(
    target_scores: pd.DataFrame, source_support: pd.DataFrame, config: ContinuousStateV1Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_label_free(target_scores)
    result = target_scores.copy()
    flags: list[np.ndarray] = []
    detail_rows: list[dict[str, object]] = []
    for feature in config.stable_plus_features:
        limits = source_support.loc[source_support["feature_name"] == feature].iloc[0]
        values = result[feature].to_numpy(float)
        outside = (values < float(limits["p01"])) | (values > float(limits["p99"]))
        flags.append(outside)
        detail_rows.append(
            {
                "direction_id": result["direction_id"].iloc[0],
                "source_dataset": result["source_dataset"].iloc[0],
                "target_dataset": result["target_dataset"].iloc[0],
                "feature_name": feature,
                "target_window_count": int(len(result)),
                "oos_window_count": int(outside.sum()),
                "oos_fraction": float(outside.mean()),
                "source_p01": float(limits["p01"]),
                "source_p99": float(limits["p99"]),
            }
        )
    matrix = np.vstack(flags)
    result["oos_feature_count"] = matrix.sum(axis=0).astype(int)
    result["oos_fraction"] = result["oos_feature_count"] / len(config.stable_plus_features)
    result["max_abs_target_z"] = result.loc[:, list(config.stable_plus_features)].abs().max(axis=1)
    return result, pd.DataFrame(detail_rows)


def target_temporal_concordance(
    target_scores: pd.DataFrame, config: ContinuousStateV1Config, random_seed: int
) -> dict[str, float | int | str]:
    """Unlabelled, post-hoc time-consistency diagnostics only."""
    assert_label_free(target_scores)
    pairs = sample_temporal_pairs(target_scores, config, random_seed=random_seed)
    lookup = target_scores.set_index("window_index")
    rows: dict[str, float | int | str] = {
        "direction_id": str(target_scores["direction_id"].iloc[0]),
        "source_dataset": str(target_scores["source_dataset"].iloc[0]),
        "target_dataset": str(target_scores["target_dataset"].iloc[0]),
    }
    if pairs.empty:
        rows.update({
            "target_pair_concordance_all": float("nan"),
            "target_pair_concordance_gap_500_2000": float("nan"),
            "target_pair_concordance_gap_2000_5000": float("nan"),
            "target_pair_concordance_gap_5000_plus": float("nan"),
            "target_long_gap_concordance": float("nan"),
            "spearman_AWR_cycle": float("nan"),
            "kendall_AWR_cycle": float("nan"),
            "spearman_BD_cycle": float("nan"),
        })
        return rows
    earlier = lookup.loc[pairs["earlier_window_index"], "AWR_rel"].to_numpy(float)
    later = lookup.loc[pairs["later_window_index"], "AWR_rel"].to_numpy(float)
    concordance = (later > earlier).astype(float) + 0.5 * (later == earlier).astype(float)
    rows["target_pair_count"] = int(len(pairs))
    rows["target_pair_concordance_all"] = float(concordance.mean())
    mapping = {
        "gap_500_2000": "target_pair_concordance_gap_500_2000",
        "gap_2000_5000": "target_pair_concordance_gap_2000_5000",
        "gap_5000_plus": "target_pair_concordance_gap_5000_plus",
    }
    for bin_name, key in mapping.items():
        mask = pairs["gap_bin"].eq(bin_name).to_numpy()
        rows[key] = float(concordance[mask].mean()) if mask.any() else float("nan")
    long_mask = pairs["cycle_gap"].to_numpy(float) >= 2000.0
    rows["target_long_gap_concordance"] = float(concordance[long_mask].mean()) if long_mask.any() else float("nan")
    usable = target_scores.loc[target_scores["is_restart_guard"].astype(int) == 0]
    cycles = usable["center_cycle"].to_numpy(float)
    rows["spearman_AWR_cycle"] = float(spearmanr(cycles, usable["AWR_rel"].to_numpy(float)).statistic)
    rows["kendall_AWR_cycle"] = float(kendalltau(cycles, usable["AWR_rel"].to_numpy(float)).statistic)
    rows["spearman_BD_cycle"] = float(spearmanr(cycles, usable["BD"].to_numpy(float)).statistic)
    return rows


def baseline_stability(target_scores: pd.DataFrame, baseline_mask: np.ndarray) -> dict[str, float | int | str]:
    assert_label_free(target_scores)
    subset = target_scores.loc[baseline_mask]
    return {
        "direction_id": str(target_scores["direction_id"].iloc[0]),
        "source_dataset": str(target_scores["source_dataset"].iloc[0]),
        "target_dataset": str(target_scores["target_dataset"].iloc[0]),
        "baseline_window_count": int(len(subset)),
        "baseline_AWR_rel_median": float(np.median(subset["AWR_rel"])),
        "baseline_AWR_rel_IQR": float(np.percentile(subset["AWR_rel"], 75) - np.percentile(subset["AWR_rel"], 25)),
        "baseline_BD_median": float(np.median(subset["BD"])),
        "baseline_BD_p95": float(np.percentile(subset["BD"], 95)),
        "baseline_oos_fraction_mean": float(np.mean(subset["oos_fraction"])),
    }


def no_stage_leakage_check(*frames: pd.DataFrame) -> dict[str, object]:
    leaked = sorted(set().union(*(FORBIDDEN_COLUMNS.intersection(frame.columns) for frame in frames)))
    return {"status": "PASS" if not leaked else "FAIL", "forbidden_columns_found": leaked}
