from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV1Config
from .data import assert_label_free


CANDIDATE_TYPES = (
    "high_AWR_high_BD",
    "high_AWR_low_BD",
    "low_AWR_high_BD",
    "largest_AWR_increase",
    "largest_BD_increase",
    "high_out_of_support",
)


def _spaced_top_rows(
    frame: pd.DataFrame, sort_column: str, config: ContinuousStateV1Config
) -> pd.DataFrame:
    ordered = frame.sort_values([sort_column, "center_cycle"], ascending=[False, True], kind="stable")
    chosen: list[int] = []
    centers: list[float] = []
    for index, row in ordered.iterrows():
        center = float(row["center_cycle"])
        if all(abs(center - prior) >= config.candidate_min_spacing_cycles for prior in centers):
            chosen.append(index)
            centers.append(center)
        if len(chosen) >= config.candidate_top_k_per_type:
            break
    return frame.loc[chosen].copy()


def _top_contributions(row: pd.Series, features: tuple[str, ...]) -> dict[str, object]:
    contributions = [(feature, float(row[f"contrib_{feature}"])) for feature in features]
    contributions.sort(key=lambda item: (-abs(item[1]), item[0]))
    output: dict[str, object] = {}
    for position in range(3):
        feature, contribution = contributions[position]
        output[f"top_feature_{position + 1}"] = feature
        output[f"top_contribution_{position + 1}"] = contribution
    return output


def select_physical_validation_candidates(
    target_scores: pd.DataFrame, config: ContinuousStateV1Config
) -> pd.DataFrame:
    """Offline inspection shortlist; this intentionally has no online threshold semantics."""
    assert_label_free(target_scores)
    eligible = target_scores.loc[target_scores["is_restart_guard"].astype(int) == 0].copy()
    if eligible.empty:
        return pd.DataFrame()
    awr_high = float(eligible["AWR_rel"].quantile(0.90))
    awr_low = float(eligible["AWR_rel"].quantile(0.50))
    bd_high = float(eligible["BD"].quantile(0.90))
    bd_low = float(eligible["BD"].quantile(0.50))
    eligible["AWR_increase"] = eligible["AWR_rel"].diff().fillna(0.0)
    eligible["BD_increase"] = eligible["BD"].diff().fillna(0.0)
    sets: dict[str, tuple[pd.DataFrame, str]] = {
        "high_AWR_high_BD": (eligible[(eligible.AWR_rel >= awr_high) & (eligible.BD >= bd_high)], "AWR_rel"),
        "high_AWR_low_BD": (eligible[(eligible.AWR_rel >= awr_high) & (eligible.BD <= bd_low)], "AWR_rel"),
        "low_AWR_high_BD": (eligible[(eligible.AWR_rel <= awr_low) & (eligible.BD >= bd_high)], "BD"),
        "largest_AWR_increase": (eligible, "AWR_increase"),
        "largest_BD_increase": (eligible, "BD_increase"),
        "high_out_of_support": (eligible, "oos_fraction"),
    }
    rows: list[pd.DataFrame] = []
    requested = [
        "direction_id", "source_dataset", "target_dataset", "window_index", "start_cycle", "end_cycle",
        "center_cycle", "AWR_raw", "AWR_rel", "AWR_scaled", "BD", "BD_diag", "oos_fraction",
        "is_restart_guard",
    ]
    for candidate_type, (candidates, sort_column) in sets.items():
        selected = _spaced_top_rows(candidates, sort_column, config)
        if selected.empty:
            continue
        output = selected.loc[:, requested].copy()
        output["candidate_type"] = candidate_type
        tops = selected.apply(_top_contributions, axis=1, features=config.stable_plus_features, result_type="expand")
        output = pd.concat([output.reset_index(drop=True), tops.reset_index(drop=True)], axis=1)
        output["diagnostic_only"] = 1
        output["not_an_online_threshold"] = 1
        rows.append(output)
    if not rows:
        return pd.DataFrame()
    columns = [
        "direction_id", "source_dataset", "target_dataset", "candidate_type", "window_index", "start_cycle",
        "end_cycle", "center_cycle", "AWR_raw", "AWR_rel", "AWR_scaled", "BD", "BD_diag",
        "oos_fraction", "is_restart_guard", "top_feature_1", "top_contribution_1", "top_feature_2",
        "top_contribution_2", "top_feature_3", "top_contribution_3", "diagnostic_only",
        "not_an_online_threshold",
    ]
    return pd.concat(rows, ignore_index=True).loc[:, columns]
