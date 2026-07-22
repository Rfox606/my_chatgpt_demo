from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV2Config
from .data import assert_label_free


def _spaced_regions(frame: pd.DataFrame, mask: pd.Series, score: pd.Series, kind: str, config: ContinuousStateV2Config) -> list[dict[str, object]]:
    selected = frame.loc[mask & frame.is_restart_guard.eq(0)].copy()
    if selected.empty:
        return []
    selected["_score"] = score.loc[selected.index]
    ordered = selected.sort_values(["_score", "center_cycle"], ascending=[False, True])
    rows = []
    peaks: list[float] = []
    for _, peak in ordered.iterrows():
        if any(abs(float(peak.center_cycle) - previous) < config.known_stop_interval_cycles for previous in peaks):
            continue
        region = selected.loc[(selected.center_cycle >= peak.center_cycle - 50) & (selected.center_cycle <= peak.center_cycle + 50)]
        rows.append({"direction_id": peak.direction_id, "source_dataset": peak.source_dataset, "target_dataset": peak.target_dataset, "candidate_type": kind, "candidate_start_cycle": float(region.start_cycle.min()), "candidate_end_cycle": float(region.end_cycle.max()), "peak_cycle": float(peak.center_cycle), "duration_windows": int(len(region)), "peak_P_common": float(peak.P_common), "peak_BD": float(peak.BD), "peak_B_terminal": float(peak.B_terminal), "peak_TES": float(peak.TES), "peak_weighted_oos": float(peak.weighted_oos_common), "diagnostic_only": 1, "not_an_online_alarm": 1, "requires_physical_validation": 1})
        peaks.append(float(peak.center_cycle))
        if len(rows) >= 20:
            break
    return rows


def select_candidates(scores: pd.DataFrame, config: ContinuousStateV2Config) -> pd.DataFrame:
    assert_label_free(scores)
    rows = []
    for _, frame in scores.loc[scores.dataset_role.eq("target")].groupby("direction_id", sort=True):
        q = frame.copy()
        p90, bd90, b90 = q.P_smooth_20.quantile(.9), q.BD.quantile(.9), q.B_terminal.quantile(.9)
        p50, b10 = q.P_smooth_20.quantile(.5), q.B_terminal.quantile(.1)
        rules = [
            ("high_P_high_BD", (q.P_smooth_20 >= p90) & (q.BD >= bd90), q.P_smooth_20 + q.BD),
            ("high_BD_low_P", (q.BD >= bd90) & (q.P_smooth_20 <= p50), q.BD),
            ("stable_branch_candidate", q.B_terminal <= b10, -q.B_terminal),
            ("severe_branch_candidate", q.B_terminal >= b90, q.B_terminal),
            ("rapid_P_growth", q.P_RS50 >= q.P_RS50.quantile(.95), q.P_RS50),
            ("rapid_BD_growth", q.BD_RS50 >= q.BD_RS50.quantile(.95), q.BD_RS50),
            ("branch_transition", q.B_RS50.abs() >= q.B_RS50.abs().quantile(.95), q.B_RS50.abs()),
            ("high_TES", q.TES >= q.TES.quantile(.95), q.TES),
            ("high_weighted_OOS", q.weighted_oos_common >= q.weighted_oos_common.quantile(.95), q.weighted_oos_common),
        ]
        for kind, mask, score in rules:
            rows.extend(_spaced_regions(q, mask, score, kind, config))
    return pd.DataFrame(rows)
