from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import assert_label_free


def target_segment_diagnostics(scores: pd.DataFrame) -> pd.DataFrame:
    assert_label_free(scores)
    target = scores.loc[scores.dataset_role.eq("target")].copy()
    rows = []
    for direction, frame in target.groupby("direction_id", sort=True):
        usable = frame.loc[frame.is_restart_guard.eq(0)].sort_values("center_cycle").reset_index(drop=True)
        n = len(usable)
        row: dict[str, object] = {"direction_id": direction, "target_dataset": usable.dataset.iloc[0]}
        for name, lo, hi in (("early", 0., .30), ("middle", .30, .70), ("late", .70, 1.)):
            part = usable.iloc[int(n * lo):int(n * hi)]
            for metric, prefix in (("P_common", "P"), ("BD", "BD"), ("B_terminal", "B")):
                row[f"{prefix}_cycle_spearman_{name}"] = float(spearmanr(part.center_cycle, part[metric]).statistic)
        high_bd = usable.BD >= usable.BD.quantile(.9)
        high_p = usable.P_common >= usable.P_common.quantile(.9)
        low_p = usable.P_common <= usable.P_common.quantile(.5)
        row.update({"P_BD_joint_high_rate": float((high_bd & high_p).mean()), "high_BD_low_P_rate": float((high_bd & low_p).mean()), "high_BD_positive_B_rate": float((high_bd & (usable.B_terminal > 0)).mean()), "high_BD_negative_B_rate": float((high_bd & (usable.B_terminal < 0)).mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def frozen_vs_adaptive_summary(scores: pd.DataFrame, adapter_log: pd.DataFrame) -> pd.DataFrame:
    assert_label_free(scores)
    rows = []
    for direction, frame in scores.loc[scores.dataset_role.eq("target")].groupby("direction_id"):
        baseline = frame.loc[(frame.end_cycle <= 500) & frame.is_restart_guard.eq(0)]
        rows.append({"direction_id": direction, "baseline_P_drift": float(abs(np.median(baseline.P_common))), "baseline_BD_median": float(np.median(baseline.BD)), "weighted_oos_common_mean": float(frame.weighted_oos_common.mean()), "P_short_volatility_median": float(frame.P_short_volatility.median()), "BD_short_volatility": float(frame.BD.diff().abs().median()), "adapter_update_rate": float(frame.adapter_updated.mean()), "rollback_count": int(frame.adapter_rollback.sum()), "beta_norm_final": float(frame.beta_norm.iloc[-1])})
    return pd.DataFrame(rows)


def online_benefit(metrics: pd.DataFrame, frozen_adapted: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for direction, frame in metrics.groupby("direction_id"):
        frozen = frame.loc[frame.model.eq("Frozen")].set_index("horizon_cycles")
        online = frame.loc[frame.model.eq("Online_RLS")].set_index("horizon_cycles")
        benefits = []
        for horizon in sorted(set(frozen.index).intersection(online.index)):
            benefits.append((online.loc[horizon, "MAE_P"] <= .95 * frozen.loc[horizon, "MAE_P"]) or (online.loc[horizon, "MAE_BD"] <= .95 * frozen.loc[horizon, "MAE_BD"]))
        drift = float(frozen_adapted.loc[frozen_adapted.direction_id.eq(direction), "baseline_P_drift"].iloc[0])
        status = "PASS" if sum(benefits) >= 2 and drift <= .10 else "FAIL"
        rows.append({"direction_id": direction, "beneficial_horizon_count": int(sum(benefits)), "baseline_replay_drift_ok": drift <= .10, "ONLINE_ADAPTATION_BENEFIT": status})
    table = pd.DataFrame(rows)
    return table, {"status": "PASS" if bool((table.ONLINE_ADAPTATION_BENEFIT == "PASS").any()) else "FAIL", "by_direction": table.to_dict(orient="records")}
