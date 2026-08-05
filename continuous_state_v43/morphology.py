from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV43Config


MORPHOLOGY = {
    "cycle_actual": (0, 8000, 16000, 24000, 32000, 40000, 48000),
    "Sa": (5.320, 5.875, 5.040, 4.231, 5.486, 3.944, 4.343),
    "Sq": (6.768, 7.282, 6.389, 5.597, 7.188, 5.310, 5.750),
    "Sz": (95.504, 84.294, 99.029, 114.806, 94.649, 103.178, 90.551),
    "Sku": (3.305, 3.032, 6.060, 8.216, 6.217, 10.786, 6.781),
}


def _at_anchors(consensus: pd.DataFrame) -> pd.DataFrame:
    anchors = pd.DataFrame(MORPHOLOGY)
    rows: list[dict[str, object]] = []
    for _, anchor in anchors.iterrows():
        nearest = consensus.iloc[(consensus.center_cycle_actual - float(anchor.cycle_actual)).abs().argmin()]
        observed = bool(nearest.start_cycle_actual <= anchor.cycle_actual <= nearest.end_cycle_actual)
        row = dict(anchor); row.update({"nearest_center_cycle_actual": float(nearest.center_cycle_actual), "anchor_observed": observed})
        for column in ("D_state_q50", "multi_scale_rate_divergence_q50", "change_configuration_support", "combined_change_score_q50"):
            row[column] = float(nearest[column]) if observed else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def morphology_posthoc(full: pd.DataFrame, no_ry: pd.DataFrame, raw_exp1: pd.DataFrame, canonical_exp1: pd.DataFrame, config: ContinuousStateV43Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sparse, post-hoc morphology correspondence only; no returned value enters state computation."""
    views = {"full": _at_anchors(full), "no_ry": _at_anchors(no_ry)}
    rows: list[dict[str, object]] = []
    for view, anchors in views.items():
        observed = anchors.loc[anchors.anchor_observed]
        for state_metric in ("D_state_q50", "multi_scale_rate_divergence_q50", "change_configuration_support"):
            for morphology_metric in ("Sa", "Sq", "Sz", "Sku"):
                valid = observed.loc[observed[state_metric].notna()]
                rows.append({"analysis": "sparse_anchor_spearman", "view": view, "state_metric": state_metric, "morphology_metric": morphology_metric,
                             "n": int(len(valid)), "spearman_rho": float(valid[state_metric].corr(valid[morphology_metric], method="spearman")) if len(valid) >= 3 else np.nan})
        for morphology_metric in ("Sa", "Sq", "Sz", "Sku"):
            valid = observed.sort_values("cycle_actual")
            changes = valid[morphology_metric].diff().abs().iloc[1:]
            support = valid.change_configuration_support.iloc[1:]
            rows.append({"analysis": "adjacent_abs_morphology_change_vs_episode_support", "view": view, "state_metric": "change_configuration_support",
                         "morphology_metric": morphology_metric, "n": int(len(changes)), "spearman_rho": float(changes.corr(support, method="spearman")) if len(changes) >= 3 else np.nan})
    full_anchors = views["full"]; no_ry_anchors = views["no_ry"]
    raw_rows: list[dict[str, object]] = []
    for _, anchor in pd.DataFrame(MORPHOLOGY).iterrows():
        nearest_raw = raw_exp1.iloc[(raw_exp1.center_cycle_actual - float(anchor.cycle_actual)).abs().argmin()]
        state = canonical_exp1.iloc[(canonical_exp1.center_cycle_actual - float(anchor.cycle_actual)).abs().argmin()]
        full_row = full_anchors.loc[full_anchors.cycle_actual.eq(anchor.cycle_actual)].iloc[0]
        no_ry_row = no_ry_anchors.loc[no_ry_anchors.cycle_actual.eq(anchor.cycle_actual)].iloc[0]
        dominance = float(state.D_ry_subspace / max(state.D_rs_subspace + state.D_rx_subspace + state.D_ry_subspace, config.eps))
        raw_rows.append({**dict(anchor), "row_type": "anchor", "ry_p2p": float(nearest_raw.ry_p2p), "ry_anchor_cycle_actual": float(nearest_raw.center_cycle_actual),
                         "ry_subspace_dominance": dominance, "full_D_state": full_row.D_state_q50, "no_ry_D_state": no_ry_row.D_state_q50,
                         "full_support": full_row.change_configuration_support, "no_ry_support": no_ry_row.change_configuration_support,
                         "state_anchor_observed": bool(full_row.anchor_observed)})
    audit = pd.DataFrame(raw_rows)
    observed = audit.loc[audit.state_anchor_observed]
    for morphology_metric in ("Sa", "Sq", "Sz", "Sku"):
        raw_rows.append({"row_type": "summary", "morphology_metric": morphology_metric, "analysis": "ry_p2p_sparse_spearman",
                         "n": int(len(observed)), "ry_p2p_spearman": float(observed.ry_p2p.corr(observed[morphology_metric], method="spearman")),
                         "full_D_spearman": float(observed.full_D_state.corr(observed[morphology_metric], method="spearman")),
                         "no_ry_D_spearman": float(observed.no_ry_D_state.corr(observed[morphology_metric], method="spearman")),
                         "ry_improves_absolute_D_correspondence": bool(abs(observed.full_D_state.corr(observed[morphology_metric], method="spearman")) > abs(observed.no_ry_D_state.corr(observed[morphology_metric], method="spearman")))})
    return pd.DataFrame(rows), pd.DataFrame(raw_rows)
