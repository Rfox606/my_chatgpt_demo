from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .config import RY_EXTENSION_FEATURES, ContinuousStateV44Config
from .data import baseline_mask, robust_location_scale


def _variant_consensus_comparison(variant_consensus: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in sorted({str(frame.dataset.iloc[0]) for frame in variant_consensus.values()}):
        for left_name, right_name in combinations(variant_consensus, 2):
            left = variant_consensus[left_name].loc[variant_consensus[left_name].dataset.eq(dataset), ["window_index", "D_state_q50", "V500_norm_q50", "V1000_norm_q50", "multi_scale_rate_divergence_q50", "state_volatility_q50"]]
            right = variant_consensus[right_name].loc[variant_consensus[right_name].dataset.eq(dataset), ["window_index", "D_state_q50", "V500_norm_q50", "V1000_norm_q50", "multi_scale_rate_divergence_q50", "state_volatility_q50"]]
            merged = left.merge(right, on="window_index", suffixes=("_left", "_right"))
            for metric in ("D_state", "V500_norm", "V1000_norm", "multi_scale_rate_divergence", "state_volatility"):
                rows.append({"row_type": "variant_trajectory_comparison", "dataset": dataset, "feature_variant": f"{left_name}_vs_{right_name}", "feature_name": metric, "common_windows": int(len(merged)),
                             "spearman_with_comparator": float(merged[f"{metric}_q50_left"].corr(merged[f"{metric}_q50_right"], method="spearman"))})
    return rows


def ry_group_audit(frame_by_dataset: dict[str, pd.DataFrame], extended_canonical_states: dict[str, pd.DataFrame], variant_consensus: dict[str, pd.DataFrame], config: ContinuousStateV44Config) -> pd.DataFrame:
    """Audit the expanded ry group without consulting morphology or changing any state parameter."""
    rows = _variant_consensus_comparison(variant_consensus)
    ry_features = ("ry_p2p", *RY_EXTENSION_FEATURES)
    for dataset, frame in frame_by_dataset.items():
        base = baseline_mask(frame, config); monitor = frame.start_cycle_effective.to_numpy(float) > config.baseline_cycles
        raw = frame.loc[:, list(ry_features)].to_numpy(float)
        location, scale = robust_location_scale(raw[base], config.eps); standardized = (raw - location) / scale
        absolute = np.abs(standardized); shares = absolute / np.maximum(absolute.sum(axis=1, keepdims=True), config.eps)
        state = extended_canonical_states[dataset].loc[:, ["window_index", "D_ry_subspace"]]
        index = frame.loc[monitor, ["window_index"]].copy()
        audit = index.copy()
        for position, feature in enumerate(ry_features):
            audit[feature] = standardized[monitor, position]; audit[f"abs_{feature}"] = absolute[monitor, position]; audit[f"share_{feature}"] = shares[monitor, position]
        audit = audit.merge(state, on="window_index", how="inner")
        pairwise = [float(audit[left].corr(audit[right], method="spearman")) for left, right in combinations(ry_features, 2)]
        rows.append({"row_type": "expanded_group_summary", "dataset": dataset, "feature_variant": "ry_extended", "feature_name": "ry_group", "common_windows": int(len(audit)),
                     "median_pairwise_signed_spearman": float(np.median(pairwise)), "q25_pairwise_signed_spearman": float(np.quantile(pairwise, .25)), "q75_pairwise_signed_spearman": float(np.quantile(pairwise, .75)),
                     "ry_p2p_mean_absolute_share": float(audit.share_ry_p2p.mean()), "ry_p2p_p95_absolute_share": float(audit.share_ry_p2p.quantile(.95)), "ry_p2p_share_over_050_fraction": float((audit.share_ry_p2p > .50).mean())})
        for feature in ry_features:
            rows.append({"row_type": "expanded_feature", "dataset": dataset, "feature_variant": "ry_extended", "feature_name": feature, "common_windows": int(len(audit)),
                         "feature_to_group_distance_spearman": float(audit[f"abs_{feature}"].corr(audit.D_ry_subspace, method="spearman")),
                         "mean_absolute_share": float(audit[f"share_{feature}"].mean()), "p95_absolute_share": float(audit[f"share_{feature}"].quantile(.95)), "share_over_050_fraction": float((audit[f"share_{feature}"] > .50).mean())})
    return pd.DataFrame(rows)
