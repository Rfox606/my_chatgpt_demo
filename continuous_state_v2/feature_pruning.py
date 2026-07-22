from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import ContinuousStateV2Config, STABLE_PLUS_FEATURES
from .data import assert_label_free
from .temporal_pairs import build_pair_batch


PHYSICAL_DIRECTNESS = {
    "rs_corrdist_base": 3, "rs_mean": 1, "rs_q05": 2, "rx_corrdist_base": 3,
    "rs_rms": 2, "ry_p2p": 2, "rx_mean": 1, "rx_absmean": 2, "rx_q05": 2,
}


def _single_feature_auc(frame: pd.DataFrame, feature: str, config: ContinuousStateV2Config, seed: int) -> float:
    batch = build_pair_batch(frame, (feature,), config, seed)
    if not batch.pair_count:
        return float("nan")
    return float(roc_auc_score(batch.labels, batch.delta_x[:, 0]))


def prune_features(source_train: pd.DataFrame, direction_id: str, config: ContinuousStateV2Config) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Prune from source-train only; target data never reaches this function."""
    assert_label_free(source_train)
    usable = source_train.loc[source_train.is_restart_guard.eq(0)].copy()
    features = [item for item in STABLE_PLUS_FEATURES if item != "rs_absmean"]
    aucs = {feature: _single_feature_auc(usable, feature, config, config.pair_random_seed + index) for index, feature in enumerate(features)}
    rows = [{"direction_id": direction_id, "feature_name": "rs_absmean", "kept": 0, "drop_reason": "EXACT_DUPLICATE_OF_rs_mean", "correlated_with": "rs_mean", "single_feature_pair_auc": np.nan}]
    active = list(features)
    correlation = usable.loc[:, active].corr(method="pearson")
    for left_index, left in enumerate(features):
        if left not in active:
            continue
        for right in features[left_index + 1:]:
            if left not in active:
                break
            if right not in active or abs(float(correlation.loc[left, right])) < config.correlation_prune_threshold:
                continue
            left_auc, right_auc = aucs[left], aucs[right]
            if abs(left_auc - right_auc) >= 0.005:
                keep, drop = (left, right) if left_auc > right_auc else (right, left)
            elif PHYSICAL_DIRECTNESS[left] != PHYSICAL_DIRECTNESS[right]:
                keep, drop = (left, right) if PHYSICAL_DIRECTNESS[left] < PHYSICAL_DIRECTNESS[right] else (right, left)
            else:
                keep, drop = sorted((left, right))
            active.remove(drop)
            rows.append({"direction_id": direction_id, "feature_name": drop, "kept": 0, "drop_reason": "HIGH_CORRELATION_SOURCE_TRAIN", "correlated_with": keep, "single_feature_pair_auc": aucs[drop]})
    dropped = {row["feature_name"] for row in rows}
    for feature in STABLE_PLUS_FEATURES:
        if feature not in dropped:
            rows.append({"direction_id": direction_id, "feature_name": feature, "kept": 1, "drop_reason": "KEPT", "correlated_with": "", "single_feature_pair_auc": aucs[feature]})
    audit = pd.DataFrame(rows).sort_values(["kept", "feature_name"], ascending=[False, True]).reset_index(drop=True)
    return tuple(active), audit
