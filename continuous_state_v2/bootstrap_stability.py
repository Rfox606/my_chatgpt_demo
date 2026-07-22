from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import ContinuousStateV2Config, STABLE_PLUS_FEATURES
from .data import assert_label_free
from .temporal_pairs import build_pair_batch, split_source


def _block_resample(frame: pd.DataFrame, block_size: int, rng: np.random.Generator) -> pd.DataFrame:
    ordered = frame.loc[frame.is_restart_guard.eq(0)].sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    starts = np.arange(0, len(ordered), block_size)
    required = int(np.ceil(len(ordered) / block_size))
    chosen = rng.choice(starts, required, replace=True)
    parts = [ordered.iloc[start:min(start + block_size, len(ordered))] for start in chosen]
    return pd.concat(parts, ignore_index=True).sort_values(["center_cycle", "window_index"]).reset_index(drop=True)


def bootstrap_head(source: pd.DataFrame, features: tuple[str, ...], selected_C: float, direction_id: str, config: ContinuousStateV2Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_label_free(source)
    train, _, _ = split_source(source, config)
    rng = np.random.default_rng(config.random_seed + (1 if direction_id.startswith("Exp1") else 2))
    rows: list[dict[str, object]] = []
    for repeat in range(config.bootstrap_repeats):
        sample = _block_resample(train, config.bootstrap_block_windows, rng)
        batch = build_pair_batch(sample, features, config, config.pair_random_seed + 1000 + repeat)
        model = LogisticRegression(penalty="l2", C=selected_C, fit_intercept=False, solver="liblinear", max_iter=5000, random_state=config.random_seed + repeat).fit(batch.delta_x, batch.labels)
        coef = model.coef_.reshape(-1)
        coef = coef / (np.abs(coef).sum() + config.eps)
        for feature, value in zip(features, coef, strict=True):
            rows.append({"direction_id": direction_id, "bootstrap_repeat": repeat, "feature_name": feature, "normalized_weight": value})
    coefficients = pd.DataFrame(rows)
    stability = coefficients.groupby(["direction_id", "feature_name"], as_index=False).normalized_weight.agg(
        median_weight="median", weight_p05=lambda x: x.quantile(.05), weight_p95=lambda x: x.quantile(.95),
        sign_stability=lambda x: max(float((x > 0).mean()), float((x < 0).mean())),
        selection_frequency=lambda x: float((x.abs() > 1e-10).mean()),
    )
    # Add absent candidate features explicitly, making common-axis exclusion auditable.
    missing = [feature for feature in STABLE_PLUS_FEATURES if feature not in features]
    if missing:
        stability = pd.concat([stability, pd.DataFrame([{"direction_id": direction_id, "feature_name": f, "median_weight": 0., "weight_p05": 0., "weight_p95": 0., "sign_stability": 0., "selection_frequency": 0.} for f in missing])], ignore_index=True)
    return coefficients, stability
