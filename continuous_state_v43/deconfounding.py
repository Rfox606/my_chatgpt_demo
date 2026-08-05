from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuousStateV43Config
from .consensus import detect_change_episodes
from .time_mapping import nearest_stop_distance_actual, stop_buffer_mask


def _iou(left: pd.Series, right: pd.Series) -> float:
    begin = max(float(left.start_cycle_actual), float(right.start_cycle_actual))
    end = min(float(left.end_cycle_actual), float(right.end_cycle_actual))
    intersection = max(0.0, end - begin)
    union = max(float(left.end_cycle_actual), float(right.end_cycle_actual)) - min(float(left.start_cycle_actual), float(right.start_cycle_actual))
    return intersection / union if union > 0 else 0.0


def stop_deconfounding(consensus: pd.DataFrame, long: pd.DataFrame, original: pd.DataFrame, config: ContinuousStateV43Config) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Re-detect episodes after fixed actual-cycle stop buffers; never alter state values or thresholds."""
    rows: list[dict[str, object]] = []; variants: dict[int, pd.DataFrame] = {}
    for half_width in config.stop_deconfounding_half_widths_actual:
        kept_parts = []
        for dataset, group in consensus.groupby("dataset", sort=False):
            mask = ~stop_buffer_mask(str(dataset), group.start_cycle_actual.to_numpy(float), group.end_cycle_actual.to_numpy(float), half_width)
            kept_parts.append(group.loc[mask])
        kept = pd.concat(kept_parts, ignore_index=True)
        variant = detect_change_episodes(kept, long, config); variants[int(half_width)] = variant
        for _, before in original.iterrows():
            candidates = variant.loc[variant.protocol_id.eq(before.protocol_id)]
            if candidates.empty:
                matched = None; value = 0.0
            else:
                scores = candidates.apply(lambda after: _iou(before, after), axis=1)
                position = int(scores.to_numpy(float).argmax()); matched = candidates.iloc[position]; value = float(scores.iloc[position])
                if value == 0.0: matched = None
            distance = nearest_stop_distance_actual(str(before.target_dataset), np.asarray([float(before.peak_cycle_actual)]))[0]
            rows.append({
                "row_type": "original_episode_match", "stop_exclusion_half_width_actual": int(half_width), "episode_id": before.episode_id,
                "protocol_id": before.protocol_id, "target_dataset": before.target_dataset,
                "original_start_cycle_effective": before.start_cycle_effective, "original_end_cycle_effective": before.end_cycle_effective,
                "original_peak_cycle_effective": before.peak_cycle_effective, "original_start_cycle_actual": before.start_cycle_actual,
                "original_end_cycle_actual": before.end_cycle_actual, "original_peak_cycle_actual": before.peak_cycle_actual,
                "retained": bool(matched is not None), "interval_iou_actual": value,
                "peak_shift_actual": float(matched.peak_cycle_actual - before.peak_cycle_actual) if matched is not None else np.nan,
                "deconfounded_episode_id": matched.episode_id if matched is not None else None,
                "deconfounded_start_cycle_actual": matched.start_cycle_actual if matched is not None else np.nan,
                "deconfounded_end_cycle_actual": matched.end_cycle_actual if matched is not None else np.nan,
                "deconfounded_peak_cycle_actual": matched.peak_cycle_actual if matched is not None else np.nan,
                "original_peak_nearest_stop_distance_actual": distance,
            })
        original_rows = [row for row in rows if row["stop_exclusion_half_width_actual"] == int(half_width)]
        retained = [row["retained"] for row in original_rows]
        rows.append({
            "row_type": "summary", "stop_exclusion_half_width_actual": int(half_width), "episode_id": "__summary__",
            "protocol_id": "ALL", "target_dataset": "ALL", "retention_rate": float(np.mean(retained)) if retained else np.nan,
            "original_episode_count": int(len(original)), "deconfounded_episode_count": int(len(variant)),
            "mean_interval_iou_actual": float(np.nanmean([row["interval_iou_actual"] for row in original_rows])) if original_rows else np.nan,
            "mean_abs_peak_shift_actual": float(np.nanmean(np.abs([row["peak_shift_actual"] for row in original_rows]))) if any(np.isfinite(row["peak_shift_actual"]) for row in original_rows) else np.nan,
        })
    return pd.DataFrame(rows), variants
