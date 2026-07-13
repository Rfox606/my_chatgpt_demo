from __future__ import annotations

import pandas as pd

from .config import ContinuousStateV2Config, STABLE_PLUS_FEATURES


FORBIDDEN_COLUMNS = frozenset({"stage", "stage_label", "Stage1to5"})
WINDOW_COLUMNS = ("dataset", "window_id", "window_index", "start_cycle", "end_cycle", "center_cycle", "baseline_window")


def assert_label_free(frame: pd.DataFrame) -> None:
    leaked = FORBIDDEN_COLUMNS.intersection(frame.columns)
    if leaked:
        raise AssertionError(f"Forbidden historical label column(s) reached a v2 model boundary: {sorted(leaked)}")


def load_window_table(config: ContinuousStateV2Config) -> pd.DataFrame:
    long = pd.read_csv(config.z_table_path)
    required = set(WINDOW_COLUMNS) | {"feature_name", "z_value"}
    missing = required.difference(long.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    # The pivot key intentionally excludes every historical label field.
    wide = (long.pivot_table(index=list(WINDOW_COLUMNS), columns="feature_name", values="z_value", aggfunc="first")
            .reset_index().rename_axis(columns=None))
    missing_features = set(STABLE_PLUS_FEATURES).difference(wide.columns)
    if missing_features:
        raise ValueError(f"Missing stable_plus features: {sorted(missing_features)}")
    result = wide.loc[:, [*WINDOW_COLUMNS, *STABLE_PLUS_FEATURES]].sort_values(["dataset", "center_cycle", "window_index"]).reset_index(drop=True)
    if result[list(STABLE_PLUS_FEATURES)].isna().any().any():
        raise ValueError("Missing z values in stable_plus feature inputs")
    assert_label_free(result)
    return result


def baseline_non_guard_mask(frame: pd.DataFrame, config: ContinuousStateV2Config) -> pd.Series:
    assert_label_free(frame)
    mask = (frame["end_cycle"] <= config.baseline_cycles) & (frame["is_restart_guard"] == 0)
    if int(mask.sum()) >= 20:
        return mask
    fallback = pd.Series(False, index=frame.index)
    usable = frame.index[frame["is_restart_guard"].eq(0)][: min(100, int((frame["is_restart_guard"] == 0).sum()))]
    fallback.loc[usable] = True
    if not fallback.any():
        raise ValueError("No non-guard windows available for baseline")
    return fallback
