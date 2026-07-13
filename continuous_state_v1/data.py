from __future__ import annotations

import pandas as pd

from .config import ContinuousStateV1Config


FORBIDDEN_COLUMNS = frozenset({"stage", "stage_label", "Stage1to5"})
WINDOW_COLUMNS = (
    "dataset",
    "window_id",
    "window_index",
    "start_cycle",
    "end_cycle",
    "center_cycle",
    "baseline_window",
)


def assert_label_free(frame: pd.DataFrame) -> None:
    leaked = FORBIDDEN_COLUMNS.intersection(frame.columns)
    if leaked:
        raise AssertionError(
            "Forbidden label column(s) reached a continuous-state function: "
            f"{sorted(leaked)}"
        )


def label_free_copy(frame: pd.DataFrame) -> pd.DataFrame:
    """Make a copy for a model boundary and reject, rather than silently keep, labels."""
    assert_label_free(frame)
    return frame.copy()


def load_window_table(config: ContinuousStateV1Config) -> pd.DataFrame:
    """Pivot the long z-value table without carrying any historical labels forward."""
    long_table = pd.read_csv(config.z_table_path)
    required = set(WINDOW_COLUMNS) | {"feature_name", "z_value"}
    missing = required.difference(long_table.columns)
    if missing:
        raise ValueError(f"Input z table is missing required columns: {sorted(missing)}")

    # WINDOW_COLUMNS deliberately omits the historical label columns present upstream.
    wide = (
        long_table.pivot_table(
            index=list(WINDOW_COLUMNS),
            columns="feature_name",
            values="z_value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    missing_features = set(config.stable_plus_features).difference(wide.columns)
    if missing_features:
        raise ValueError(f"Input z table is missing stable_plus features: {sorted(missing_features)}")
    wide = wide.loc[:, [*WINDOW_COLUMNS, *config.stable_plus_features]].copy()
    wide = wide.sort_values(["dataset", "center_cycle", "window_index"]).reset_index(drop=True)
    assert_label_free(wide)
    if wide[list(config.stable_plus_features)].isna().any().any():
        raise ValueError("stable_plus feature table contains missing values")
    return wide
