from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from continuous_state_v31.config import ContinuousStateV31Config, FEATURES
from continuous_state_v31.guards import add_restart_guard
from continuous_state_v31.state_engine import PlateauPrior, run_target_state


def state_config(**changes: object) -> ContinuousStateV31Config:
    base = ContinuousStateV31Config(
        baseline_cycles=500,
        known_stop_interval_cycles=500,
        restart_guard_cycles=100,
        plateau_min_cycle=950,
        plateau_lock_valid_cycles=500,
        plateau_candidate_valid_cycles=300,
        plateau_failure_reset_valid_cycles=150,
        plateau_reference_valid_cycles=500,
        exit_candidate_valid_cycles=300,
        exit_confirm_valid_cycles=500,
        exit_failure_reset_valid_cycles=150,
    )
    return replace(base, **changes)


def stream(dataset: str = "Synthetic", end_cycle: int = 3200, shifted_after: float | None = None) -> pd.DataFrame:
    centres = np.arange(10.5, end_cycle, 5.0)
    data: dict[str, object] = {
        "dataset": dataset,
        "window_id": np.arange(len(centres)),
        "window_index": np.arange(len(centres)),
        "start_cycle": centres - 9.5,
        "end_cycle": centres + 9.5,
        "center_cycle": centres,
        "baseline_window": (centres + 9.5 <= 500).astype(int),
    }
    shift = np.zeros(len(centres))
    if shifted_after is not None:
        shift = np.maximum(0.0, (centres - shifted_after) / 50.0)
    for number, feature in enumerate(FEATURES):
        data[feature] = shift * (1.0 + .05 * number)
    return pd.DataFrame(data)


def plateau_prior() -> PlateauPrior:
    return PlateauPrior(baseline_d_p95=-1.0, v50_threshold=1.0, v100_threshold=1.0, volatility_threshold=1.0, quantile=.75)


def run_plateau(config: ContinuousStateV31Config | None = None, shifted_after: float | None = None, end_cycle: int = 3200):
    config = config or state_config()
    source = add_restart_guard(stream("Source", end_cycle=end_cycle, shifted_after=shifted_after), config)
    target = add_restart_guard(stream("Target", end_cycle=end_cycle, shifted_after=shifted_after), config)
    features = ("rs_corrdist_base", "rs_mean", "rs_q05")
    return run_target_state(target, source, features, {name: 1.0 for name in features}, plateau_prior(), None, "SYNTHETIC", config), source, target, features


def forecast_states(n: int = 320, unavailable_prefix: int = 60) -> pd.DataFrame:
    cycle = np.arange(n, dtype=float) * 5.0 + 10.5
    d = .01 * cycle + .2 * np.sin(cycle / 20.0)
    s = .03 * cycle
    frame = pd.DataFrame({
        "dataset": "Forecast", "window_index": np.arange(n), "center_cycle": cycle, "is_restart_guard": 0,
        "nominal_stride_cycles": 5.0, "D_state": d, "V20_norm": .1 + .01 * np.sin(cycle / 30.0),
        "V50_norm": .12 + .01 * np.cos(cycle / 40.0), "V100_norm": .14, "direction_consistency": .8,
        "A_state": .01, "state_volatility_20": .02, "state_volatility_50": .02, "weighted_oos": .0,
        "plateau_locked": 1, "instability_score": .02 * cycle, "severe_direction_available": 1,
        "S_severe_candidate": s,
    })
    frame.loc[:unavailable_prefix - 1, "severe_direction_available"] = 0
    frame.loc[:unavailable_prefix - 1, "S_severe_candidate"] = np.nan
    return frame
