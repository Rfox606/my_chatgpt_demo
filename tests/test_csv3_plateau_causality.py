from dataclasses import replace

import numpy as np
import pandas as pd

from continuous_state_v3.config import ContinuousStateV3Config
from continuous_state_v3.state_engine import PlateauPrior, run_target_state


def _stream(n: int = 320) -> pd.DataFrame:
    cycles = np.arange(n) * 10. + 10.
    value = np.where(cycles <= 500., 0., 2.)
    return pd.DataFrame({"dataset": ["Exp1"] * n, "window_id": range(n), "window_index": range(n), "start_cycle": cycles - 9., "end_cycle": cycles + 9., "center_cycle": cycles, "baseline_window": (cycles <= 500).astype(int), "is_restart_guard": [0] * n, "f1": value, "f2": value * .5})


def _config() -> ContinuousStateV3Config:
    return replace(ContinuousStateV3Config(), plateau_min_cycle=600, plateau_candidate_cycles=30, plateau_lock_cycles=60, plateau_reference_cycles=80, plateau_exit_candidate_cycles=30, plateau_exit_confirm_cycles=60)


def test_plateau_reference_is_locked_from_arrived_windows_only():
    frame = _stream(); prior = PlateauPrior(.01, 100., 100., 100.)
    states, events, _, _, meta = run_target_state(frame, frame, ("f1", "f2"), {"f1": .5, "f2": .5}, prior, None, "A", _config())
    assert meta["plateau_status"] == "LOCKED"
    assert len(events) == 1
    assert states.loc[states.plateau_locked.eq(1), "plateau_lock_cycle"].nunique() == 1


def test_target_severe_direction_is_not_available_before_a_confirmed_exit():
    frame = _stream(); prior = PlateauPrior(.01, 100., 100., 100.)
    states, _, _, updates, _ = run_target_state(frame, frame, ("f1", "f2"), {"f1": .5, "f2": .5}, prior, None, "A", _config())
    assert not states.severe_direction_available.any()
    assert updates.empty


def test_prefix_state_scores_match_full_run_prefix():
    frame = _stream(); prior = PlateauPrior(.01, 100., 100., 100.); config = _config()
    full, *_ = run_target_state(frame, frame, ("f1", "f2"), {"f1": .5, "f2": .5}, prior, None, "A", config)
    prefix, *_ = run_target_state(frame.loc[frame.center_cycle <= 2000], frame, ("f1", "f2"), {"f1": .5, "f2": .5}, prior, None, "A", config)
    merged = prefix.merge(full.loc[full.center_cycle <= 2000, ["window_index", "D_state", "V50_norm", "A_smooth_20"]], on="window_index", suffixes=("_prefix", "_full"))
    assert np.max(np.abs(merged.D_state_prefix - merged.D_state_full)) < 1e-10
    assert np.max(np.abs(merged.V50_norm_prefix - merged.V50_norm_full)) < 1e-10
