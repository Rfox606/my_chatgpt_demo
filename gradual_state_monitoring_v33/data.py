from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MAIN_FEATURES = ("rx_mean", "rx_q05", "ry_mean", "ry_q05", "ry_p2p", "rs_rms")
FORBIDDEN_INPUT_NAMES = frozenset({
    "stage", "stage1to5", "state", "state_label", "morphology", "sa", "sq", "sz",
    "wear_debris", "cycle_fraction", "target_final_length",
})


def _validate_columns(columns: list[str]) -> None:
    missing = sorted(set(MAIN_FEATURES).difference(columns))
    if missing:
        raise ValueError(f"Input lacks required V3.3 features: {missing}")
    leaked = sorted(FORBIDDEN_INPUT_NAMES.intersection({name.lower() for name in columns}))
    if leaked:
        raise AssertionError(f"Forbidden online input columns present: {leaked}")


def load_window_data(path: str | Path) -> pd.DataFrame:
    """Load only identifiers, physical cycle and the six approved no-label features."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"V3.3 input does not exist: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    _validate_columns(header)
    id_columns = [name for name in ("dataset", "window_index", "center_cycle") if name in header]
    usecols = [*id_columns, *MAIN_FEATURES]
    frame = pd.read_csv(path, usecols=usecols)
    if "dataset" not in frame:
        frame.insert(0, "dataset", "sequence")
    if "window_index" not in frame:
        frame.insert(1, "window_index", np.arange(len(frame), dtype=int))
    if "center_cycle" not in frame:
        frame["center_cycle"] = frame["window_index"].to_numpy(float)
    if not np.isfinite(frame.loc[:, MAIN_FEATURES].to_numpy(float)).all():
        raise ValueError("V3.3 refuses non-finite feature values rather than using non-causal imputation")
    return frame.sort_values(["dataset", "window_index"], kind="stable").reset_index(drop=True)


def make_synthetic_sequence(kind: str, length: int = 620, seed: int = 3303) -> pd.DataFrame:
    """Six-feature causal-test fixture with no labels used by the detector itself."""
    if kind not in {"stable", "step", "ramp", "spike"}:
        raise ValueError(f"Unknown synthetic sequence kind: {kind}")
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=float)
    base = np.array((0.23, 0.13, -0.036, -0.080, 0.091, 0.242), dtype=float)
    direction = np.array((1.00, 0.75, -0.80, -0.55, 0.42, 0.90), dtype=float)
    scale = np.array((0.012, 0.010, 0.008, 0.008, 0.010, 0.012), dtype=float)
    values = base + rng.normal(0.0, scale, size=(length, len(MAIN_FEATURES)))
    if kind == "step":
        values[t >= 220] += 0.16 * direction
    elif kind == "ramp":
        ramp = np.clip((t - 170.0) / 260.0, 0.0, 1.0)
        values += ramp[:, None] * 0.22 * direction
    elif kind == "spike":
        values[(t >= 250) & (t < 262)] += 0.32 * direction
    frame = pd.DataFrame(values, columns=MAIN_FEATURES)
    frame.insert(0, "dataset", f"synthetic_{kind}")
    frame.insert(1, "window_index", np.arange(length, dtype=int))
    frame.insert(2, "center_cycle", t)
    return frame
