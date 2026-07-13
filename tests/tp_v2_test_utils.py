from __future__ import annotations

import numpy as np
import pandas as pd

from temporal_prototype_v2.config import TemporalPrototypeConfig
from temporal_prototype_v2.data import SourceScaler
from temporal_prototype_v2.model import TemporalPrototypeNet
from temporal_prototype_v2.source import SourceBundle


def config(**changes):
    base = TemporalPrototypeConfig(epochs=1, patience=1, min_memory_to_update=2, update_every_accepted=2, online_steps=1)
    return base.__class__(**{**base.__dict__, **changes})


def source_bundle(cfg: TemporalPrototypeConfig | None = None) -> SourceBundle:
    cfg = cfg or config()
    model = TemporalPrototypeNet(len(cfg.input_features))
    prototypes = np.zeros((5, 16), dtype=float)
    prototypes[:, 0] = np.arange(5)
    scaler = SourceScaler(cfg.input_features, np.zeros(17), np.ones(17), np.full(17, -100), np.full(17, 100))
    return SourceBundle("Source", model, scaler, prototypes, np.ones((5, 16)), np.ones(5), np.ones(5, dtype=int), 100.0, np.eye(16)[0], {})


def target_frame(cfg: TemporalPrototypeConfig | None = None, n: int = 8) -> pd.DataFrame:
    cfg = cfg or config()
    data = {feature: np.linspace(0, 0.1, n) for feature in cfg.input_features}
    data.update({"window_index": np.arange(n), "start_cycle": np.arange(n) * 5 + 1, "end_cycle": np.arange(n) * 5 + 20,
                 "center_cycle": np.arange(n) * 5 + 10.5, "restart_mask": np.zeros(n, dtype=int),
                 "is_restart_guard": np.zeros(n, dtype=int)})
    return pd.DataFrame(data)
