from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .causal_metrics import finite, robust_iqr
from .config import AdaptiveAWRV11Config


@dataclass(frozen=True)
class TargetLogitAlignment:
    source_early_median: float
    source_early_iqr: float
    target_calibration_median: float
    target_calibration_iqr: float
    scale: float
    offset: float

    def transform(self, raw_logit: float) -> float:
        return float(self.source_early_median + self.scale * (raw_logit - self.target_calibration_median))

    def row(self, direction_id: str, source_dataset: str, target_dataset: str) -> dict[str, object]:
        return {
            "direction_id": direction_id,
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "source_early_median": self.source_early_median,
            "source_early_IQR": self.source_early_iqr,
            "target_calibration_median": self.target_calibration_median,
            "target_calibration_IQR": self.target_calibration_iqr,
            "alignment_scale": self.scale,
            "alignment_offset": self.offset,
        }


def fit_target_logit_alignment(source_early_logits: np.ndarray, target_calibration_logits: np.ndarray, config: AdaptiveAWRV11Config) -> TargetLogitAlignment:
    source = finite(source_early_logits)
    target = finite(target_calibration_logits)
    source_median = float(np.nanmedian(source)) if source.size else 0.0
    target_median = float(np.nanmedian(target)) if target.size else 0.0
    source_iqr = robust_iqr(source, config.eps)
    target_iqr = robust_iqr(target, config.eps)
    scale = float(np.clip(source_iqr / max(target_iqr, config.eps), 0.5, 2.0))
    offset = float(source_median - scale * target_median)
    return TargetLogitAlignment(source_median, source_iqr, target_median, target_iqr, scale, offset)
