from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .causal_metrics import robust_mad
from .config import AdaptiveAWRV11Config


@dataclass
class ReliabilityController:
    features: Sequence[str]
    baseline_mad: Mapping[str, float]
    config: AdaptiveAWRV11Config

    def __post_init__(self) -> None:
        self.values = {feature: 1.0 for feature in self.features}

    def evidence(self, histories: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
        evidence: dict[str, dict[str, float]] = {}
        for feature in self.features:
            recent = np.asarray(histories[feature][-self.config.reliability_window :], dtype=float)
            finite = recent[np.isfinite(recent)]
            missing_rate = float(1.0 - len(finite) / max(len(recent), 1))
            clipping_rate = float(np.mean(np.abs(finite) >= self.config.z_clip_abs)) if len(finite) else 0.0
            current_mad = robust_mad(finite, self.config.eps)
            noise_ratio = float(current_mad / max(float(self.baseline_mad[feature]), self.config.eps))
            integrity = float((1.0 - missing_rate) * (1.0 - 0.8 * clipping_rate))
            stability = float(np.exp(-0.25 * max(noise_ratio - 1.0, 0.0)))
            evidence[feature] = {
                "missing_rate": missing_rate,
                "clipping_rate": clipping_rate,
                "noise_ratio": noise_ratio,
                "integrity_reliability": integrity,
                "stability_reliability": stability,
                "raw_reliability": float(np.clip(integrity * stability, self.config.reliability_min, self.config.reliability_max)),
            }
        return evidence

    def immediately_reduce_integrity_only(self, evidence: Mapping[str, Mapping[str, float]]) -> dict[str, bool]:
        """Frozen state may react only to missingness or clipping, never MAD alone."""
        changed = {}
        for feature in self.features:
            item = evidence[feature]
            before = self.values[feature]
            if item["missing_rate"] > 0.0 or item["clipping_rate"] > 0.0:
                integrity_limited = float(np.clip(item["integrity_reliability"], self.config.reliability_min, self.config.reliability_max))
                self.values[feature] = min(before, integrity_limited)
            changed[feature] = self.values[feature] < before
        return changed

    def controlled_update(self, evidence: Mapping[str, Mapping[str, float]]) -> dict[str, tuple[float, float]]:
        changes = {}
        for feature in self.features:
            before = self.values[feature]
            target = float(evidence[feature]["raw_reliability"])
            delta = target - before
            bounded = float(np.clip(delta, -self.config.reliability_max_down_step, self.config.reliability_max_up_step))
            self.values[feature] = float(np.clip(before + bounded, self.config.reliability_min, self.config.reliability_max))
            changes[feature] = (before, self.values[feature])
        return changes

    def weighted_awr(self, values: Mapping[str, float], directions: Mapping[str, int]) -> float:
        feature_values = np.asarray([values[feature] for feature in self.features], dtype=float)
        weights = np.asarray([self.values[feature] for feature in self.features], dtype=float)
        signs = np.asarray([directions[feature] for feature in self.features], dtype=float)
        valid = np.isfinite(feature_values)
        return float(np.sum(weights[valid] * signs[valid] * feature_values[valid]) / max(np.sum(weights[valid]), self.config.eps)) if valid.any() else 0.0
