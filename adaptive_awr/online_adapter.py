from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .causal_metrics import BaselineReferences, sigmoid
from .config import AdaptiveAWRConfig


def logit(probability: float, eps: float = 1e-9) -> float:
    probability = float(np.clip(probability, eps, 1.0 - eps))
    return float(np.log(probability / (1.0 - probability)))


@dataclass
class AdapterCheckpoint:
    window_index: int
    reliability: dict[str, float]
    offset: float
    observation_noise: float


class OnlineAdapter:
    """Gated target adapter with immutable feature centres, directions and risk weights."""

    def __init__(
        self,
        config: AdaptiveAWRConfig,
        refs: BaselineReferences,
        features: Sequence[str],
        source_tes_threshold: float,
        source_rs_threshold: float,
        risk_threshold: float,
        enable_reliability: bool,
        enable_offset: bool,
        enforce_freeze_and_rollback: bool,
    ) -> None:
        self.config = config
        self.refs = refs
        self.features = tuple(features)
        self.source_tes_threshold = float(source_tes_threshold)
        self.source_rs_threshold = float(source_rs_threshold)
        self.risk_threshold = float(risk_threshold)
        self.enable_reliability = bool(enable_reliability)
        self.enable_offset = bool(enable_offset)
        self.enforce_freeze_and_rollback = bool(enforce_freeze_and_rollback)
        self.reliability = {feature: 1.0 for feature in self.features}
        self.domain_logit_offset = 0.0
        self.observation_noise = 1.0
        self.freeze_remaining = 0
        self.events: list[dict[str, object]] = []
        self.checkpoint: AdapterCheckpoint | None = None
        self.last_update_window: int | None = None
        self._was_gate_open = False

    def initialize_offset(self, calibration_logits: Sequence[float]) -> None:
        """One calibration-only initialization; no target labels enter this calculation."""
        if not self.enable_offset:
            return
        finite_logits = np.asarray(calibration_logits, dtype=float)
        finite_logits = finite_logits[np.isfinite(finite_logits)]
        if finite_logits.size == 0:
            return
        proposed = logit(self.config.target_safe_risk, self.config.eps) - float(np.nanmedian(finite_logits))
        self.domain_logit_offset = float(
            np.clip(proposed, self.config.online_offset_min, self.config.online_offset_max)
        )

    def weighted_awr(self, feature_values: Mapping[str, float], directions: Mapping[str, int]) -> float:
        weights = np.asarray([self.reliability[feature] for feature in self.features], dtype=float)
        values = np.asarray([feature_values.get(feature, np.nan) for feature in self.features], dtype=float)
        signs = np.asarray([directions[feature] for feature in self.features], dtype=float)
        valid = np.isfinite(values)
        if not valid.any():
            return 0.0
        denominator = float(np.sum(weights[valid]))
        if denominator <= self.config.eps:
            return 0.0
        return float(np.sum(weights[valid] * signs[valid] * values[valid]) / denominator)

    def reliability_evidence(self, feature_history: Mapping[str, Sequence[float]]) -> tuple[dict[str, float], dict[str, float], float]:
        raw: dict[str, float] = {}
        noise_ratios: dict[str, float] = {}
        saturation_rates = []
        for feature in self.features:
            history = np.asarray(feature_history[feature][-self.config.reliability_window :], dtype=float)
            finite_history = history[np.isfinite(history)]
            missing_rate = float(1.0 - finite_history.size / max(len(history), 1))
            if finite_history.size:
                median = float(np.nanmedian(finite_history))
                recent_mad = float(max(np.nanmedian(np.abs(finite_history - median)) * 1.4826, self.config.eps))
                saturation_rate = float(np.mean(np.abs(finite_history) >= self.config.saturation_abs_z))
            else:
                recent_mad = float(self.refs.feature_mad[feature])
                saturation_rate = 0.0
            baseline_mad = float(max(self.refs.feature_mad[feature], self.config.eps))
            noise_ratio = recent_mad / baseline_mad
            raw_value = np.exp(-0.5 * max(noise_ratio - 1.0, 0.0))
            raw_value *= (1.0 - 0.8 * saturation_rate) * (1.0 - missing_rate)
            raw[feature] = float(np.clip(raw_value, self.config.reliability_min, self.config.reliability_max))
            noise_ratios[feature] = float(noise_ratio)
            saturation_rates.append(saturation_rate)
        return raw, noise_ratios, float(np.mean(saturation_rates)) if saturation_rates else 0.0

    def immediately_reduce_for_saturation(self, raw_reliability: Mapping[str, float]) -> None:
        """Saturation can lower trust even while the adapter is otherwise frozen."""
        if not self.enable_reliability:
            return
        for feature in self.features:
            if raw_reliability[feature] < self.reliability[feature]:
                self.reliability[feature] = float(raw_reliability[feature])

    def safety_gate(
        self,
        *,
        slow_risk: float,
        awr: float,
        bd: float,
        rs50: float,
        tes: float,
        recent_tes_event: bool,
        recent_high_risk: bool,
        saturation_feature_rate: float,
    ) -> tuple[bool, list[str]]:
        reasons = []
        if self.freeze_remaining > 0:
            reasons.append("forced_freeze")
        if slow_risk >= 0.25:
            reasons.append("slow_risk")
        if awr >= self.refs.awr_p95:
            reasons.append("awr_calibration_p95")
        if bd >= self.refs.bd_p95:
            reasons.append("bd_calibration_p95")
        if not np.isfinite(rs50) or rs50 > 0.0:
            reasons.append("rs50_positive")
        if tes >= self.source_tes_threshold:
            reasons.append("tes_threshold")
        if recent_tes_event:
            reasons.append("recent_tes_event")
        if recent_high_risk:
            reasons.append("recent_high_risk")
        if saturation_feature_rate >= 0.30:
            reasons.append("saturation_feature_rate")
        return (len(reasons) == 0), reasons

    def force_freeze(self, window_index: int, reason: str, details: Mapping[str, object], duration: int | None = None) -> None:
        duration = int(duration if duration is not None else self.config.event_freeze_windows)
        old_remaining = self.freeze_remaining
        self.freeze_remaining = max(self.freeze_remaining, duration)
        self.events.append(
            {
                "window_index": int(window_index),
                "event_type": "FREEZE",
                "reason": reason,
                "old_freeze_remaining": int(old_remaining),
                "new_freeze_remaining": int(self.freeze_remaining),
                "old_parameters": "",
                "new_parameters": "",
                "details": json.dumps(dict(details), ensure_ascii=False, sort_keys=True),
            }
        )

    def update(
        self,
        *,
        window_index: int,
        raw_reliability: Mapping[str, float],
        noise_ratios: Mapping[str, float],
        gate_open: bool,
        base_logit: float,
    ) -> None:
        if self.freeze_remaining > 0:
            self.freeze_remaining -= 1
        if not (self.enable_reliability or self.enable_offset):
            self._was_gate_open = False
            return
        if not gate_open:
            self._was_gate_open = False
            return
        if self.enable_reliability:
            for feature in self.features:
                old = self.reliability[feature]
                self.reliability[feature] = float(
                    np.clip(
                        (1.0 - self.config.reliability_ewma) * old + self.config.reliability_ewma * raw_reliability[feature],
                        self.config.reliability_min,
                        self.config.reliability_max,
                    )
                )
        self.observation_noise = float(
            0.95 * self.observation_noise + 0.05 * np.nanmedian(list(noise_ratios.values()))
        )
        if self.enable_offset and window_index % self.config.online_update_interval == 0:
            prediction = sigmoid(base_logit + self.domain_logit_offset)
            correction = self.config.online_offset_eta * (logit(self.config.target_safe_risk, self.config.eps) - logit(prediction, self.config.eps))
            self.domain_logit_offset = float(
                np.clip(
                    self.domain_logit_offset + correction,
                    self.config.online_offset_min,
                    self.config.online_offset_max,
                )
            )
        self.last_update_window = int(window_index)
        if not self._was_gate_open:
            self.events.append(
                {
                    "window_index": int(window_index),
                    "event_type": "UPDATE_RESUMED",
                    "reason": "all_safety_conditions_met",
                    "old_freeze_remaining": 0,
                    "new_freeze_remaining": int(self.freeze_remaining),
                    "old_parameters": "",
                    "new_parameters": "",
                    "details": "",
                }
            )
        self._was_gate_open = True

    def save_checkpoint(self, window_index: int) -> None:
        if not self.enforce_freeze_and_rollback or window_index == 0 or window_index % self.config.checkpoint_interval != 0:
            return
        self.checkpoint = AdapterCheckpoint(
            window_index=int(window_index),
            reliability=dict(self.reliability),
            offset=float(self.domain_logit_offset),
            observation_noise=float(self.observation_noise),
        )
        self.events.append(
            {
                "window_index": int(window_index),
                "event_type": "CHECKPOINT",
                "reason": "periodic",
                "old_freeze_remaining": int(self.freeze_remaining),
                "new_freeze_remaining": int(self.freeze_remaining),
                "old_parameters": "",
                "new_parameters": json.dumps(self.parameter_snapshot(), sort_keys=True),
                "details": "",
            }
        )

    def maybe_rollback(
        self,
        window_index: int,
        awr_history: Sequence[float],
        bd_history: Sequence[float],
        slow_risk_history: Sequence[float],
    ) -> bool:
        if not self.enforce_freeze_and_rollback or self.checkpoint is None:
            return False
        if self.last_update_window is None or self.last_update_window <= self.checkpoint.window_index:
            return False
        window = self.config.rollback_eval_windows
        if len(awr_history) < window or len(bd_history) < window or len(slow_risk_history) < window:
            return False
        x = np.arange(window, dtype=float)
        awr_slope = float(np.polyfit(x, np.asarray(awr_history[-window:], dtype=float), 1)[0])
        bd_slope = float(np.polyfit(x, np.asarray(bd_history[-window:], dtype=float), 1)[0])
        slow_drop = float(np.nanmax(np.asarray(slow_risk_history[-window:], dtype=float)) - slow_risk_history[-1])
        if not (awr_slope > 0.0 and bd_slope > 0.0 and slow_drop > self.config.rollback_risk_drop):
            return False
        old = self.parameter_snapshot()
        self.reliability = dict(self.checkpoint.reliability)
        self.domain_logit_offset = float(self.checkpoint.offset)
        self.observation_noise = float(self.checkpoint.observation_noise)
        self.freeze_remaining = max(self.freeze_remaining, self.config.rollback_freeze_windows)
        self.events.append(
            {
                "window_index": int(window_index),
                "event_type": "ROLLBACK",
                "reason": "awr_and_bd_rising_while_slow_risk_suppressed",
                "old_freeze_remaining": 0,
                "new_freeze_remaining": int(self.freeze_remaining),
                "old_parameters": json.dumps(old, sort_keys=True),
                "new_parameters": json.dumps(self.parameter_snapshot(), sort_keys=True),
                "details": json.dumps(
                    {"awr_slope": awr_slope, "bd_slope": bd_slope, "slow_risk_drop": slow_drop}, sort_keys=True
                ),
            }
        )
        return True

    def parameter_snapshot(self) -> dict[str, object]:
        return {
            "domain_logit_offset": float(self.domain_logit_offset),
            "observation_noise": float(self.observation_noise),
            "feature_reliability": {feature: float(value) for feature, value in self.reliability.items()},
        }
