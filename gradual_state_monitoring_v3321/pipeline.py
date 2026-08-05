from __future__ import annotations

from typing import Any

from gradual_state_monitoring_v332 import pipeline as legacy

from .config import GradualStateMonitoringV3321Config
from .forecasting import run_label_free_forecasts


def run_pipeline(config: GradualStateMonitoringV3321Config) -> dict[str, Any]:
    """Run the legacy evaluator with only its origin generator corrected."""
    original = legacy.run_label_free_forecasts
    legacy.run_label_free_forecasts = run_label_free_forecasts
    try:
        return legacy.run_pipeline(config)
    finally:
        legacy.run_label_free_forecasts = original
