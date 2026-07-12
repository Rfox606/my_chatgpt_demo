import unittest

import numpy as np
import pandas as pd

from adaptive_awr.causal_metrics import build_baseline_references
from adaptive_awr.config import AdaptiveAWRConfig
from adaptive_awr.online_adapter import AdapterCheckpoint, OnlineAdapter


class OnlineAdapterRollbackTests(unittest.TestCase):
    def test_rising_observations_and_suppressed_slow_risk_trigger_rollback(self) -> None:
        config = AdaptiveAWRConfig(rollback_eval_windows=10, rollback_risk_drop=0.15)
        calibration = np.linspace(0.0, 1.0, 30)
        features = pd.DataFrame({feature: calibration for feature in config.stable_plus_features})
        refs = build_baseline_references(calibration, calibration, calibration, features, config)
        adapter = OnlineAdapter(
            config,
            refs,
            config.stable_plus_features,
            source_tes_threshold=3.0,
            source_rs_threshold=0.005,
            risk_threshold=0.6,
            enable_reliability=True,
            enable_offset=True,
            enforce_freeze_and_rollback=True,
        )
        adapter.checkpoint = AdapterCheckpoint(5, {feature: 1.0 for feature in config.stable_plus_features}, 0.0, 1.0)
        adapter.reliability = {feature: 0.3 for feature in config.stable_plus_features}
        adapter.domain_logit_offset = 0.8
        adapter.observation_noise = 4.0
        adapter.last_update_window = 8
        slow = [0.8] * 9 + [0.5]
        rolled_back = adapter.maybe_rollback(20, np.arange(10), np.arange(10), slow)
        self.assertTrue(rolled_back)
        self.assertEqual(adapter.domain_logit_offset, 0.0)
        self.assertEqual(adapter.observation_noise, 1.0)
        self.assertEqual(adapter.freeze_remaining, config.rollback_freeze_windows)
        self.assertEqual(adapter.events[-1]["event_type"], "ROLLBACK")


if __name__ == "__main__":
    unittest.main()
