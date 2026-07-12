import unittest

import numpy as np
import pandas as pd

from adaptive_awr.causal_metrics import build_baseline_references
from adaptive_awr.config import AdaptiveAWRConfig
from adaptive_awr.online_adapter import OnlineAdapter


class OnlineAdapterGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AdaptiveAWRConfig()
        calibration = np.linspace(0.0, 1.0, 30)
        feature_frame = pd.DataFrame({feature: calibration for feature in self.config.stable_plus_features})
        refs = build_baseline_references(calibration, calibration, calibration, feature_frame, self.config)
        self.adapter = OnlineAdapter(
            self.config,
            refs,
            self.config.stable_plus_features,
            source_tes_threshold=3.0,
            source_rs_threshold=0.005,
            risk_threshold=0.6,
            enable_reliability=True,
            enable_offset=True,
            enforce_freeze_and_rollback=True,
        )

    def test_gate_opens_only_for_safe_state(self) -> None:
        open_gate, reasons = self.adapter.safety_gate(
            slow_risk=0.1,
            awr=-0.1,
            bd=-0.1,
            rs50=-0.01,
            tes=0.0,
            recent_tes_event=False,
            recent_high_risk=False,
            saturation_feature_rate=0.0,
        )
        self.assertTrue(open_gate)
        self.assertEqual(reasons, [])

    def test_forced_event_freeze_closes_gate(self) -> None:
        self.adapter.force_freeze(5, "TES_event", {"TES": 4.0})
        open_gate, reasons = self.adapter.safety_gate(
            slow_risk=0.1,
            awr=-0.1,
            bd=-0.1,
            rs50=-0.01,
            tes=0.0,
            recent_tes_event=False,
            recent_high_risk=False,
            saturation_feature_rate=0.0,
        )
        self.assertFalse(open_gate)
        self.assertIn("forced_freeze", reasons)


if __name__ == "__main__":
    unittest.main()
