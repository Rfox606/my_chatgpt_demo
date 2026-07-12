import unittest

import numpy as np
import pandas as pd

from adaptive_awr.causal_metrics import CausalMetricTracker, build_baseline_references
from adaptive_awr.config import AdaptiveAWRConfig


class CausalityTests(unittest.TestCase):
    def test_future_mutation_does_not_change_previous_metrics(self) -> None:
        config = AdaptiveAWRConfig(reliability_window=8, occupancy_window=12)
        awr = np.linspace(0.0, 2.0, 80)
        bd = np.linspace(0.1, 0.6, 80)
        shape = np.linspace(0.05, 0.3, 80)
        features = pd.DataFrame({feature: awr[:20] for feature in config.stable_plus_features})
        refs = build_baseline_references(awr[:20], bd[:20], shape[:20], features, config)

        def run(values: np.ndarray) -> list[dict[str, float]]:
            tracker = CausalMetricTracker(refs, config, 1.5, 0.5)
            return [tracker.step(values[index], bd[index], shape[index]) for index in range(len(values))]

        original = run(awr)
        changed = awr.copy()
        changed[45:] += 100.0
        perturbed = run(changed)
        for index in range(45):
            for key, value in original[index].items():
                other = perturbed[index][key]
                if np.isnan(value):
                    self.assertTrue(np.isnan(other), msg=f"{key} at {index}")
                else:
                    self.assertEqual(value, other, msg=f"{key} at {index}")


if __name__ == "__main__":
    unittest.main()
