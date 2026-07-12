import numpy as np

from adaptive_awr_v11.config import AdaptiveAWRV11Config
from adaptive_awr_v11.reliability import ReliabilityController


def test_frozen_noise_does_not_immediately_reduce_but_integrity_can() -> None:
    config = AdaptiveAWRV11Config(reliability_window=10)
    controller = ReliabilityController(["x"], {"x": 1.0}, config)
    noisy = {"x": list(np.linspace(-10, 10, 10))}
    evidence = controller.evidence(noisy)
    controller.immediately_reduce_integrity_only(evidence)
    assert controller.values["x"] == 1.0
    clipped = {"x": [12.0] * 10}
    clipped_evidence = controller.evidence(clipped)
    controller.immediately_reduce_integrity_only(clipped_evidence)
    assert controller.values["x"] < 1.0
    before = controller.values["x"]
    controller.controlled_update({"x": {**clipped_evidence["x"], "raw_reliability": 0.5}})
    assert before - controller.values["x"] <= config.reliability_max_down_step + 1e-12
