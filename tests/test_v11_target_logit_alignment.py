import numpy as np

from adaptive_awr_v11.config import AdaptiveAWRV11Config
from adaptive_awr_v11.target_calibration import fit_target_logit_alignment


def test_location_scale_alignment_matches_source_reference() -> None:
    config = AdaptiveAWRV11Config()
    source = np.linspace(-2.0, 2.0, 101)
    target = 10.0 + 2.0 * source
    alignment = fit_target_logit_alignment(source, target, config)
    mapped = np.asarray([alignment.transform(value) for value in target])
    assert abs(np.median(mapped) - np.median(source)) < 1e-10
    assert abs((np.percentile(mapped, 75) - np.percentile(mapped, 25)) - (np.percentile(source, 75) - np.percentile(source, 25))) < 1e-10
