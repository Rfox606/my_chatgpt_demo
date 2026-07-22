from continuous_state_v31.evaluation import prefix_causality
from continuous_state_v31.source_prior import SourceProtocolModel
from tests.csv31_test_utils import plateau_prior, run_plateau, state_config


def test_prefix_state_scores_match_full_causal_run() -> None:
    config = state_config()
    (full, *_), source, target, features = run_plateau(config)
    model = SourceProtocolModel("SYNTHETIC", "Source", features, {name: 1.0 for name in features}, plateau_prior(), full, None, float("nan"))
    result = prefix_causality(model, target, full, config)
    assert result["status"] == "PASS"
