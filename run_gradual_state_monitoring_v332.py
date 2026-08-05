from __future__ import annotations

import argparse
import traceback
from dataclasses import replace

from gradual_state_monitoring_v332.config import GradualStateMonitoringV332Config
from gradual_state_monitoring_v332.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V3.3.2 post-hoc Stage validation and causal nonlinear forecast comparison")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = GradualStateMonitoringV332Config()
    if args.input:
        config = replace(config, input_path=args.input)
    if args.output_dir:
        config = replace(config, output_dir=args.output_dir)
    result = run_pipeline(config)
    print(f"V3.3.2 completed: {result['output_dir']}")
    print(f"Overall Gate: {result['gate']['overall_status']}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
