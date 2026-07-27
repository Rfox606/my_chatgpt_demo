from __future__ import annotations

import argparse
from dataclasses import replace

from gradual_state_monitoring_v331.config import GradualStateMonitoringV331Config
from gradual_state_monitoring_v331.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run audited causal gradual-state monitoring V3.3.1")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = GradualStateMonitoringV331Config()
    if args.input:
        config = replace(config, input_path=args.input)
    if args.output_dir:
        config = replace(config, output_dir=args.output_dir)
    result = run_pipeline(config)
    print(f"V3.3.1 completed: {result['output_dir']}")
    print(f"Overall Gate: {result['gate']['overall_status']}")


if __name__ == "__main__":
    main()
