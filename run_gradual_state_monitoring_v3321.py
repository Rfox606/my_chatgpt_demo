from __future__ import annotations

import argparse
from dataclasses import replace

from gradual_state_monitoring_v3321.config import GradualStateMonitoringV3321Config
from gradual_state_monitoring_v3321.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated V3.3.2.1 strict-target correction")
    parser.add_argument("--input", default=None); parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = GradualStateMonitoringV3321Config()
    if args.input:
        config = replace(config, input_path=args.input)
    if args.output_dir:
        config = replace(config, output_dir=args.output_dir)
    result = run_pipeline(config)
    print(f"V3.3.2.1 completed: {result['output_dir']}")


if __name__ == "__main__":
    main()
