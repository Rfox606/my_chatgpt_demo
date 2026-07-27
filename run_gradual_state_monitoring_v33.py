from __future__ import annotations

import argparse
from dataclasses import replace

from gradual_state_monitoring_v33.config import GradualStateMonitoringV33Config
from gradual_state_monitoring_v33.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal gradual-state monitoring V3.3")
    parser.add_argument("--input", default=None, help="CSV containing the six approved V3.3 window features")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-benchmarks", action="store_true", help="Run online monitoring and diagnostics without benchmark refits")
    args = parser.parse_args()
    config = GradualStateMonitoringV33Config()
    if args.input:
        config = replace(config, input_path=args.input)
    if args.output_dir:
        config = replace(config, output_dir=args.output_dir)
    result = run_pipeline(config, run_benchmarks=not args.skip_benchmarks)
    print(f"V3.3 completed: {result['output_dir']}")
    print(f"Overall Gate: {result['gates'].get('overall_status')}")


if __name__ == "__main__":
    main()
