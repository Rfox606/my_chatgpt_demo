from __future__ import annotations

import argparse
import traceback
from dataclasses import replace

from gradual_state_monitoring_v34.config import GradualStateMonitoringV34Config
from gradual_state_monitoring_v34.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V3.4 source-weakly-supervised target-label-free monitoring")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", default=None, help="comma-separated fixed seed subset, e.g. 3401,3402")
    args = parser.parse_args()
    config = GradualStateMonitoringV34Config()
    if args.input:
        config = replace(config, input_path=args.input)
    if args.output_dir:
        config = replace(config, output_dir=args.output_dir)
    if args.seeds:
        config = replace(config, random_seeds=tuple(int(value) for value in args.seeds.split(",") if value.strip()))
    result = run_pipeline(config)
    print(f"V3.4 completed: {result['output_dir']}")
    print(f"Frozen target score rows: {result['scores']}; candidate events: {result['events']}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
