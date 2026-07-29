from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from gradual_state_monitoring_v35.config import GradualStateMonitoringV35Config
from gradual_state_monitoring_v35.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir"); parser.add_argument("--seeds")
    args = parser.parse_args(); config = GradualStateMonitoringV35Config()
    if args.output_dir:
        config = replace(config, output_dir=args.output_dir)
    if args.seeds:
        config = replace(config, random_seeds=tuple(int(value) for value in args.seeds.split(",")))
    result = run_pipeline(config)
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_gradual_state_monitoring_v341.py", "tests/test_gradual_state_monitoring_v35.py"], capture_output=True, text=True)
    Path(config.output_dir, "v35_test_report.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    if tests.returncode:
        raise SystemExit(tests.returncode)
    print(result)


if __name__ == "__main__":
    main()
