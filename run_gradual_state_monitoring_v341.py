from __future__ import annotations
import argparse, subprocess, sys
from dataclasses import replace
from gradual_state_monitoring_v341.config import GradualStateMonitoringV341Config
from gradual_state_monitoring_v341.pipeline import run_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir"); parser.add_argument("--seeds")
    args = parser.parse_args(); config = GradualStateMonitoringV341Config()
    if args.output_dir: config = replace(config, output_dir=args.output_dir)
    if args.seeds: config = replace(config, random_seeds=tuple(int(x) for x in args.seeds.split(",")))
    result = run_pipeline(config)
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_gradual_state_monitoring_v34.py", "tests/test_gradual_state_monitoring_v341.py"], capture_output=True, text=True)
    from pathlib import Path
    Path(config.output_dir, "v341_test_report.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    if tests.returncode: raise SystemExit(tests.returncode)
    print(result)
if __name__ == "__main__": main()
