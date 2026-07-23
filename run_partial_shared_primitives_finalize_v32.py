from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from partial_shared_primitives_progression_v32.config import V32Config
from partial_shared_primitives_progression_v32.report import write_report


def main() -> None:
    root = V32Config().root()
    xml_root = ET.parse(root / "full_pytest_v32.xml").getroot()
    suites = [xml_root] if xml_root.tag == "testsuite" else xml_root.findall("testsuite")
    tests = sum(int(item.attrib.get("tests", 0)) for item in suites)
    failures = sum(int(item.attrib.get("failures", 0)) for item in suites)
    errors = sum(int(item.attrib.get("errors", 0)) for item in suites)
    decision_path = root / "gate_decision_v32.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["pytest"] = {
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "status": "PASS" if failures == 0 and errors == 0 else "FAIL",
    }
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    names = {
        "prefix": "gate_a_prefix_prediction_metrics_v32.csv",
        "forecast": "gate_a_online_predictions_v32.csv",
        "weights": "gate_a_model_weight_log_v32.csv",
        "bocpd": "gate_b_bocpd_v32.csv",
        "segments": "gate_b_confirmed_segments_v32.csv",
        "descriptors": "gate_b_segment_descriptors_v32.csv",
        "primitives": "gate_b_source_primitives_v32.csv",
        "states": "gate_b_target_state_path_v32.csv",
        "private": "gate_b_private_state_log_v32.csv",
        "continuous": "gate_c_progression_v32.csv",
        "uncertainty": "gate_c_uncertainty_v32.csv",
        "delayed": "gate_c_delayed_entry_v32.csv",
    }
    tables = {name: pd.read_csv(root / filename) for name, filename in names.items()}
    write_report(root, decision, tables)


if __name__ == "__main__":
    main()
