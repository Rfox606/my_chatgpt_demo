from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from partial_shared_primitives_progression_v31.config import V31Config
from partial_shared_primitives_progression_v31.report import write_report


def main() -> None:
    config = V31Config(); paths = config.paths(); root = ET.parse(paths["diagnostics"] / "full_pytest_v31.xml").getroot(); suites = [root] if root.tag == "testsuite" else root.findall("testsuite"); tests = sum(int(item.attrib.get("tests", 0)) for item in suites); failures = sum(int(item.attrib.get("failures", 0)) for item in suites); errors = sum(int(item.attrib.get("errors", 0)) for item in suites)
    decision_path = paths["diagnostics"] / "gate_decision_v31.json"; decision = json.loads(decision_path.read_text(encoding="utf-8")); decision["tests"] = {"tests": tests, "failures": failures, "errors": errors, "status": "PASS" if failures == 0 and errors == 0 else "FAIL"}; decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = paths["diagnostics"] / "run_manifest_v31.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); manifest["engineering_status"] = decision["tests"]["status"]; manifest["tests"] = decision["tests"]; manifest["output_files"] = [str(path.relative_to(paths["root"])) for path in paths["root"].rglob("*") if path.is_file()]; manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(paths, decision)


if __name__ == "__main__": main()

