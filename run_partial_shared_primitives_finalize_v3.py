from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from partial_shared_primitives_progression_v3.config import PartialSharedPrimitivesConfig
from partial_shared_primitives_progression_v3.report import write_report


def main() -> None:
    config = PartialSharedPrimitivesConfig(); paths = config.paths()
    decision_path = paths["diagnostics"] / "acceptance_decision_v3.json"; decision = json.loads(decision_path.read_text(encoding="utf-8"))
    junit_path = paths["diagnostics"] / "full_pytest_v3.xml"; root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites); failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites); errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    retained_root = Path("retained_failures_partial_shared_primitives_v3")
    retained = sorted(str(path.as_posix()) for path in retained_root.iterdir() if path.is_dir()) if retained_root.exists() else []
    test_status = "PASS" if failures == 0 and errors == 0 else "FAIL"
    (paths["diagnostics"] / "tests_v3.json").write_text(json.dumps({"status": test_status, "tests": tests, "failures": failures, "errors": errors, "junit": str(junit_path), "retained_failure_runs": retained}, ensure_ascii=False, indent=2), encoding="utf-8")
    decision["engineering_test_status"] = test_status; decision["retained_failure_runs"] = retained; decision["overall_status_after_tests"] = decision["status"] if test_status == "PASS" else "FAIL"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = paths["diagnostics"] / "run_manifest_v3.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); manifest["engineering_status"] = test_status; manifest["scientific_status"] = decision["status"]; manifest["full_pytest"] = {"tests": tests, "failures": failures, "errors": errors}; manifest["retained_failure_runs"] = retained
    manifest["output_files"] = [str(path.relative_to(paths["root"])) for path in paths["root"].rglob("*") if path.is_file()]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(paths, decision)


if __name__ == "__main__":
    main()
