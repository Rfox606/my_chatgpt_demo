from __future__ import annotations

import json
from pathlib import Path

from partial_shared_primitives_progression_v3.config import PartialSharedPrimitivesConfig
from partial_shared_primitives_progression_v3.report import write_report


def main() -> None:
    config = PartialSharedPrimitivesConfig(); paths = config.paths()
    decision = json.loads((paths["diagnostics"] / "acceptance_decision_v3.json").read_text(encoding="utf-8"))
    write_report(paths, decision)


if __name__ == "__main__":
    main()

