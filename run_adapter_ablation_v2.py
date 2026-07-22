from __future__ import annotations

import json
from pathlib import Path

from multistage_trajectory_state_v2.adapter_ablation import run_adapter_ablation
from multistage_trajectory_state_v2.config import MultiStageTrajectoryConfig
from multistage_trajectory_state_v2.data import load_windows


def main() -> None:
    config = MultiStageTrajectoryConfig(); paths = config.paths(); frame = load_windows(config)
    summary, parameter_path, score_paths, decision = run_adapter_ablation(frame, config, paths["figures"])
    summary.to_csv(paths["results"] / "adapter_ablation_future_frozen_v2.csv", index=False)
    parameter_path.to_csv(paths["results"] / "adapter_parameter_path_v2.csv", index=False)
    score_paths.to_csv(paths["results"] / "adapter_future_score_paths_v2.csv", index=False)
    Path(paths["diagnostics"] / "adapter_unbounded_safety_v2.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
