from __future__ import annotations

import json

from multistage_trajectory_state_v2.config import MultiStageTrajectoryConfig
from multistage_trajectory_state_v2.report import write_final_artifacts


if __name__ == "__main__":
    print(json.dumps(write_final_artifacts(MultiStageTrajectoryConfig()), ensure_ascii=False, indent=2))
