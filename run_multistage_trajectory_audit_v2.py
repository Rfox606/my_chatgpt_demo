from __future__ import annotations

import json
from pathlib import Path

from multistage_trajectory_state_v2.audit import (
    change_point_audit,
    local_monotonicity_audit,
    multistage_decision,
    offline_preprocess,
    rolling_ranker_direction_audit,
    trajectory_recurrence_audit,
)
from multistage_trajectory_state_v2.config import MultiStageTrajectoryConfig
from multistage_trajectory_state_v2.data import input_hash, load_windows


def main() -> None:
    config = MultiStageTrajectoryConfig(); paths = config.paths()
    Path(paths["configs"] / "multistage_trajectory_v2_config.json").write_text(json.dumps(config.jsonable(), ensure_ascii=False, indent=2), encoding="utf-8")
    frame = load_windows(config)
    preprocessing = offline_preprocess(frame, config)
    preprocessing.to_csv(paths["results"] / "trajectory_preprocessing_audit_only_v2.csv", index=False)
    local = local_monotonicity_audit(frame, config); local.to_csv(paths["results"] / "local_monotonicity_audit_v2.csv", index=False)
    ranker = rolling_ranker_direction_audit(frame, config); ranker.to_csv(paths["results"] / "rolling_ranker_direction_v2.csv", index=False)
    recurrence = trajectory_recurrence_audit(frame, config, paths["figures"]); recurrence.to_csv(paths["results"] / "trajectory_recurrence_audit_v2.csv", index=False)
    candidates, consensus = change_point_audit(preprocessing, config, paths["figures"])
    candidates.to_csv(paths["results"] / "change_point_candidates_v2.csv", index=False); consensus.to_csv(paths["results"] / "change_point_consensus_v2.csv", index=False)
    decision = multistage_decision(local, ranker, recurrence, consensus)
    decision.update({"input_path": config.input_path, "input_sha256": input_hash(config.input_path), "offline_smoothing_used_for": "Phase A diagnostic preprocessing only", "stage_morphology_debris_read": False})
    Path(paths["diagnostics"] / "multistage_hypothesis_decision_v2.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
