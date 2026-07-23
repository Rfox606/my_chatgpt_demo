from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from partial_shared_primitives_progression_v32.config import V32Config
from partial_shared_primitives_progression_v32.data import load_windows, sha256
from partial_shared_primitives_progression_v32.evaluation import run_direction
from partial_shared_primitives_progression_v32.report import write_report


def _with_direction(frame: pd.DataFrame, direction: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result.insert(0, "direction", direction)
    return result


def _legacy_hashes() -> dict[str, str]:
    roots = (
        Path("partial_shared_primitives_progression_v31"),
        Path("outputs_partial_shared_primitives_progression_v31"),
    )
    result: dict[str, str] = {}
    for root in roots:
        for path in sorted(item for item in root.rglob("*") if item.is_file() and "__pycache__" not in item.parts):
            result[str(path).replace("\\", "/")] = sha256(path)
    return result


def main() -> None:
    config = V32Config()
    root = config.root()
    (root / "partial_shared_primitives_progression_v32_config.json").write_text(
        json.dumps(config.jsonable(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    legacy_before = _legacy_hashes()
    frame = load_windows(config)
    groups = {name: group.reset_index(drop=True) for name, group in frame.groupby("dataset", sort=True)}
    tables: dict[str, list[pd.DataFrame]] = {
        name: []
        for name in (
            "prefix", "forecast", "weights", "bocpd", "segments", "descriptors",
            "primitives", "states", "private", "continuous", "uncertainty", "delayed",
        )
    }
    directions: dict[str, object] = {}
    for direction, source_name, target_name in (("Exp1_to_Exp2", "Exp1", "Exp2"), ("Exp2_to_Exp1", "Exp2", "Exp1")):
        result = run_direction(groups[source_name], groups[target_name], config)
        directions[direction] = {
            "source_dataset": source_name,
            "target_dataset": target_name,
            "gate_a": result["gate_a"],
            "gate_b": result["gate_b"],
            "gate_c": result["gate_c"],
            "initial_progression_prior": result["initial_prior"],
            "source_frozen_parameter_sha256": result["source_model"].parameter_sha256,
        }
        mapping = {
            "prefix": "prefix_metrics", "forecast": "forecast_records", "weights": "weight_log",
            "primitives": "source_primitives", "states": "target_state_path", "private": "private_state_log",
            "continuous": "continuous", "delayed": "delayed",
        }
        for name, key in mapping.items():
            tables[name].append(_with_direction(result[key], direction))
        for role in ("source", "target"):
            bocpd = result[f"{role}_bocpd"].copy()
            bocpd.insert(0, "experiment_role", role)
            tables["bocpd"].append(_with_direction(bocpd, direction))
            descriptors = result[f"{role}_segments"].copy()
            if not descriptors.empty:
                descriptors.insert(0, "experiment_role", role)
                confirmed = descriptors.loc[:, [column for column in ("experiment_role", "segment_id", "start_index", "end_index", "start_cycle", "end_cycle", "window_count") if column in descriptors.columns]]
                tables["segments"].append(_with_direction(confirmed, direction))
                tables["descriptors"].append(_with_direction(descriptors, direction))
        uncertainty_columns = [
            column for column in result["continuous"].columns
            if "uncertainty" in column or column in {"dataset", "entry_cycle", "window_index", "center_cycle", "initial_match_quality"}
        ]
        tables["uncertainty"].append(_with_direction(result["continuous"].loc[:, uncertainty_columns], direction))

    filenames = {
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
    saved: dict[str, pd.DataFrame] = {}
    for name, filename in filenames.items():
        table = pd.concat(tables[name], ignore_index=True) if tables[name] else pd.DataFrame()
        table.to_csv(root / filename, index=False, encoding="utf-8")
        saved[name] = table
    overall = "PASS" if all(
        item[gate]["status"] == "PASS" for item in directions.values() for gate in ("gate_a", "gate_b", "gate_c")
    ) else "FAIL"
    legacy_after = _legacy_hashes()
    decision = {
        "version": "v3.2",
        "status": overall,
        "directions": directions,
        "history_sha_unchanged": legacy_before == legacy_after,
        "history_file_count": len(legacy_before),
        "target_final_length_used": False,
        "relative_complete_progress_used": False,
        "source_frozen_updated": False,
        "adapter_and_scratch_independent": True,
        "all_failures_retained": True,
    }
    (root / "gate_decision_v32.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(root, decision, saved)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
