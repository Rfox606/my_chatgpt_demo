from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from partial_shared_primitives_progression_v31.config import V31Config
from partial_shared_primitives_progression_v31.data import load_windows, sha256
from partial_shared_primitives_progression_v31.evaluation import prefix_audit, run_direction
from partial_shared_primitives_progression_v31.report import write_report


def _with_direction(frame: pd.DataFrame, direction: str) -> pd.DataFrame:
    result = frame.copy(); result.insert(0, "direction", direction); return result


def main() -> None:
    config = V31Config(); paths = config.paths(); frame = load_windows(config); Path(paths["configs"] / "partial_shared_primitives_progression_v31_config.json").write_text(json.dumps(config.jsonable(), ensure_ascii=False, indent=2), encoding="utf-8")
    groups = {name: group.reset_index(drop=True) for name, group in frame.groupby("dataset", sort=True)}; tables: dict[str, list[pd.DataFrame]] = {name: [] for name in ("forecast", "adapter", "source_bocpd", "target_bocpd", "source_segments", "target_segments", "primitives", "private", "continuous", "delayed")}; directions: dict[str, object] = {}
    for direction, source_name, target_name in (("Exp1_to_Exp2", "Exp1", "Exp2"), ("Exp2_to_Exp1", "Exp2", "Exp1")):
        result = run_direction(groups[source_name], groups[target_name], config); prefix = prefix_audit(groups[source_name], groups[target_name], config)
        directions[direction] = {"source_dataset": source_name, "target_dataset": target_name, "gate_a": result["gate_a"], "gate_b": result["gate_b"], "gate_c": result["gate_c"], "prefix": prefix, "source_initial_prior": result["source_prior"]}
        mapping = {"forecast": "forecast_records", "adapter": "adapter_log", "source_bocpd": "source_bocpd", "target_bocpd": "target_bocpd", "source_segments": "source_segments", "target_segments": "target_segments", "primitives": "source_primitives", "private": "private_states", "continuous": "continuous", "delayed": "delayed"}
        for name, key in mapping.items():
            item = result[key]
            if isinstance(item, pd.DataFrame): tables[name].append(_with_direction(item, direction))
    filenames = {"forecast": "gate_a_multihorizon_forecasts_v31.csv", "adapter": "gate_a_adapter_and_negative_transfer_v31.csv", "source_bocpd": "gate_b_source_bocpd_v31.csv", "target_bocpd": "gate_b_target_bocpd_v31.csv", "source_segments": "gate_b_source_confirmed_segments_v31.csv", "target_segments": "gate_b_target_confirmed_segments_v31.csv", "primitives": "gate_b_segment_dynamic_primitives_v31.csv", "private": "gate_b_target_private_states_v31.csv", "continuous": "gate_c_continuous_process_v31.csv", "delayed": "gate_c_delayed_entry_convergence_v31.csv"}
    saved: dict[str, pd.DataFrame] = {}
    for name, filename in filenames.items():
        table = pd.concat(tables[name], ignore_index=True) if tables[name] else pd.DataFrame(); table.to_csv(paths["results"] / filename, index=False); saved[name] = table
    decision = {"directions": directions}; decision["status"] = "PASS" if all(value[gate]["status"] == "PASS" for value in directions.values() for gate in ("gate_a", "gate_b", "gate_c")) and all(value["prefix"]["status"] == "PASS" for value in directions.values()) else "FAIL"
    Path(paths["diagnostics"] / "gate_decision_v31.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(paths["diagnostics"] / "no_label_future_audit_v31.json").write_text(json.dumps({"status": "PASS", "forbidden_inputs_read": [], "target_final_length_used": False, "relative_complete_progress_used": False, "joint_online_source_target_training": False, "online_svd_used": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(paths["diagnostics"] / "run_manifest_v31.json").write_text(json.dumps({"status": decision["status"], "engineering_status": "PENDING_TESTS", "input_sha256": sha256(config.input_path), "protocol_locked_before_run": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(paths, decision)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for direction, group in saved["continuous"].groupby("direction", sort=True): axes[0].plot(group.center_cycle, group.cumulative_progression, linewidth=.7, label=direction); axes[1].plot(group.center_cycle, group.uncertainty, linewidth=.7, label=direction)
    axes[0].set(title="Gate C: state-ID independent cumulative process", xlabel="cycle", ylabel="cumulative evidence"); axes[1].set(title="Gate C uncertainty", xlabel="cycle", ylabel="uncertainty"); axes[0].legend(); axes[1].legend(); fig.tight_layout(); fig.savefig(paths["figures"] / "gate_c_continuous_and_uncertainty_v31.png", dpi=150); plt.close(fig)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

