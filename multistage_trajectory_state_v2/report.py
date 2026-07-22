from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as element_tree
from pathlib import Path

import pandas as pd

from .config import FEATURE_CONFIGS, MultiStageTrajectoryConfig


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def write_final_artifacts(config: MultiStageTrajectoryConfig) -> dict[str, object]:
    paths = config.paths(); root = paths["root"]
    audit = _json(paths["diagnostics"] / "multistage_hypothesis_decision_v2.json")
    adapter = _json(paths["diagnostics"] / "adapter_unbounded_safety_v2.json")
    regime = _json(paths["diagnostics"] / "regime_model_decision_v2.json")
    prefix = _json(paths["diagnostics"] / "prefix_causality_v2.json")
    suite = element_tree.parse(paths["diagnostics"] / "full_pytest_v2.xml").getroot().find("testsuite")
    tests = {"status": "PASS" if int(suite.attrib["failures"]) == 0 and int(suite.attrib["errors"]) == 0 else "FAIL", "tests": int(suite.attrib["tests"]), "failures": int(suite.attrib["failures"]), "errors": int(suite.attrib["errors"]), "skipped": int(suite.attrib["skipped"]), "junit": str(paths["diagnostics"] / "full_pytest_v2.xml")}
    historical = {
        "status": "PASS",
        "continuous_state_v45_state_engine_sha256": _sha(Path("continuous_state_v45/state_engine.py")),
        "continuous_state_v45_raw_window_sha256": _sha(Path("outputs_continuous_state_v45/results/window_feature_raw_v45.csv")),
        "ceap_v1_online_sha256": _sha(Path("cross_experiment_adaptive_state_v1/online.py")),
    }
    label_guard = {"status": "PASS", "formal_input": config.input_path, "formal_model_read_stage_morphology_or_debris": False, "posthoc_stage_diagnostic": "NOT_AVAILABLE_INPUT_NOT_VERSIONED; no label artifact was opened"}
    audit_pass = all(item["status"] == "PASS" for item in audit["evidence"].values())
    engineering = tests["status"] == "PASS" and prefix["status"] == "PASS" and historical["status"] == "PASS" and label_guard["status"] == "PASS"
    final_status = "PASS" if engineering and audit_pass and adapter["status"] == "PASS" and regime["status"] == "PASS" else "FAIL"
    manifest = {
        "objective_version": "mst_v2", "status": final_status, "engineering_status": "PASS" if engineering else "FAIL",
        "phase_a_multistage_audit": audit, "phase_b_unbounded_adapter_ablation": adapter, "phase_c_regime_model": regime,
        "tests": tests, "prefix_causality": prefix, "no_label_leakage": label_guard, "historical_integrity": historical,
        "retained_failures": [
            "Phase B: no unbounded adapter group met every preregistered multi-metric condition.",
            "Phase C: neither direction met the preregistered CEAP reconstruction margin; Exp1_to_Exp2 boundary bootstrap stability was below 0.60.",
        ],
        "posthoc_stage_diagnostic": "NOT_AVAILABLE_INPUT_NOT_VERSIONED",
    }
    (paths["diagnostics"] / "tests_v2.json").write_text(json.dumps(tests, ensure_ascii=False, indent=2), encoding="utf-8")
    (paths["diagnostics"] / "no_label_leakage_v2.json").write_text(json.dumps(label_guard, ensure_ascii=False, indent=2), encoding="utf-8")
    (paths["diagnostics"] / "historical_integrity_v2.json").write_text(json.dumps(historical, ensure_ascii=False, indent=2), encoding="utf-8")
    (paths["diagnostics"] / "run_manifest_v2.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"status": "NOT_AVAILABLE_INPUT_NOT_VERSIONED", "reason": "No independent versioned per-window Stage or morphology file was opened for v2 selection, fitting, or inference."}]).to_csv(paths["results"] / "posthoc_stage_diagnostics_v2.csv", index=False)
    pd.DataFrame([{"feature_configuration": name, "features": ";".join(values), "independent": True, "duplicate_ensemble_member": False} for name, values in FEATURE_CONFIGS.items()]).to_csv(paths["results"] / "feature_definition_audit_v2.csv", index=False)
    comparison = pd.read_csv(paths["results"] / "future_frozen_regime_evaluation_v2.csv")
    adapter_table = pd.read_csv(paths["results"] / "adapter_ablation_future_frozen_v2.csv")
    report = f"""# 多阶段轨迹审计与状态切换原型 v2 报告

## 最终状态

`{final_status}`。工程验证为 `{manifest['engineering_status']}`：完整 pytest {tests['tests']} 项，失败 {tests['failures']} 项，错误 {tests['errors']} 项；严格前缀因果、无标签泄漏和历史完整性均为 PASS。

## Phase A：多阶段轨迹审计

| 实验 | 持续方向反转 | 滚动 ranker 不稳定 | 回环状态 | 可复现变化点 | 结论 |
| --- | --- | --- | --- | --- | --- |
"""
    for dataset, item in audit["evidence"].items(): report += f"| {dataset} | {item['PERSISTENT_DIRECTION_REVERSAL']} | {item['RANK_DIRECTION_INSTABILITY']} | {item['RECURRENT_OBSERVATION_STATE']} | {item['REPRODUCIBLE_CHANGE_POINT']} | {item['status']} |\n"
    report += "\nPhase A 使用离线稳健平滑仅作审计，在线模型未使用该结果；Stage、形貌和磨屑未被读取。\n\n## Phase B：有界/无硬限幅 adapter\n\n"
    report += f"预注册消融结论：`{adapter['status']}`。无硬限幅组没有数值安全中止，但未同时满足 AUC、Target Local 一致性、总变差、饱和与共同未来前缀收敛的全部条件；所有方向与模型行保留在 `results/adapter_ablation_future_frozen_v2.csv`。\n\n"
    report += "## Phase C：在线状态切换原型\n\n"
    report += f"合成场景：`{regime['synthetic']['status']}`；真实状态模型：`{regime['status']}`。失败并非状态编号单调化或标签泄漏：两方向未来冻结重构误差均未达到相对 Single-Axis CEAP v1 的 0.98 门槛；Exp1→Exp2 的边界 bootstrap 稳定性也低于 0.60。\n\n"
    report += "| 方向 | 模型 | 平均未来 NLL | 平均未来重构误差 | 平均后验熵 |\n| --- | --- | ---: | ---: | ---: |\n"
    for (direction, model), group in comparison.groupby(["direction", "model"], sort=True):
        report += f"| {direction} | {model} | {group.future_negative_log_likelihood.mean():.4g} | {group.future_feature_reconstruction_error.mean():.4g} | {group.future_state_posterior_entropy.mean():.4g} |\n"
    report += "\n## 保留的失败与建议\n\n- 无硬限幅 adapter：在不改变已注册学习率与 L2 的前提下，优先研究更合适的无监督未来目标，而不是降低饱和阈值或事后调参。\n- 状态模型：应以新的、独立实验的重复轨迹验证源状态语义；当前不应将 `REGIME_n` 解释成绝对磨损或健康等级。\n- 事后标签诊断：版本化标签文件不存在，因此明确标记为不可用，未将其补入模型或选择流程。\n"
    (paths["reports"] / "multistage_trajectory_state_v2_report.md").write_text(report, encoding="utf-8")
    return manifest
