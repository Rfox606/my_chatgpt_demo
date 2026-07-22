from __future__ import annotations

import json
from pathlib import Path


def write_report(paths: dict[str, Path], decision: dict[str, object]) -> None:
    """Write a compact report that preserves both scientific and engineering FAILs."""
    lines = [
        "# 部分共享动态原语与实验特异连续进程模型 v3.1：自动报告",
        "",
        f"最终科学结论：**{decision['status']}**。所有 Gate 使用运行前冻结的阈值；未按结果改参。",
        "",
        "## Gate 判定",
        "",
        "| 方向 | Gate A | Gate B | Gate C | 固定 cycle 因果审计 |",
        "|---|---:|---:|---:|---:|",
    ]
    for direction, value in decision["directions"].items():
        lines.append(f"| {direction} | {value['gate_a']['status']} | {value['gate_b']['status']} | {value['gate_c']['status']} | {value['prefix']['status']} |")
        for gate in ("gate_a", "gate_b", "gate_c"):
            lines.append(f"- `{direction}` / {gate}: `{value[gate]['reason']}`")
    lines += [
        "",
        "## 结果与失败保留",
        "",
        "- Gate A 原始逐预测记录：`results/gate_a_multihorizon_forecasts_v31.csv`；adapter 与负迁移记录：`results/gate_a_adapter_and_negative_transfer_v31.csv`。",
        "- Gate B 的 BOCPD、确认片段、动态原语和目标 private-state CSV 全部保留，包括空表（它们如实表示未达到六段校准条件）。",
        "- Gate C 连续进程、activity、initial prior、uncertainty 和 delayed-entry 收敛 CSV 全部保留。",
        "- 归一化 LMS 修复前的两次工程失败分别保留在 `diagnostics/attempt_001_unstable_online_update.json` 与 `diagnostics/attempt_002_unstable_online_update_terminal.json`；它们不是被覆盖或删除的负结果。",
        "",
        "## 关键决策 JSON",
        "",
        "```json",
        json.dumps(decision, ensure_ascii=False, indent=2),
        "```",
        "",
        "连续进程模块没有接收 state-ID，也没有使用 rolling z。动态原语仅来自 BOCPD 确认的片段级描述符；目标 private-state 从不使用 source state centre 或跨实验 ID 对齐。",
        "",
    ]
    (paths["reports"] / "partial_shared_primitives_progression_v31_report.md").write_text("\n".join(lines), encoding="utf-8")
