from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _fmt(value: object) -> str:
    return f"{float(value):.6g}" if isinstance(value, (float, int)) else str(value)


def write_report(root: Path, decision: dict[str, object], tables: dict[str, pd.DataFrame]) -> None:
    """Write an UTF-8 report that distinguishes transfer from scratch fallback."""
    lines = [
        "# 部分共享动态原语与实验特异连续进程模型 v3.2 修正版",
        "",
        f"总体判定：**{decision['status']}**。所有预注册的 BOCPD hazard、确认阈值和最短片段参数保持不变。",
        "",
        "## Gate 判定",
        "",
        "| 方向 | Gate A | Gate B | Gate C | 迁移结论 |",
        "|---|---:|---:|---:|---|",
    ]
    for direction, result in decision["directions"].items():
        gate_a = result["gate_a"]
        claim = "可以讨论 Adapter 迁移收益" if gate_a["migration_claim_permitted"] else "不可将 Gated Mixture 的 scratch 回退称为迁移成功"
        lines.append(f"| {direction} | {gate_a['status']} | {result['gate_b']['status']} | {result['gate_c']['status']} | {claim} |")
    lines += ["", "## Gate A：冻结前缀后的共同未来预测", ""]
    prefix = tables.get("prefix", pd.DataFrame())
    if not prefix.empty:
        weighted = prefix.loc[prefix.horizon.astype(str).eq("weighted")]
        lines += ["| 方向 | 前缀 | 模型 | 加权 MAE |", "|---|---:|---|---:|"]
        for _, row in weighted.iterrows():
            lines.append(f"| {row.direction} | {float(row.prefix_fraction):.0%} | {row.model} | {_fmt(row.mae)} |")
        lines += ["", "逐 horizon 的 1/5/20-step MAE 见 `gate_a_prefix_prediction_metrics_v32.csv`；Adapter 与 Scratch 分别记录，Gated Mixture 另列输出。"]
    lines += ["", "## Gate B：片段与目标状态", ""]
    for direction, result in decision["directions"].items():
        gate_b = result["gate_b"]
        lines.append(
            f"- `{direction}`：源确认片段 {gate_b['source_confirmed_segments']}，目标确认片段 {gate_b['target_confirmed_segments']}，"
            f"源原语 {gate_b['source_primitive_count']}，目标 private state {gate_b['private_state_count']}。"
        )
        if gate_b["status"] == "FAIL":
            lines.append("  - FAIL 原因：未确认足够多个 BOCPD 片段；因此不把整条实验误称为动态原语。")
    lines += ["", "## Gate C：连续进程与延迟接入", ""]
    for direction, result in decision["directions"].items():
        prior = result["initial_progression_prior"]
        gate_c = result["gate_c"]
        lines.append(
            f"- `{direction}` 初始进程先验 mean={_fmt(prior['initial_progression_prior_mean'])}、"
            f"std={_fmt(prior['initial_progression_prior_std'])}、匹配质量={_fmt(prior['initial_match_quality'])}；"
            f"平台期检查={gate_c['platform_increment_not_higher_than_change_increment']}，短尖峰保护={gate_c['short_spike_increment_guard']}，"
            f"延迟接入检查={gate_c['delayed_entry_ok']}。"
        )
    lines += [
        "",
        "## 审计结论",
        "",
        f"- 历史目录 SHA 保持：`{decision.get('history_sha_unchanged', 'not-audited')}`（{decision.get('history_file_count', 0)} 个文件）。",
        "- BOCPD r=0 使用先验 Student-t predictive，增长分支使用每个 run length 的后验 Student-t predictive；输出中保留两者审计列。",
        "- progression 不读取 cycle、state-ID 或滚动 z 值；activity 可上下波动，而 progression_increment 仅使用持续趋势、持续预测异常、确认转变和持续新颖片段。",
        "- 若 Gate A 失败，报告保留所有数值结果，但不声称历史实验已证明可迁移。",
        "",
        "```json",
        json.dumps(decision, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (root / "partial_shared_primitives_progression_v32_report.md").write_text("\n".join(lines), encoding="utf-8")
