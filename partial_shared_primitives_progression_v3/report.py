from __future__ import annotations

import json
from pathlib import Path


def write_report(paths: dict[str, Path], decision: dict[str, object]) -> Path:
    criteria = decision["criteria"]
    lines = [
        "# 部分共享动态原语与实验特异连续进程模型 v3：自动报告", "",
        f"最终科学结论：**{decision['status']}**。预注册阈值未通过时，本报告保留失败结果且不重调参数。", "",
        "## 预注册验收", "",
        "| 项目 | 结果 |", "|---|---|",
    ]
    lines.extend(f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in criteria.items())
    lines += ["", "## 解释边界", "", "共享部分仅为因果预测参数和动态原语字典；Exp1/Exp2 的 K、state centre、state-ID、语义和路径均独立。连续进程分数不读取状态输出，且不是全程时间排名、固定五分类或绝对磨损量。", "", "## 数值结果", "", "```json", json.dumps({key: value for key, value in decision.items() if key != 'criteria'}, ensure_ascii=False, indent=2), "```", ""]
    retained = decision.get("retained_failure_runs", [])
    if retained:
        lines += ["## 已保留的实现级失败快照", "", *[f"- `{item}`" for item in retained], ""]
    target = paths["reports"] / "partial_shared_primitives_progression_v3_report.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
