from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_swebench_runs import _aggregate, _analyze_record, _read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two SWE-bench batch summaries.")
    parser.add_argument("--base", required=True, help="Base summary JSONL.")
    parser.add_argument("--variant", required=True, help="Variant summary JSONL.")
    parser.add_argument("--base-name", default="base")
    parser.add_argument("--variant-name", default="variant")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_records = [_analyze_record(record) for record in _read_jsonl(Path(args.base))]
    variant_records = [_analyze_record(record) for record in _read_jsonl(Path(args.variant))]
    text = _format_compare(
        base_path=Path(args.base),
        variant_path=Path(args.variant),
        base_name=args.base_name,
        variant_name=args.variant_name,
        base_records=base_records,
        variant_records=variant_records,
    )
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


def _format_compare(
    base_path: Path,
    variant_path: Path,
    base_name: str,
    variant_name: str,
    base_records: list[dict[str, Any]],
    variant_records: list[dict[str, Any]],
) -> str:
    base_agg = _aggregate(base_records)
    variant_agg = _aggregate(variant_records)
    variant_by_index = {item["index"]: item for item in variant_records}

    lines = [
        "# SWE-bench A/B 对比",
        "",
        f"- {base_name}: `{base_path}`",
        f"- {variant_name}: `{variant_path}`",
        "",
        "## 聚合指标",
        "",
        "| 指标 | {base} | {variant} |".format(base=base_name, variant=variant_name),
        "|---|---:|---:|",
    ]
    for key in [
        "total",
        "completed",
        "resolved",
        "has_patch",
        "total_workflow_guards",
        "total_workflow_reflections",
        "total_must_edit_guards",
        "total_different_edit_guards",
        "total_different_pattern_guards",
        "total_syntax_failures",
        "total_patch_review_failures",
    ]:
        lines.append(f"| {key} | {base_agg.get(key, 0)} | {variant_agg.get(key, 0)} |")

    lines.extend(
        [
            "",
            "## 失败类型",
            "",
            "| 类型 | {base} | {variant} |".format(base=base_name, variant=variant_name),
            "|---|---:|---:|",
        ]
    )
    failure_names = sorted(
        set(base_agg["failure_counts"]) | set(variant_agg["failure_counts"])
    )
    for name in failure_names:
        lines.append(
            f"| {name} | {base_agg['failure_counts'].get(name, 0)} | "
            f"{variant_agg['failure_counts'].get(name, 0)} |"
        )

    lines.extend(
        [
            "",
            "## 按实例对比",
            "",
            "| index | instance | base 类型 | LoRA 类型 | patch | edits | guard | reflect | syntax |",
            "|---:|---|---|---|---|---|---|---|---|",
        ]
    )
    for base_item in base_records:
        variant_item = variant_by_index.get(base_item["index"], {})
        lines.append(
            "| {index} | {instance} | {base_type} | {variant_type} | {patch} | "
            "{edits} | {guard} | {reflect} | {syntax} |".format(
                index=base_item["index"],
                instance=base_item["instance_id"],
                base_type=base_item["failure_type"],
                variant_type=variant_item.get("failure_type", "-"),
                patch=_pair(base_item.get("has_patch"), variant_item.get("has_patch")),
                edits=_pair(
                    f"{base_item['successful_edits']}/{base_item['edit_attempts']}",
                    f"{variant_item.get('successful_edits', '-')}/{variant_item.get('edit_attempts', '-')}",
                ),
                guard=_pair(base_item["workflow_guards"], variant_item.get("workflow_guards", "-")),
                reflect=_pair(
                    base_item["workflow_reflections"],
                    variant_item.get("workflow_reflections", "-"),
                ),
                syntax=_pair(base_item["syntax_failures"], variant_item.get("syntax_failures", "-")),
            )
        )
    return "\n".join(lines)


def _pair(base: Any, variant: Any) -> str:
    return f"{base} -> {variant}"


if __name__ == "__main__":
    main()
