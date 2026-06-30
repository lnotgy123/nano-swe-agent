from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SWE-bench batch summaries and trajectories.")
    parser.add_argument("summary_file", help="Path to a run_swebench_batch JSONL summary file.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of Markdown.")
    args = parser.parse_args()

    records = _read_jsonl(Path(args.summary_file))
    analyses = [_analyze_record(record) for record in records]
    aggregate = _aggregate(analyses)

    if args.json:
        print(json.dumps({"aggregate": aggregate, "records": analyses}, ensure_ascii=False, indent=2))
    else:
        print(_format_markdown(Path(args.summary_file), aggregate, analyses))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    trajectory = _read_trajectory(record.get("trajectory_path"))
    steps = trajectory.get("steps", []) if trajectory else []
    tool_counts = Counter(step.get("tool", "<missing>") for step in steps)
    failed_tools = Counter(
        step.get("tool", "<missing>")
        for step in steps
        if not step.get("result", {}).get("ok", True)
    )

    patch_review_failures = _count_failed_tool(steps, "patch_review")
    syntax_failures = _count_failed_tool(steps, "syntax_check")
    workflow_guards = tool_counts.get("workflow_guard", 0)
    workflow_reflections = tool_counts.get("workflow_reflection", 0)
    must_edit_guards = _count_workflow_guard_required(steps, "source_edit_now")
    different_edit_guards = _count_workflow_guard_required(steps, "different_edit")
    different_pattern_guards = _count_workflow_guard_required(steps, "different_edit_pattern")
    run_shell_failures = _count_failed_tool(steps, "run_shell") + _count_failed_tool(steps, "bash")
    editor_edit_steps = [
        step
        for step in steps
        if step.get("tool") == "str_replace_editor"
        and step.get("args", {}).get("command") in {"create", "str_replace", "insert"}
    ]
    shell_edit_steps = [
        step
        for step in steps
        if step.get("tool") == "run_shell"
        and _looks_like_shell_edit(step.get("args", {}).get("command", ""))
    ]
    edit_attempts = (
        tool_counts.get("replace_text", 0)
        + tool_counts.get("replace_lines", 0)
        + len(shell_edit_steps)
        + len(editor_edit_steps)
    )
    successful_edits = sum(
        1
        for step in steps
        if step.get("tool") in {"replace_text", "replace_lines"}
        and step.get("result", {}).get("ok", False)
    )
    successful_edits += sum(
        1 for step in shell_edit_steps if step.get("result", {}).get("ok", False)
    )
    successful_edits += sum(
        1 for step in editor_edit_steps if step.get("result", {}).get("ok", False)
    )

    failure_type = _failure_type(
        record,
        patch_review_failures,
        syntax_failures,
        workflow_guards,
        workflow_reflections,
        must_edit_guards,
        different_edit_guards,
        different_pattern_guards,
        successful_edits,
    )
    evaluation = trajectory.get("evaluation", {}) if trajectory else {}
    patch = evaluation.get("patch", "")

    return {
        "index": record.get("index"),
        "instance_id": record.get("instance_id"),
        "status": record.get("status"),
        "resolved": record.get("local_resolved"),
        "has_patch": record.get("has_patch"),
        "fail_to_pass_passed": record.get("fail_to_pass_passed"),
        "pass_to_pass_passed": record.get("pass_to_pass_passed"),
        "duration_seconds": record.get("duration_seconds"),
        "summary": record.get("summary"),
        "trajectory_path": record.get("trajectory_path"),
        "steps": len(steps) or record.get("steps"),
        "edit_attempts": edit_attempts,
        "successful_edits": successful_edits,
        "patch_review_failures": patch_review_failures,
        "syntax_failures": syntax_failures,
        "workflow_guards": workflow_guards,
        "workflow_reflections": workflow_reflections,
        "must_edit_guards": must_edit_guards,
        "different_edit_guards": different_edit_guards,
        "different_pattern_guards": different_pattern_guards,
        "run_shell_failures": run_shell_failures,
        "patch_chars": len(patch),
        "failure_type": failure_type,
        "tool_counts": dict(tool_counts),
        "failed_tools": dict(failed_tools),
    }


def _read_trajectory(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def _count_failed_tool(steps: list[dict[str, Any]], tool: str) -> int:
    return sum(
        1
        for step in steps
        if step.get("tool") == tool and not step.get("result", {}).get("ok", True)
    )


def _looks_like_shell_edit(command: str) -> bool:
    lowered = command.lower()
    markers = (
        "sed -i",
        "perl -pi",
        "apply_patch",
        ".write_text(",
        ".write_bytes(",
    )
    if any(marker in lowered for marker in markers):
        return True
    if re.search(r"(^|[;&|]\s*)patch\s", lowered):
        return True
    if re.search(r"\b(cat|printf|echo)\b.*>{1,2}\s*[^\s]+", command, re.DOTALL):
        return "patch.txt" not in lowered
    return bool(re.search(r"\btee\s+(?:-a\s+)?[^\s]+", lowered))


def _count_workflow_guard_required(steps: list[dict[str, Any]], required: str) -> int:
    return sum(
        1
        for step in steps
        if step.get("tool") == "workflow_guard"
        and step.get("args", {}).get("required") == required
    )


def _failure_type(
    record: dict[str, Any],
    patch_review_failures: int,
    syntax_failures: int,
    workflow_guards: int,
    workflow_reflections: int,
    must_edit_guards: int,
    different_edit_guards: int,
    different_pattern_guards: int,
    successful_edits: int,
) -> str:
    if record.get("local_resolved") is True:
        return "resolved"
    if record.get("status") != "completed":
        return f"status:{record.get('status')}"
    summary = str(record.get("summary") or "").lower()
    if "invalid json" in summary:
        return "invalid_json"
    if different_edit_guards:
        return "repeated_same_edit"
    if different_pattern_guards:
        return "repeated_bad_pattern"
    if syntax_failures:
        return "syntax_error"
    if patch_review_failures >= 3:
        return "repeated_bad_patch"
    if must_edit_guards:
        return "ignored_must_edit"
    if workflow_reflections and not successful_edits:
        return "reflection_no_retry_edit"
    if workflow_guards:
        return "workflow_guard_blocked"
    if successful_edits and record.get("has_patch"):
        if record.get("local_resolved") is None:
            return "patch_unevaluated"
        if record.get("pass_to_pass_passed") is False:
            return "regression_or_bad_patch"
        return "patch_failed_tests"
    if successful_edits:
        return "edited_but_no_patch"
    return "no_edit"


def _aggregate(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    failure_counts = Counter(item["failure_type"] for item in analyses)
    return {
        "total": len(analyses),
        "completed": sum(1 for item in analyses if item["status"] == "completed"),
        "resolved": sum(1 for item in analyses if item["resolved"] is True),
        "has_patch": sum(1 for item in analyses if item["has_patch"] is True),
        "fail_to_pass_passed": sum(1 for item in analyses if item["fail_to_pass_passed"] is True),
        "pass_to_pass_passed": sum(1 for item in analyses if item["pass_to_pass_passed"] is True),
        "failure_counts": dict(failure_counts),
        "total_patch_review_failures": sum(item["patch_review_failures"] for item in analyses),
        "total_workflow_guards": sum(item["workflow_guards"] for item in analyses),
        "total_workflow_reflections": sum(item["workflow_reflections"] for item in analyses),
        "total_must_edit_guards": sum(item["must_edit_guards"] for item in analyses),
        "total_different_edit_guards": sum(item["different_edit_guards"] for item in analyses),
        "total_different_pattern_guards": sum(item["different_pattern_guards"] for item in analyses),
        "total_syntax_failures": sum(item["syntax_failures"] for item in analyses),
    }


def _format_markdown(path: Path, aggregate: dict[str, Any], analyses: list[dict[str, Any]]) -> str:
    lines = [
        f"# SWE-bench 批量运行分析",
        "",
        f"- summary: `{path}`",
        f"- 样本数: {aggregate['total']}",
        f"- completed: {aggregate['completed']}",
        f"- resolved: {aggregate['resolved']}",
        f"- has_patch: {aggregate['has_patch']}",
        f"- fail_to_pass_passed: {aggregate['fail_to_pass_passed']}",
        f"- pass_to_pass_passed: {aggregate['pass_to_pass_passed']}",
        f"- patch_review 失败次数: {aggregate['total_patch_review_failures']}",
        f"- workflow_guard 次数: {aggregate['total_workflow_guards']}",
        f"- workflow_reflection 次数: {aggregate['total_workflow_reflections']}",
        f"- must_edit guard 次数: {aggregate['total_must_edit_guards']}",
        f"- different_edit guard 次数: {aggregate['total_different_edit_guards']}",
        f"- different_pattern guard 次数: {aggregate['total_different_pattern_guards']}",
        f"- syntax_check 失败次数: {aggregate['total_syntax_failures']}",
        "",
        "## 失败类型",
    ]
    for name, count in aggregate["failure_counts"].items():
        lines.append(f"- {name}: {count}")

    lines.extend(
        [
            "",
            "## 明细",
            "",
            "| index | instance | resolved | has_patch | F2P | P2P | 类型 | edits | review_fail | guard | reflect | must_edit | diff_edit | diff_pattern | 秒 |",
            "|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in analyses:
        lines.append(
            "| {index} | {instance_id} | {resolved} | {has_patch} | {f2p} | {p2p} | {failure_type} | "
            "{successful_edits}/{edit_attempts} | {patch_review_failures} | {workflow_guards} | "
            "{workflow_reflections} | {must_edit_guards} | {different_edit_guards} | "
            "{different_pattern_guards} | {duration} |".format(
                index=item["index"],
                instance_id=item["instance_id"],
                resolved=_flag(item["resolved"]),
                has_patch=_flag(item["has_patch"]),
                f2p=_flag(item["fail_to_pass_passed"]),
                p2p=_flag(item["pass_to_pass_passed"]),
                failure_type=item["failure_type"],
                successful_edits=item["successful_edits"],
                edit_attempts=item["edit_attempts"],
                patch_review_failures=item["patch_review_failures"],
                workflow_guards=item["workflow_guards"],
                workflow_reflections=item["workflow_reflections"],
                must_edit_guards=item["must_edit_guards"],
                different_edit_guards=item["different_edit_guards"],
                different_pattern_guards=item["different_pattern_guards"],
                duration=item["duration_seconds"],
            )
        )
    return "\n".join(lines)


def _flag(value: Any) -> str:
    if value is True:
        return "Y"
    if value is False:
        return "N"
    return "-"


if __name__ == "__main__":
    main()
