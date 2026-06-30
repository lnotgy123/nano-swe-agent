from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_NAME = "swe_agent_sft"
DEFAULT_RECOVERY_DATASET_NAME = "swe_agent_recovery_sft"


def iter_jsonl(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def trajectory_to_sharegpt(record: dict, assistant_json_actions: bool = False) -> dict:
    messages = record.get("messages", [])
    system = _first_message_content(messages, "system")
    initial_user = _first_message_content(messages, "user")

    conversations = [{"from": "human", "value": initial_user}]

    for step in record.get("steps", []):
        if _is_internal_workflow_step(step):
            result = step.get("result")
            if result is not None:
                conversations.append(
                    {
                        "from": "human",
                        "value": _format_internal_feedback(result, step["tool"]),
                    }
                )
            continue

        action = _format_action(step, assistant_json_actions=assistant_json_actions)

        conversations.append(
            {
                "from": "gpt" if assistant_json_actions else "function_call",
                "value": json.dumps(action, ensure_ascii=False),
            }
        )

        result = step.get("result")
        if result is not None:
            conversations.append(
                {
                    "from": "observation",
                    "value": _format_observation(result, step["tool"]),
                }
            )

    return {
        "conversations": conversations,
        "system": system,
        "tools": _tool_description(),
        "metadata": {
            "run_id": record.get("run_id"),
            "run_name": record.get("run_name"),
            "created_at": record.get("created_at"),
            "tests_passed": record.get("final", {}).get("tests_passed"),
            "finished": record.get("final", {}).get("finished"),
        },
    }


def trajectory_to_recovery_sharegpt(
    record: dict,
    include_edit_targets: bool = False,
    assistant_json_actions: bool = False,
) -> list[dict]:
    messages = record.get("messages", [])
    system = _first_message_content(messages, "system")
    initial_user = _first_message_content(messages, "user")
    steps = record.get("steps", [])

    samples = []
    for index, step in enumerate(steps):
        if step.get("tool") not in {"workflow_guard", "workflow_reflection"}:
            continue
        result = step.get("result") or {}
        feedback = result.get("output", "")
        if not feedback:
            continue

        next_index = _next_external_step_index(steps, index + 1)
        if next_index is None:
            continue
        next_step = steps[next_index]
        if not _is_good_recovery_target(steps, next_index, step, include_edit_targets=include_edit_targets):
            continue

        action = _format_action(next_step, assistant_json_actions=assistant_json_actions)

        samples.append(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": _format_recovery_prompt(
                            initial_user=initial_user,
                            prior_steps=steps[max(0, index - 4):index],
                            workflow_step=step,
                            workflow_feedback=feedback,
                        ),
                    },
                    {
                        "from": "gpt" if assistant_json_actions else "function_call",
                        "value": json.dumps(action, ensure_ascii=False),
                    },
                ],
                "system": system,
                "tools": _tool_description(),
                "metadata": {
                    "run_id": record.get("run_id"),
                    "run_name": record.get("run_name"),
                    "created_at": record.get("created_at"),
                    "source": "workflow_recovery",
                    "workflow_tool": step.get("tool"),
                    "target_tool": next_step.get("tool"),
                },
            }
        )
    return samples


def write_dataset_info(dataset_dir: Path, dataset_name: str, file_name: str) -> Path:
    dataset_info_path = dataset_dir / "dataset_info.json"
    dataset_info = {}
    if dataset_info_path.exists():
        dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))

    dataset_info[dataset_name] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
            "tools": "tools",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "observation_tag": "observation",
            "function_tag": "function_call",
            "system_tag": "system",
        },
    }

    dataset_info_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset_info_path


def convert(
    input_glob: str,
    output_file: Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
    only_solved: bool = True,
    write_info: bool = True,
    require_official_resolved: bool = False,
    mode: str = "full",
    include_recovery_edits: bool = False,
    assistant_json_actions: bool = False,
) -> tuple[int, Path | None]:
    input_paths = sorted(Path(path) for path in glob.glob(input_glob, recursive=True))
    if not input_paths:
        raise FileNotFoundError(f"No trajectory files matched: {input_glob}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_file.open("w", encoding="utf-8") as f:
        for record in iter_jsonl(input_paths):
            if mode == "recovery":
                for sample in trajectory_to_recovery_sharegpt(
                    record,
                    include_edit_targets=include_recovery_edits,
                    assistant_json_actions=assistant_json_actions,
                ):
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    count += 1
            else:
                if require_official_resolved and not record.get("official_evaluation", {}).get("resolved"):
                    continue
                if only_solved and not _is_solved(record):
                    continue
                sample = trajectory_to_sharegpt(
                    record,
                    assistant_json_actions=assistant_json_actions,
                )
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1

    dataset_info_path = None
    if write_info:
        dataset_info_path = write_dataset_info(
            dataset_dir=output_file.parent,
            dataset_name=dataset_name,
            file_name=output_file.name,
        )

    return count, dataset_info_path


def _next_external_step_index(steps: list[dict], start: int) -> int | None:
    for index in range(start, len(steps)):
        if not _is_internal_workflow_step(steps[index]):
            return index
    return None


def _format_action(step: dict, assistant_json_actions: bool = False) -> dict:
    if assistant_json_actions:
        action = {
            "tool": step["tool"],
            "args": step.get("args", {}),
        }
    else:
        action = {
            "name": step["tool"],
            "arguments": step.get("args", {}),
        }
    if step.get("thought"):
        action["thought"] = step["thought"]
    return action


def _is_good_recovery_target(
    steps: list[dict],
    target_index: int,
    workflow_step: dict,
    include_edit_targets: bool = False,
) -> bool:
    target = steps[target_index]
    tool = target.get("tool")
    result = target.get("result") or {}
    if tool not in {"read_file", "find_symbol", "search_text", "replace_text", "replace_lines"}:
        return False
    if not result.get("ok"):
        return False

    rejected_action = workflow_step.get("args", {}).get("rejected_action")
    if isinstance(rejected_action, dict) and _action_signature(rejected_action) == _step_action_signature(target):
        return False

    if tool in {"replace_text", "replace_lines"}:
        if not include_edit_targets:
            return False
        return _edit_passed_automatic_checks(steps, target_index)
    return True


def _edit_passed_automatic_checks(steps: list[dict], edit_index: int) -> bool:
    saw_syntax = False
    saw_review = False
    for step in steps[edit_index + 1: edit_index + 5]:
        tool = step.get("tool")
        result = step.get("result") or {}
        if tool == "syntax_check":
            saw_syntax = True
            if not result.get("ok"):
                return False
        elif tool == "patch_review":
            saw_review = True
            if not result.get("ok"):
                return False
        elif _is_internal_workflow_step(step):
            continue
        elif tool in {"run_shell", "git_diff"}:
            break
        elif tool in {"replace_text", "replace_lines"}:
            break
    return saw_syntax and saw_review


def _action_signature(action: dict) -> str:
    return json.dumps(
        {"tool": action.get("tool"), "args": action.get("args", {})},
        ensure_ascii=False,
        sort_keys=True,
    )


def _step_action_signature(step: dict) -> str:
    return json.dumps(
        {"tool": step.get("tool"), "args": step.get("args", {})},
        ensure_ascii=False,
        sort_keys=True,
    )


def _format_recovery_prompt(
    initial_user: str,
    prior_steps: list[dict],
    workflow_step: dict,
    workflow_feedback: str,
) -> str:
    return (
        "你正在训练 SWE-agent 的 workflow 恢复策略。根据任务、最近工具历史和 workflow 反馈，"
        "输出下一步最合适的一个工具调用。不要重复被拒绝的动作。\n\n"
        f"任务摘要:\n{_truncate_text(initial_user, 6000)}\n\n"
        f"最近工具历史:\n{_format_recent_steps(prior_steps)}\n\n"
        f"workflow 反馈 ({workflow_step.get('tool')}):\n{workflow_feedback}\n\n"
        "请只给出下一步工具调用。"
    )


def _format_recent_steps(steps: list[dict]) -> str:
    chunks = []
    for step in steps:
        result = step.get("result") or {}
        chunks.append(
            "tool={tool}\nargs={args}\nok={ok}\noutput={output}".format(
                tool=step.get("tool"),
                args=json.dumps(step.get("args", {}), ensure_ascii=False),
                ok=result.get("ok"),
                output=_truncate_text(result.get("output", ""), 1200),
            )
        )
    return "\n\n".join(chunks) or "<none>"


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + f"\n... omitted {len(text) - limit} chars ...\n" + text[-limit // 2:]


def _is_solved(record: dict) -> bool:
    official = record.get("official_evaluation")
    if isinstance(official, dict) and "resolved" in official:
        return bool(official.get("resolved"))

    evaluation = record.get("evaluation")
    if isinstance(evaluation, dict) and "resolved" in evaluation:
        return bool(evaluation.get("resolved"))

    final = record.get("final", {})
    return bool(final.get("finished") and final.get("tests_passed"))


def _first_message_content(messages: list[dict], role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            return message.get("content", "")
    return ""


def _format_observation(result: dict, tool_name: str) -> str:
    return (
        f"tool: {tool_name}\n"
        f"ok: {result.get('ok')}\n"
        f"output:\n{result.get('output', '')}"
    )


def _is_internal_workflow_step(step: dict) -> bool:
    return str(step.get("tool", "")).startswith("workflow_")


def _format_internal_feedback(result: dict, tool_name: str) -> str:
    return (
        f"Internal workflow feedback ({tool_name}, ok={result.get('ok')}):\n"
        f"{result.get('output', '')}\n\n"
        "Use this feedback to choose the next valid tool call."
    )


def _tool_description() -> str:
    tools = [
        {
            "name": "list_files",
            "description": "List files under a repository-relative path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Repository-relative path."}},
                "required": [],
            },
        },
        {
            "name": "search_text",
            "description": "Search for exact text in repository files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Exact text to search for."},
                    "path": {"type": "string", "description": "Repository-relative search root."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "find_symbol",
            "description": "Find Python function or class definitions by exact symbol name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Function or class name."},
                    "path": {"type": "string", "description": "Repository-relative search root."},
                },
                "required": ["name"],
            },
        },
        {
            "name": "read_file",
            "description": "Read a repository file, optionally with line bounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository-relative file path."},
                    "start_line": {"type": "integer", "description": "1-based first line."},
                    "end_line": {"type": "integer", "description": "1-based last line."},
                },
                "required": ["path"],
            },
        },
        {
            "name": "replace_text",
            "description": "Replace one exact text span in a repository file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository-relative file path."},
                    "old": {"type": "string", "description": "Exact text to replace."},
                    "new": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old", "new"],
            },
        },
        {
            "name": "replace_lines",
            "description": "Replace an inclusive line range in a repository file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository-relative file path."},
                    "start_line": {"type": "integer", "description": "1-based first line to replace."},
                    "end_line": {"type": "integer", "description": "1-based last line to replace."},
                    "new": {"type": "string", "description": "Replacement lines."},
                },
                "required": ["path", "start_line", "end_line", "new"],
            },
        },
        {
            "name": "syntax_check",
            "description": "Run Python syntax compilation for one repository file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository-relative Python file path."},
                },
                "required": ["path"],
            },
        },
        {
            "name": "patch_review",
            "description": "Review the current git diff for obviously bad or unsafe patch patterns.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "run_shell",
            "description": "Run a shell command in the repository root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds."},
                },
                "required": ["command"],
            },
        },
        {
            "name": "git_diff",
            "description": "Show the current git diff.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "finish",
            "description": "Finish the task after tests and diff have been checked.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string", "description": "Short summary of the fix."}},
                "required": ["summary"],
            },
        },
    ]
    return json.dumps(tools, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SWE-agent trajectories to LLaMA-Factory ShareGPT SFT data."
    )
    parser.add_argument(
        "--input-glob",
        default="data/trajectories/**/*.jsonl",
        help="Glob for trajectory JSONL files.",
    )
    parser.add_argument(
        "--output-file",
        default="data/llamafactory/swe_agent_sft.jsonl",
        help="Output ShareGPT JSONL file.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--mode",
        choices=["full", "recovery"],
        default="full",
        help="full converts whole trajectories; recovery mines workflow recovery decisions only.",
    )
    parser.add_argument(
        "--include-recovery-edits",
        action="store_true",
        help="In recovery mode, also include edit targets that passed syntax_check and patch_review. Disabled by default to avoid learning failed patches.",
    )
    parser.add_argument(
        "--assistant-json-actions",
        action="store_true",
        help="Write target actions as normal assistant JSON messages matching the runtime parser instead of LLaMA-Factory function_call messages.",
    )
    parser.add_argument("--include-unsolved", action="store_true")
    parser.add_argument("--require-official-resolved", action="store_true")
    parser.add_argument("--no-dataset-info", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    if args.mode == "recovery" and dataset_name == DEFAULT_DATASET_NAME:
        dataset_name = DEFAULT_RECOVERY_DATASET_NAME

    count, dataset_info_path = convert(
        input_glob=args.input_glob,
        output_file=Path(args.output_file),
        dataset_name=dataset_name,
        only_solved=not args.include_unsolved,
        write_info=not args.no_dataset_info,
        require_official_resolved=args.require_official_resolved,
        mode=args.mode,
        include_recovery_edits=args.include_recovery_edits,
        assistant_json_actions=args.assistant_json_actions,
    )
    print(f"converted_samples: {count}")
    print(f"output_file: {Path(args.output_file).resolve()}")
    if dataset_info_path is not None:
        print(f"dataset_info: {dataset_info_path}")


if __name__ == "__main__":
    main()
