from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from agent.types import AgentRunResult, AgentStep, LLM
from tools.executor import ToolExecutor, ToolResult


VIRTUAL_REPO_ROOT = "/testbed"

SWE_AGENT_SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer to solve tasks.
<IMPORTANT>
* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.
</IMPORTANT>

You have access to the following functions:

---- BEGIN FUNCTION #1: bash ----
Description: Execute a bash command in the terminal.

Parameters:
  (1) command (string, required): The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: submit ----
Description: Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.
No parameters are required for this function.
---- END FUNCTION #2 ----

---- BEGIN FUNCTION #3: str_replace_editor ----
Description: Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`

Parameters:
  (1) command (string, required): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
Allowed values: [`view`, `create`, `str_replace`, `insert`, `undo_edit`]
  (2) path (string, required): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.
  (3) file_text (string, optional): Required parameter of `create` command, with the content of the file to be created.
  (4) old_str (string, optional): Required parameter of `str_replace` command containing the string in `path` to replace.
  (5) new_str (string, optional): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
  (6) insert_line (integer, optional): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
  (7) view_range (array, optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
---- END FUNCTION #3 ----


If you choose to call a function ONLY reply in the following format with NO suffix:

Provide any reasoning for the function call here.
<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Always provide reasoning for your function call in natural language BEFORE the function call (not after)
</IMPORTANT>"""


SWE_AGENT_INSTANCE_PROMPT = """<uploaded_files>
/testbed
</uploaded_files>
I've uploaded a python code repository in the directory /testbed. Consider the following PR description:

<pr_description>
{problem_statement}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <pr_description> is satisfied.
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>
2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
Your thinking should be thorough and so it's fine if it's very long."""


SUBMIT_REVIEW_TEMPLATE = """Thank you for your work on this issue. Please carefully follow the steps below to help review your changes.

1. If you made any changes to your code after running the reproduction script, please run the reproduction script again.
  If the reproduction script is failing, please revisit your changes and make sure they are correct.
  If you have already removed your reproduction script, please ignore this step.
2. Remove your reproduction script (if you haven't done so already).
3. If you have modified any TEST files, please revert them to the state they had before you started fixing the issue.
  You can do this with `git checkout -- /path/to/test/file.py`. Use below <diff> to find the files you need to revert.
4. Run the submit command again to confirm.

Here is a list of all of your changes:

<diff>
{diff}
</diff>"""


@dataclass(frozen=True)
class SWEAgentXMLConfig:
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    max_steps: int = 150
    test_timeout: int = 300
    max_consecutive_format_errors: int = 3
    hard_context_tokens: int = 28_672
    total_execution_timeout: int = 1_800
    max_consecutive_command_timeouts: int = 3
    identical_failure_block_at: int = 3
    identical_failure_stop_at: int = 5
    max_empty_submit_attempts: int = 2


@dataclass(frozen=True)
class XMLFunctionAction:
    name: str
    params: dict[str, object]
    thought: str
    raw: str


class SWEAgentXMLWorkflow:
    """Minimal loop compatible with SWE-smith's XML-converted trajectories."""

    def __init__(self, llm: LLM, tools: ToolExecutor, config: SWEAgentXMLConfig) -> None:
        self.llm = llm
        self.tools = tools
        self.config = config
        self.editor = StrReplaceEditor(tools.repo_root)
        self.submit_stage = 0

    def run(self, problem_statement: str) -> AgentRunResult:
        repo_context = self.tools.list_files(".")
        messages = [
            {"role": "system", "content": SWE_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": SWE_AGENT_INSTANCE_PROMPT.format(problem_statement=problem_statement)},
        ]
        steps: list[AgentStep] = []
        consecutive_format_errors = 0
        consecutive_command_timeouts = 0
        total_execution_time = 0.0
        last_failed_action: str | None = None
        last_failed_error_type: str | None = None
        identical_failure_attempts = 0
        empty_submit_attempts = 0
        started_at = time.time()

        for _ in range(self.config.max_steps):
            context_tokens = _count_tokens(self.llm, messages)
            if context_tokens > self.config.hard_context_tokens:
                return self._autosubmit_on_limit(
                    tool="context_limit",
                    message="Context hard limit reached.",
                    args={"tokens": context_tokens, "hard_limit": self.config.hard_context_tokens},
                    problem_statement=problem_statement,
                    repo_context=repo_context,
                    messages=messages,
                    steps=steps,
                )
            if total_execution_time > self.config.total_execution_timeout > 0:
                return self._autosubmit_on_limit(
                    tool="execution_time_limit",
                    message="Total tool execution time limit reached.",
                    args={
                        "execution_seconds": round(total_execution_time, 3),
                        "limit_seconds": self.config.total_execution_timeout,
                    },
                    problem_statement=problem_statement,
                    repo_context=repo_context,
                    messages=messages,
                    steps=steps,
                )

            raw = self.llm.generate(messages)
            try:
                action = parse_xml_function_action(raw)
            except ValueError as exc:
                consecutive_format_errors += 1
                steps.append(
                    AgentStep(
                        thought="sweagent_xml: format error",
                        tool="format_error",
                        args={"raw_chars": len(raw)},
                        result=ToolResult("format_error", False, str(exc)),
                    )
                )
                messages.append({"role": "assistant", "content": _compact(raw)})
                messages.append({"role": "user", "content": _format_error_message(str(exc))})
                if consecutive_format_errors >= self.config.max_consecutive_format_errors:
                    return _run_result(False, "Stopped after repeated XML function-call errors.", problem_statement, repo_context, messages, steps)
                continue

            consecutive_format_errors = 0
            messages.append({"role": "assistant", "content": raw})
            action_fingerprint = _action_fingerprint(action)
            should_block_repeat = (
                action_fingerprint == last_failed_action
                and last_failed_error_type not in {"timeout", "empty_submit"}
                and identical_failure_attempts >= self.config.identical_failure_block_at - 1
            )
            if should_block_repeat:
                identical_failure_attempts += 1
                finished = False
                result = ToolResult(
                    action.name,
                    False,
                    (
                        f"Blocked identical failed action attempt {identical_failure_attempts}. "
                        "Do not repeat it again. Change the path, parameters, or strategy after inspecting "
                        "the repository with a different command."
                    ),
                    "repeated_action",
                )
            else:
                execution_started = time.monotonic()
                finished, result = self._execute(action)
                total_execution_time += time.monotonic() - execution_started
                if result.ok:
                    last_failed_action = None
                    last_failed_error_type = None
                    identical_failure_attempts = 0
                elif action_fingerprint == last_failed_action:
                    identical_failure_attempts += 1
                    last_failed_error_type = result.error_type
                else:
                    last_failed_action = action_fingerprint
                    last_failed_error_type = result.error_type
                    identical_failure_attempts = 1
            steps.append(
                AgentStep(
                    thought=action.thought,
                    tool=action.name,
                    args=dict(action.params),
                    result=result,
                )
            )

            if result.error_type == "empty_submit":
                empty_submit_attempts += 1
            elif action.name != "submit":
                empty_submit_attempts = 0
            if empty_submit_attempts >= self.config.max_empty_submit_attempts > 0:
                return self._autosubmit_on_limit(
                    tool="empty_submit_limit",
                    message="Stopped after repeated attempts to submit an empty patch.",
                    args={"empty_submit_attempts": empty_submit_attempts},
                    problem_statement=problem_statement,
                    repo_context=repo_context,
                    messages=messages,
                    steps=steps,
                )
            if (
                result.error_type == "repeated_action"
                and identical_failure_attempts >= self.config.identical_failure_stop_at
            ):
                return self._autosubmit_on_limit(
                    tool="repeated_action_limit",
                    message="Stopped after repeated identical failed actions.",
                    args={"identical_failure_attempts": identical_failure_attempts, "action": action.name},
                    problem_statement=problem_statement,
                    repo_context=repo_context,
                    messages=messages,
                    steps=steps,
                )

            if action.name == "bash" and _is_command_timeout(result):
                consecutive_command_timeouts += 1
            else:
                consecutive_command_timeouts = 0
            if consecutive_command_timeouts >= self.config.max_consecutive_command_timeouts > 0:
                return self._autosubmit_on_limit(
                    tool="command_timeout_limit",
                    message="Too many consecutive command timeouts.",
                    args={"consecutive_timeouts": consecutive_command_timeouts},
                    problem_statement=problem_statement,
                    repo_context=repo_context,
                    messages=messages,
                    steps=steps,
                )

            if finished:
                return _run_result(True, result.output, problem_statement, repo_context, messages, steps)

            messages.append({"role": "user", "content": _observation(result)})

        return self._autosubmit_on_limit(
            tool="step_limit",
            message=f"Stopped after reaching max_steps={self.config.max_steps} in {int(time.time() - started_at)}s.",
            args={"max_steps": self.config.max_steps},
            problem_statement=problem_statement,
            repo_context=repo_context,
            messages=messages,
            steps=steps,
        )

    def _autosubmit_on_limit(
        self,
        *,
        tool: str,
        message: str,
        args: dict[str, object],
        problem_statement: str,
        repo_context: ToolResult,
        messages: list[dict[str, str]],
        steps: list[AgentStep],
    ) -> AgentRunResult:
        diff = _current_diff(self.tools)
        has_patch = _has_patch(diff)
        output = diff if has_patch else f"{message} No valid patch was produced."
        steps.append(
            AgentStep(
                thought=f"sweagent_xml: {message}",
                tool=tool,
                args=args,
                result=ToolResult(tool, has_patch, output),
            )
        )
        return _run_result(has_patch, output, problem_statement, repo_context, messages, steps)

    def _execute(self, action: XMLFunctionAction) -> tuple[bool, ToolResult]:
        if action.name == "bash":
            command = str(action.params["command"])
            command_root = str(getattr(self.tools, "command_repo_root", self.tools.repo_root))
            command = command.replace(VIRTUAL_REPO_ROOT, command_root)
            result = self.tools.run_shell(command, timeout=self.config.test_timeout)
            output = _clean_shell_output(result.output).replace(command_root, VIRTUAL_REPO_ROOT)
            return False, ToolResult("bash", result.ok, output, result.error_type)

        if action.name == "str_replace_editor":
            result = self.editor.execute(action.params)
            return False, result

        diff = _current_diff(self.tools)
        if not _has_patch(diff):
            return False, ToolResult(
                "submit",
                False,
                "Submission rejected because the repository diff is empty.",
                "empty_submit",
            )
        if self.submit_stage == 0:
            self.submit_stage = 1
            return False, ToolResult("submit", True, SUBMIT_REVIEW_TEMPLATE.format(diff=diff))
        return True, ToolResult("submit", True, diff)


class StrReplaceEditor:
    """Thin adapter around SWE-agent's official Anthropic editor executable."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        project_root = Path(__file__).resolve().parents[1]
        self.executable = project_root / "SWE-agent" / "tools" / "edit_anthropic" / "bin" / "str_replace_editor"
        self.registry_lib = project_root / "SWE-agent" / "tools" / "registry" / "lib"
        if not self.executable.is_file():
            raise FileNotFoundError(f"Official str_replace_editor not found: {self.executable}")
        self._state_dir = tempfile.TemporaryDirectory(prefix="sweagent-editor-")
        self.state_file = Path(self._state_dir.name) / "registry.json"
        self.history: dict[Path, list[str | None]] = {}

    def execute(self, params: dict[str, object]) -> ToolResult:
        try:
            command = str(params["command"])
            path = self._map_path(str(params["path"]))
            if command == "undo_edit":
                return self._undo(path)
            previous = path.read_text(encoding="utf-8") if path.is_file() else None
            argv = [sys.executable, str(self.executable), command, str(path)]
            for name in ("file_text", "old_str", "new_str", "insert_line"):
                if name in params:
                    argv.extend([f"--{name}", str(params[name])])
            if "view_range" in params:
                view_range = params["view_range"]
                if not isinstance(view_range, list) or len(view_range) != 2:
                    raise ValueError("view_range must be [start_line, end_line]")
                argv.extend(["--view_range", str(view_range[0]), str(view_range[1])])
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{self.registry_lib}{os.pathsep}{env.get('PYTHONPATH', '')}"
            env["SWE_AGENT_ENV_FILE"] = str(self.state_file)
            env["USE_FILEMAP"] = "true"
            proc = subprocess.run(
                argv,
                cwd=self.repo_root,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
            )
            output = "\n".join(part.rstrip() for part in (proc.stdout, proc.stderr) if part.rstrip())
            output = output.replace(str(self.repo_root), VIRTUAL_REPO_ROOT)
            if proc.returncode == 0 and command in {"create", "str_replace", "insert"}:
                self.history.setdefault(path, []).append(previous)
            error_type = None if proc.returncode == 0 else "editor_error"
            return ToolResult("str_replace_editor", proc.returncode == 0, output, error_type)
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or exc.stderr or "Official str_replace_editor timed out after 60 seconds."
            return ToolResult(
                "str_replace_editor",
                False,
                str(output).replace(str(self.repo_root), VIRTUAL_REPO_ROOT),
                "timeout",
            )
        except Exception as exc:
            return ToolResult("str_replace_editor", False, f"{type(exc).__name__}: {exc}", "editor_error")

    def _map_path(self, raw: str) -> Path:
        if raw == VIRTUAL_REPO_ROOT:
            path = self.repo_root
        elif raw.startswith(VIRTUAL_REPO_ROOT + "/"):
            path = self.repo_root / raw[len(VIRTUAL_REPO_ROOT) + 1 :]
        else:
            candidate = Path(raw)
            path = candidate if candidate.is_absolute() else self.repo_root / candidate
        path = path.resolve()
        if path != self.repo_root and self.repo_root not in path.parents:
            raise ValueError(f"Path escapes /testbed: {raw}")
        return path

    def _undo(self, path: Path) -> ToolResult:
        versions = self.history.get(path, [])
        if not versions:
            return ToolResult("str_replace_editor", False, f"No edit history found for {self._virtual(path)}.")
        previous = versions.pop()
        if previous is None:
            path.unlink(missing_ok=True)
            content = ""
        else:
            path.write_text(previous, encoding="utf-8")
            content = previous
        numbered = "\n".join(f"{i:6}\t{line}" for i, line in enumerate(content.split("\n"), start=1))
        output = f"Last edit to {self._virtual(path)} undone successfully."
        if content:
            output += f" Here's the result of running `cat -n` on {self._virtual(path)}:\n{numbered}\n"
        return ToolResult("str_replace_editor", True, output)

    def _virtual(self, path: Path) -> str:
        if path == self.repo_root:
            return VIRTUAL_REPO_ROOT
        return f"{VIRTUAL_REPO_ROOT}/{path.relative_to(self.repo_root)}"


def parse_xml_function_action(raw: str) -> XMLFunctionAction:
    matches = list(re.finditer(r"<function=([^>]+)>\n?(.*?)</function>", raw, re.DOTALL))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one XML function call, found {len(matches)}.")
    match = matches[0]
    name = match.group(1).strip()
    aliases = {"execute_bash": "bash", "finish": "submit"}
    name = aliases.get(name, name)
    if name not in {"bash", "str_replace_editor", "submit"}:
        raise ValueError(f"Unknown function: {name}")
    params: dict[str, object] = {}
    for key, value in re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", match.group(2), re.DOTALL):
        key = key.strip()
        if key in params:
            raise ValueError(f"Duplicate parameter: {key}")
        params[key] = re.sub(r"^\n|\n$", "", value)
    required = {"bash": {"command"}, "str_replace_editor": {"command", "path"}, "submit": set()}[name]
    allowed = {
        "bash": {"command"},
        "str_replace_editor": {"command", "path", "file_text", "old_str", "new_str", "insert_line", "view_range"},
        "submit": set(),
    }[name]
    missing = required - params.keys()
    extra = params.keys() - allowed
    if missing:
        raise ValueError(f"Required parameter(s) missing: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"Unexpected parameter(s): {', '.join(sorted(extra))}")
    if "insert_line" in params:
        params["insert_line"] = int(str(params["insert_line"]))
    if "view_range" in params:
        range_match = re.fullmatch(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", str(params["view_range"]))
        if not range_match:
            raise ValueError("view_range must have the form [start_line, end_line]")
        params["view_range"] = [int(range_match.group(1)), int(range_match.group(2))]
    thought = (raw[: match.start()] + raw[match.end() :]).strip()
    return XMLFunctionAction(name=name, params=params, thought=thought, raw=raw)


def _format_error_message(error: str) -> str:
    return f"""Your action could not be parsed properly: {error}.
Please make sure your output includes a thought and exactly _ONE_ function call.
Do not include extra arguments, and only use bash, str_replace_editor, or submit."""


def _observation(result: ToolResult) -> str:
    output = result.output
    if not output.strip() and result.ok:
        return "Your command ran successfully and did not produce any output."
    if result.error_type == "timeout":
        return (
            "OBSERVATION:\nThe tool timed out before completing. Its process was terminated. "
            "Use a narrower command, inspect a smaller target, or run a focused test before retrying.\n\n"
            + output
        )
    if result.error_type == "command_failed":
        return (
            "OBSERVATION:\nThe shell command exited with a non-zero status. Inspect the error below and "
            "adjust the command or implementation; do not repeat the identical command unchanged.\n\n"
            + output
        )
    if result.error_type == "editor_error":
        return (
            "OBSERVATION:\nThe editor rejected this action. Re-read the relevant file or range, then retry "
            "with an exact path and uniquely matching text.\n\n"
            + output
        )
    if result.error_type == "empty_submit":
        return (
            "OBSERVATION:\nSubmission was rejected because no repository patch exists. Make a source change "
            "before submitting, or stop repeating submit if the task cannot be solved.\n\n"
            + output
        )
    if result.error_type == "repeated_action":
        return "OBSERVATION:\n" + output
    return f"OBSERVATION:\n{output}"


def _clean_shell_output(output: str) -> str:
    match = re.fullmatch(r"exit_code: -?\d+\nstdout:\n(.*?)\nstderr:\n(.*)", output, re.DOTALL)
    if not match:
        return output
    return "\n".join(part.rstrip() for part in match.groups() if part.rstrip())


def _current_diff(tools: ToolExecutor) -> str:
    root = shlex.quote(str(getattr(tools, "command_repo_root", tools.repo_root)))
    result = tools.run_shell(
        f"git -C {root} add -A && git -C {root} diff --cached; "
        f"status=$?; git -C {root} reset >/dev/null; exit $status",
        timeout=30,
    )
    return _clean_shell_output(result.output)


def _has_patch(diff: str) -> bool:
    return "diff --git " in diff and "--- " in diff and "+++ " in diff


def _count_tokens(llm: LLM, messages: list[dict[str, str]]) -> int:
    counter = getattr(llm, "count_tokens", None)
    if callable(counter):
        return int(counter(messages))
    # Test doubles and non-transformers clients may not expose a tokenizer.
    # Three characters per token is deliberately conservative for code.
    return max(1, sum(len(message.get("content", "")) for message in messages) // 3)


def _is_command_timeout(result: ToolResult) -> bool:
    return not result.ok and result.error_type == "timeout"


def _action_fingerprint(action: XMLFunctionAction) -> str:
    return json.dumps(
        {"name": action.name, "params": action.params},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _compact(raw: str) -> str:
    if len(raw) <= 4000:
        return raw
    return raw[:2000] + "\n... model output truncated ...\n" + raw[-2000:]


def _run_result(
    finished: bool,
    summary: str,
    task: str,
    repo_context: ToolResult,
    messages: list[dict[str, str]],
    steps: list[AgentStep],
) -> AgentRunResult:
    return AgentRunResult(finished, summary, task, repo_context, messages, steps)
