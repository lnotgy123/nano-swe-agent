from __future__ import annotations

import builtins
import os
import re
import signal
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    output: str
    error_type: str | None = None


class ToolExecutor:
    def __init__(
        self,
        repo_root: str | Path,
        max_output_chars: int = 6000,
        env_path: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.max_output_chars = max_output_chars
        self.env_path = Path(env_path).resolve() if env_path else None
        self._shell_cwd = self.repo_root
        self._shell_env = self._base_shell_env()
        self._shell_state_dir = tempfile.TemporaryDirectory(prefix="sweagent-shell-")

    def run(self, name: str, args: dict) -> ToolResult:
        try:
            if name == "list_files":
                return self.list_files(args.get("path", "."))
            if name == "search_text":
                return self.search_text(args["query"], args.get("path", "."))
            if name == "find_symbol":
                return self.find_symbol(args["name"], args.get("path", "."))
            if name == "read_file":
                return self.read_file(args["path"], args.get("start_line"), args.get("end_line"))
            if name == "replace_text":
                return self.replace_text(args["path"], args["old"], args["new"])
            if name == "replace_lines":
                return self.replace_lines(args["path"], args["start_line"], args["end_line"], args["new"])
            if name == "syntax_check":
                return self.syntax_check(args["path"])
            if name == "patch_review":
                return self.patch_review()
            if name == "run_shell":
                return self.run_shell(args["command"], args.get("timeout", 30))
            if name == "git_diff":
                return self.git_diff()
            return ToolResult(name=name, ok=False, output=f"Unknown tool: {name}")
        except Exception as exc:
            return ToolResult(name=name, ok=False, output=f"{type(exc).__name__}: {exc}")

    def list_files(self, path: str = ".") -> ToolResult:
        root = self._safe_path(path)
        if not root.exists():
            return ToolResult("list_files", False, f"Path does not exist: {path}")
        files = []
        for item in sorted(root.iterdir()):
            if self._skip_path(item):
                continue
            rel = item.relative_to(self.repo_root)
            suffix = "/" if item.is_dir() else ""
            files.append(f"{rel}{suffix}")
            if len(files) >= 300:
                files.append("... truncated ...")
                break
        return ToolResult("list_files", True, self._truncate("\n".join(files)))

    def search_text(self, query: str, path: str = ".") -> ToolResult:
        root = self._safe_path(path)
        matches: list[str] = []
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for file_path in candidates:
            if self._skip_path(file_path) or not file_path.is_file():
                continue
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for index, line in enumerate(lines, start=1):
                if query in line:
                    rel = file_path.relative_to(self.repo_root)
                    matches.append(f"{rel}:{index}: {line}")
                    if len(matches) >= 100:
                        matches.append("... truncated ...")
                        return ToolResult("search_text", True, self._truncate("\n".join(matches)))
        return ToolResult("search_text", True, self._truncate("\n".join(matches) or "No matches."))

    def find_symbol(self, name: str, path: str = ".") -> ToolResult:
        root = self._safe_path(path)
        patterns = [
            f"def {name}(",
            f"async def {name}(",
            f"class {name}(",
            f"class {name}:",
        ]
        matches: list[str] = []
        if root.is_file():
            candidates = [root] if root.suffix == ".py" else []
        else:
            candidates = sorted(root.rglob("*.py"))
        for file_path in candidates:
            if self._skip_path(file_path) or not file_path.is_file():
                continue
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for index, line in enumerate(lines, start=1):
                stripped = line.lstrip()
                if any(stripped.startswith(pattern) for pattern in patterns):
                    rel = file_path.relative_to(self.repo_root)
                    start = max(1, index - 5)
                    end = min(len(lines), index + 40)
                    matches.append(f"{rel}:{index}: {line}\n  suggested_read_file: {rel} start_line={start} end_line={end}")
                    if len(matches) >= 50:
                        matches.append("... truncated ...")
                        return ToolResult("find_symbol", True, self._truncate("\n".join(matches)))
        return ToolResult("find_symbol", True, self._truncate("\n".join(matches) or "No symbol definitions found."))

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> ToolResult:
        file_path = self._safe_path(path)
        if not file_path.is_file():
            return ToolResult("read_file", False, f"Not a file: {path}")
        lines = file_path.read_text(encoding="utf-8").splitlines()
        start = max((start_line or 1), 1)
        end = min((end_line or len(lines)), len(lines))
        selected = lines[start - 1 : end]
        numbered = [f"{line_no}: {line}" for line_no, line in enumerate(selected, start=start)]
        return ToolResult("read_file", True, self._truncate("\n".join(numbered)))

    def replace_text(self, path: str, old: str, new: str) -> ToolResult:
        if old == new:
            return ToolResult("replace_text", False, "Refusing no-op edit: old and new text are identical.")
        file_path = self._safe_path(path)
        if not file_path.is_file():
            return ToolResult("replace_text", False, f"Not a file: {path}")
        rel = file_path.relative_to(self.repo_root)
        if _is_test_path(str(rel)):
            return ToolResult("replace_text", False, f"Refusing to edit test file: {rel}")
        text = file_path.read_text(encoding="utf-8")
        if old not in text:
            return ToolResult("replace_text", False, "Old text was not found exactly once.")
        if text.count(old) != 1:
            line_numbers = _match_line_numbers(text, old)
            return ToolResult(
                "replace_text",
                False,
                (
                    f"Old text matched {text.count(old)} times at line(s): {line_numbers}; refusing ambiguous edit. "
                    "Use replace_lines with the exact start_line and end_line from a recent read_file result."
                ),
            )
        file_path.write_text(text.replace(old, new), encoding="utf-8")
        return ToolResult("replace_text", True, f"Updated {path}")

    def replace_lines(self, path: str, start_line: int, end_line: int, new: str) -> ToolResult:
        file_path = self._safe_path(path)
        if not file_path.is_file():
            return ToolResult("replace_lines", False, f"Not a file: {path}")
        rel = file_path.relative_to(self.repo_root)
        if _is_test_path(str(rel)):
            return ToolResult("replace_lines", False, f"Refusing to edit test file: {rel}")
        if start_line < 1 or end_line < start_line:
            return ToolResult("replace_lines", False, "Invalid line range.")

        text = file_path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        if end_line > len(lines):
            return ToolResult("replace_lines", False, f"Line range exceeds file length: {len(lines)}")

        replacement_lines = [f"{line}\n" for line in new.splitlines()]

        updated = lines[: start_line - 1] + replacement_lines + lines[end_line:]
        if updated == lines:
            return ToolResult("replace_lines", False, "Refusing no-op edit: replacement did not change the file.")
        file_path.write_text("".join(updated), encoding="utf-8")
        return ToolResult("replace_lines", True, f"Updated {path}:{start_line}-{end_line}")

    def run_shell(self, command: str, timeout: int = 30) -> ToolResult:
        state_dir = Path(self._shell_state_dir.name)
        command_file = state_dir / "command.sh"
        cwd_file = state_dir / "cwd"
        env_file = state_dir / "env"
        wrapper_file = state_dir / "run.sh"
        cwd_file.unlink(missing_ok=True)
        env_file.unlink(missing_ok=True)
        command_file.write_text(command, encoding="utf-8")
        wrapper_file.write_text(
            """#!/usr/bin/env bash
__sweagent_save_state() {
  pwd -P > "$SWE_AGENT_CWD_FILE"
  env -0 > "$SWE_AGENT_ENV_STATE_FILE"
}
trap __sweagent_save_state EXIT
source "$SWE_AGENT_COMMAND_FILE"
exit $?
""",
            encoding="utf-8",
        )
        env = dict(self._shell_env)
        env.update(
            {
                "SWE_AGENT_COMMAND_FILE": str(command_file),
                "SWE_AGENT_CWD_FILE": str(cwd_file),
                "SWE_AGENT_ENV_STATE_FILE": str(env_file),
            }
        )
        proc = subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc", str(wrapper_file)],
            cwd=self._shell_cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            output = "\n".join(
                part
                for part in [
                    f"timeout_seconds: {timeout}",
                    "stdout:",
                    _as_text(exc.stdout) or stdout or "",
                    "stderr:",
                    _as_text(exc.stderr) or stderr or "",
                    f"Command timed out: {command}",
                ]
                if part is not None
            )
            return ToolResult("run_shell", False, self._truncate(output), "timeout")
        self._restore_shell_state(cwd_file, env_file)
        output = "\n".join(
            part
            for part in [
                f"exit_code: {proc.returncode}",
                "stdout:",
                stdout,
                "stderr:",
                stderr,
            ]
            if part is not None
        )
        error_type = None if proc.returncode == 0 else "command_failed"
        return ToolResult("run_shell", proc.returncode == 0, self._truncate(output), error_type)

    def git_diff(self) -> ToolResult:
        result = self.run_shell(f"git -C {shlex.quote(str(self.repo_root))} diff -- .", timeout=30)
        return ToolResult("git_diff", result.ok, result.output)

    def syntax_check(self, path: str) -> ToolResult:
        file_path = self._safe_path(path)
        if not file_path.is_file():
            return ToolResult("syntax_check", False, f"Not a file: {path}")
        result = self.run_shell(f"python -m py_compile {shlex.quote(str(file_path))}", timeout=30)
        return ToolResult("syntax_check", result.ok, result.output)

    def patch_review(self) -> ToolResult:
        diff = self.git_diff().output
        issues = _review_diff(diff)
        issues.extend(_review_repo_diff(diff, self.repo_root))
        if issues:
            return ToolResult(
                "patch_review",
                False,
                "Patch review failed:\n" + "\n".join(f"- {issue}" for issue in issues) + "\n\nDiff:\n" + diff,
            )
        return ToolResult("patch_review", True, "Patch review passed.\n\nDiff:\n" + diff)

    def _safe_path(self, path: str) -> Path:
        resolved = (self.repo_root / path).resolve()
        if resolved != self.repo_root and self.repo_root not in resolved.parents:
            raise ValueError(f"Path escapes repo root: {path}")
        return resolved

    def _skip_path(self, path: Path) -> bool:
        parts = set(path.parts)
        return bool(parts & {".git", "__pycache__", ".pytest_cache", ".mypy_cache"})

    def _truncate(self, output: str) -> str:
        if len(output) <= self.max_output_chars:
            return output
        marker = "\n... output clipped; middle omitted ...\n"
        available = max(self.max_output_chars - len(marker), 2)
        head_chars = available // 2
        tail_chars = available - head_chars
        return output[:head_chars] + marker + output[-tail_chars:]

    def _base_shell_env(self) -> dict[str, str]:
        env = os.environ.copy()
        python_bin = str((self.env_path / "bin").resolve()) if self.env_path else str(Path(sys.executable).resolve().parent)
        env["PATH"] = f"{python_bin}{os.pathsep}{env.get('PATH', '')}"
        return env

    def _restore_shell_state(self, cwd_file: Path, env_file: Path) -> None:
        if cwd_file.is_file():
            candidate = Path(cwd_file.read_text(encoding="utf-8").strip())
            if candidate.is_dir():
                self._shell_cwd = candidate
        if not env_file.is_file():
            return
        internal = {"SWE_AGENT_COMMAND_FILE", "SWE_AGENT_CWD_FILE", "SWE_AGENT_ENV_STATE_FILE"}
        entries = env_file.read_bytes().split(b"\0")
        restored: dict[str, str] = {}
        for entry in entries:
            if not entry or b"=" not in entry:
                continue
            raw_key, raw_value = entry.split(b"=", 1)
            key = os.fsdecode(raw_key)
            if key not in internal:
                restored[key] = os.fsdecode(raw_value)
        self._shell_env = restored


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _match_line_numbers(text: str, needle: str) -> list[int]:
    line_numbers = []
    offset = 0
    while True:
        index = text.find(needle, offset)
        if index < 0:
            break
        line_numbers.append(text.count("\n", 0, index) + 1)
        offset = index + max(len(needle), 1)
    return line_numbers


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    parts = lowered.split("/")
    filename = parts[-1]
    return "tests" in parts or "test" in parts or filename.startswith("test_") or filename.endswith("_test.py")


def _review_diff(diff: str) -> list[str]:
    if not diff.strip():
        return ["Patch is empty."]

    issues: list[str] = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[2].removeprefix("a/")
                if _is_test_path(path):
                    issues.append(f"Patch edits test file: {path}")

    added = [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    removed = [line[1:] for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    added_text = "\n".join(added)
    removed_text = "\n".join(removed)

    suspicious_constants = [
        "np.array([True] *",
        "np.array([False] *",
        "return True",
        "return False",
        "return np.ones",
        "return np.zeros",
    ]
    for pattern in suspicious_constants:
        if pattern in added_text:
            issues.append(f"Suspicious constant-return style edit: {pattern}")
    issues.extend(_suspicious_boolean_condition_constants(added, removed_text))

    suspicious_hardcoded_values = [
        "settings.LOCAL_FILE_DIR",
        "example_dir",
        "example_file",
    ]
    for pattern in suspicious_hardcoded_values:
        if pattern in added_text:
            issues.append(f"Suspicious hard-coded test/local value in source patch: {pattern}")

    issues.extend(_removed_definitions(added, removed))
    issues.extend(_assignment_replaced_by_self_mutation(added, removed))

    if "separable_matrix = _separable(transform)" in removed_text and "separable_matrix = _separable(transform)" not in added_text:
        issues.append("Removed the call from is_separable/separability_matrix to _separable.")

    if "return _operators[transform.op](sepleft, sepright)" in removed_text and "return _operators[transform.op](sepleft, sepright)" not in added_text:
        issues.append("Removed CompoundModel operator recursion.")

    if "\nelse:" in added_text and "elif " in removed_text:
        issues.append("Changed an elif branch to else; verify this does not broaden behavior incorrectly.")

    issues.extend(_suspicious_unconditional_argument_calls(added, removed_text))
    issues.extend(_duplicated_context_lines(diff, added))
    issues.extend(_callable_consumed_on_public_attribute_assignment(added, removed_text))

    return issues


def _removed_definitions(added_lines: list[str], removed_lines: list[str]) -> list[str]:
    added_defs = {_definition_signature(line) for line in added_lines}
    added_defs.discard(None)

    issues = []
    for line in removed_lines:
        signature = _definition_signature(line)
        if signature is None or signature in added_defs:
            continue
        issues.append(f"Patch removes a Python definition: {signature}")
    return list(dict.fromkeys(issues))


def _suspicious_boolean_condition_constants(added_lines: list[str], removed_text: str) -> list[str]:
    issues: list[str] = []
    for line in added_lines:
        stripped = line.strip()
        if not stripped.startswith(("if ", "elif ", "while ", "return ")):
            continue
        if not re.search(r"\b(and\s+True|or\s+False)\b", stripped):
            continue
        comparable = stripped.replace(" and True", "").replace(" or False", "")
        if comparable in removed_text:
            issues.append(
                "Suspicious boolean-condition constant edit: added 'and True' or 'or False' to existing logic."
            )
        else:
            issues.append(
                "Suspicious boolean-condition constant edit: do not force a branch with 'and True' or 'or False'."
            )
    return list(dict.fromkeys(issues))


def _definition_signature(line: str) -> str | None:
    stripped = line.strip()
    match = re.match(r"(?:async\s+def|def|class)\s+([A-Za-z_]\w*)\b", stripped)
    if match is None:
        return None
    return stripped.split(":", 1)[0]


def _assignment_replaced_by_self_mutation(added_lines: list[str], removed_lines: list[str]) -> list[str]:
    removed_assignments = set()
    for line in removed_lines:
        match = re.match(r"\s*self\.([A-Za-z_]\w*)\s*=", line)
        if match:
            removed_assignments.add(match.group(1))

    issues = []
    for line in added_lines:
        match = re.match(r"\s*self\.([A-Za-z_]\w*)\.(append|extend|update|add)\s*\(", line)
        if not match:
            continue
        attr = match.group(1)
        method = match.group(2)
        if attr in removed_assignments:
            issues.append(
                f"Replaced initialization assignment self.{attr} = ... with self.{attr}.{method}(...). "
                "Do not mutate a self attribute before preserving its initialization."
            )
    return list(dict.fromkeys(issues))


def _suspicious_unconditional_argument_calls(added_lines: list[str], removed_text: str) -> list[str]:
    issues: list[str] = []
    added_text = "\n".join(added_lines)
    ignored_call_names = {
        "bool",
        "bytes",
        "dict",
        "float",
        "int",
        "len",
        "list",
        "set",
        "str",
        "super",
        "tuple",
        "type",
    }
    for line in added_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "callable(" in stripped:
            continue
        for name in re.findall(r"\b([A-Za-z_]\w*)\s*\(\s*\)", stripped):
            if name in ignored_call_names:
                continue
            if not re.search(rf"\b{name}\b", removed_text):
                continue
            if re.search(rf"\b{name}\s*\(", removed_text):
                continue
            if re.search(rf"callable\s*\(\s*{name}\s*\)", added_text):
                continue
            issues.append(
                f"Suspicious unconditional call of existing value: {name}(). "
                f"Preserve non-callable {name} values or guard with callable({name})."
            )
    return list(dict.fromkeys(issues))


def _duplicated_context_lines(diff: str, added_lines: list[str]) -> list[str]:
    context_lines = []
    for line in diff.splitlines():
        if line.startswith(" ") and line.strip():
            context_lines.append(line[1:].strip())
    context_set = set(context_lines)

    issues = []
    for line in added_lines:
        stripped = line.strip()
        if not _is_meaningful_duplicate_line(stripped):
            continue
        if stripped in context_set:
            issues.append(f"Added line duplicates nearby existing code: {stripped}")
    return list(dict.fromkeys(issues))


def _is_meaningful_duplicate_line(line: str) -> bool:
    if not line or line.startswith("#"):
        return False
    if line in {"else:", "try:", "finally:"}:
        return False
    if line.startswith(("def ", "class ", "import ", "from ")):
        return False
    return "=" in line or line.endswith(")")


def _callable_consumed_on_public_attribute_assignment(added_lines: list[str], removed_text: str) -> list[str]:
    issues = []
    for line in added_lines:
        stripped = line.strip()
        match = re.search(
            r"\bself\.(?P<attr>[A-Za-z_]\w*)\b.*=\s*(?P<value>.+\bcallable\s*\(\s*(?P<name>[A-Za-z_]\w*)\s*\).*)",
            stripped,
        )
        if match is None:
            continue
        attr = match.group("attr")
        name = match.group("name")
        if attr != name:
            continue
        if not re.search(rf"\bself\.{attr}\b.*=\s*.*\b{name}\b", removed_text):
            continue
        issues.append(
            f"Suspicious callable consumption while storing public attribute self.{attr}. "
            f"Preserve the original {name} value on the object and evaluate it at the use site if needed."
        )
    return list(dict.fromkeys(issues))


def _review_repo_diff(diff: str, repo_root: Path) -> list[str]:
    issues: list[str] = []
    introduced_self_calls = _introduced_self_calls(diff)
    if introduced_self_calls:
        defined_methods = _defined_method_names(repo_root)
        for method in introduced_self_calls:
            if method in defined_methods:
                continue
            issues.append(
                f"Introduced call to undefined self method: self.{method}(). "
                "Use an existing method/expression or define the method in the patch."
            )
    issues.extend(_undefined_keyword_arguments(diff, repo_root))
    issues.extend(_undefined_file_local_calls(diff, repo_root))
    issues.extend(_removed_imported_names_still_used(diff, repo_root))
    return issues


def _introduced_self_calls(diff: str) -> list[str]:
    names = []
    for line in diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for name in re.findall(r"\bself\.([A-Za-z_]\w*)\s*\(", line[1:]):
            if name not in names:
                names.append(name)
    return names


def _defined_method_names(repo_root: Path) -> set[str]:
    names: set[str] = set()
    for path in repo_root.rglob("*.py"):
        if _is_skipped_review_path(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        names.update(re.findall(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", text, flags=re.MULTILINE))
        names.update(re.findall(r"^\s*async\s+def\s+([A-Za-z_]\w*)\s*\(", text, flags=re.MULTILINE))
    return names


def _undefined_keyword_arguments(diff: str, repo_root: Path) -> list[str]:
    calls = _introduced_keyword_calls(diff)
    if not calls:
        return []

    signatures = _defined_function_signatures(repo_root)
    issues: list[str] = []
    for function_name, keywords in calls.items():
        signature = signatures.get(function_name)
        if signature is None or signature["has_var_kwargs"]:
            continue
        valid = signature["params"]
        for keyword in sorted(keywords):
            if keyword not in valid:
                issues.append(
                    f"Introduced call to {function_name}() with undefined keyword argument '{keyword}'."
                )
    return list(dict.fromkeys(issues))


def _introduced_keyword_calls(diff: str) -> dict[str, set[str]]:
    calls: dict[str, set[str]] = {}
    for line in diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added = line[1:]
        for match in re.finditer(r"\b(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^()]*)\)", added):
            name = match.group("name")
            if name in {"dict", "format", "print", "super"}:
                continue
            keywords = set(re.findall(r"\b([A-Za-z_]\w*)\s*=", match.group("args")))
            if keywords:
                calls.setdefault(name, set()).update(keywords)
    return calls


def _defined_function_signatures(repo_root: Path) -> dict[str, dict[str, object]]:
    signatures: dict[str, dict[str, object]] = {}
    for path in repo_root.rglob("*.py"):
        if _is_skipped_review_path(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in re.finditer(r"^\s*(?:async\s+def|def)\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", text, flags=re.MULTILINE):
            name = match.group(1)
            raw_params = match.group(2)
            params = set()
            has_var_kwargs = False
            for raw_param in raw_params.split(","):
                param = raw_param.strip()
                if not param:
                    continue
                if param.startswith("**"):
                    has_var_kwargs = True
                    continue
                if param.startswith("*"):
                    continue
                param_name = param.split("=", 1)[0].split(":", 1)[0].strip()
                if param_name:
                    params.add(param_name)
            signatures.setdefault(name, {"params": params, "has_var_kwargs": has_var_kwargs})
    return signatures


def _undefined_file_local_calls(diff: str, repo_root: Path) -> list[str]:
    issues: list[str] = []
    for path, added_lines in _added_lines_by_file(diff).items():
        file_path = repo_root / path
        if not file_path.is_file() or file_path.suffix != ".py":
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        available_names = _file_available_names(text)
        for line in added_lines:
            for name in _bare_call_names(line):
                if name in available_names or hasattr(builtins, name):
                    continue
                issues.append(
                    f"Introduced call to undefined or unimported name {name}() in {path}."
                )
    return list(dict.fromkeys(issues))


def _added_lines_by_file(diff: str) -> dict[str, list[str]]:
    by_file: dict[str, list[str]] = {}
    current_path = ""
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            current_path = parts[3].removeprefix("b/") if len(parts) >= 4 else ""
            continue
        if current_path and line.startswith("+") and not line.startswith("+++"):
            by_file.setdefault(current_path, []).append(line[1:])
    return by_file


def _removed_imported_names_still_used(diff: str, repo_root: Path) -> list[str]:
    issues: list[str] = []
    for path, removed_names in _removed_import_names_by_file(diff).items():
        if not removed_names:
            continue
        file_path = repo_root / path
        if not file_path.is_file() or file_path.suffix != ".py":
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        import_lines = _import_line_numbers(text)
        for name in sorted(removed_names):
            if _name_is_still_available_from_imports(text, name):
                continue
            if _name_used_outside_imports(text, name, import_lines):
                issues.append(
                    f"Removed import for name {name} but {name} is still referenced in {path}."
                )
    return list(dict.fromkeys(issues))


def _removed_import_names_by_file(diff: str) -> dict[str, set[str]]:
    removed_by_file: dict[str, set[str]] = {}
    current_path = ""
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            current_path = parts[3].removeprefix("b/") if len(parts) >= 4 else ""
            continue
        if not current_path or not line.startswith("-") or line.startswith("---"):
            continue
        imported = _imported_names_from_line(line[1:])
        if imported:
            removed_by_file.setdefault(current_path, set()).update(imported)
    return removed_by_file


def _imported_names_from_line(line: str) -> set[str]:
    stripped = line.strip()
    names: set[str] = set()
    if stripped.startswith("import "):
        for imported in stripped.removeprefix("import ").split(","):
            name = imported.strip().split(" as ")[-1].split(".", 1)[0]
            if name:
                names.add(name)
    elif stripped.startswith("from ") and " import " in stripped:
        imported_part = stripped.split(" import ", 1)[1]
        if imported_part.startswith("("):
            return names
        for imported in imported_part.split(","):
            name = imported.strip().split(" as ")[-1]
            if name and name != "*":
                names.add(name)
    return names


def _import_line_numbers(text: str) -> set[int]:
    lines = set()
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            lines.add(index)
    return lines


def _name_is_still_available_from_imports(text: str, name: str) -> bool:
    return name in _file_available_names(text)


def _name_used_outside_imports(text: str, name: str, import_lines: set[int]) -> bool:
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    for index, line in enumerate(text.splitlines(), start=1):
        if index in import_lines:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if pattern.search(line):
            return True
    return False


def _file_available_names(text: str) -> set[str]:
    names: set[str] = set()
    names.update(re.findall(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_]\w*)\b", text, flags=re.MULTILINE))
    for match in re.finditer(r"^\s*import\s+(.+)$", text, flags=re.MULTILINE):
        for imported in match.group(1).split(","):
            name = imported.strip().split(" as ")[-1].split(".", 1)[0]
            if name:
                names.add(name)
    for match in re.finditer(r"^\s*from\s+[\w.]+\s+import\s+(.+)$", text, flags=re.MULTILINE):
        for imported in match.group(1).split(","):
            name = imported.strip().split(" as ")[-1]
            if name and name != "*":
                names.add(name)
    return names


def _bare_call_names(line: str) -> list[str]:
    ignored = {
        "callable",
        "dict",
        "format",
        "isinstance",
        "len",
        "list",
        "set",
        "str",
        "super",
        "tuple",
        "type",
    }
    names: list[str] = []
    for match in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)\s*\(", line):
        name = match.group(1)
        if name in ignored or name.startswith("__"):
            continue
        if name not in names:
            names.append(name)
    return names


def _is_skipped_review_path(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "site-packages"})
