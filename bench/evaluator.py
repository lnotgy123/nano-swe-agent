from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from bench.test_commands import command_argv


@dataclass(frozen=True)
class TestRun:
    test: str
    command: str
    exit_code: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True)
class EvaluationResult:
    resolved: bool
    patch: str
    fail_to_pass: list[TestRun]
    pass_to_pass: list[TestRun]

    @property
    def fail_to_pass_passed(self) -> bool:
        return bool(self.fail_to_pass) and all(run.passed for run in self.fail_to_pass)

    @property
    def pass_to_pass_passed(self) -> bool:
        return all(run.passed for run in self.pass_to_pass)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["fail_to_pass_passed"] = self.fail_to_pass_passed
        data["pass_to_pass_passed"] = self.pass_to_pass_passed
        return data


def evaluate_patch(
    repo_root: str | Path,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    timeout: int = 120,
    env_path: str | Path | None = None,
) -> EvaluationResult:
    repo_path = Path(repo_root).resolve()
    env_path = Path(env_path).resolve() if env_path else None
    patch = _command_output(repo_path, ["git", "diff", "--", "."], env_path=env_path)
    fail_runs = [
        run
        for test in fail_to_pass
        if (run := _run_pytest(repo_path, test, timeout=timeout, env_path=env_path)) is not None
    ]
    pass_runs = [
        run
        for test in pass_to_pass
        if (run := _run_pytest(repo_path, test, timeout=timeout, env_path=env_path)) is not None
    ]
    resolved = bool(patch.strip()) and bool(fail_runs) and all(run.passed for run in fail_runs) and all(
        run.passed for run in pass_runs
    )
    return EvaluationResult(
        resolved=resolved,
        patch=patch,
        fail_to_pass=fail_runs,
        pass_to_pass=pass_runs,
    )


def _run_pytest(repo_root: Path, test: str, timeout: int, env_path: Path | None) -> TestRun | None:
    command = command_argv(test, repo_root)
    if command is None:
        return None
    try:
        proc = subprocess.run(
            command,
            cwd=repo_root,
            env=_subprocess_env(env_path),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return TestRun(
            test=test,
            command=" ".join(command),
            exit_code=124,
            stdout=_timeout_output_text(exc.stdout),
            stderr=_timeout_output_text(exc.stderr) + f"\nTimed out after {timeout} seconds.",
        )
    return TestRun(
        test=test,
        command=" ".join(command),
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _command_output(repo_root: Path, command: list[str], env_path: Path | None = None) -> str:
    proc = subprocess.run(
        command,
        cwd=repo_root,
        env=_subprocess_env(env_path),
        text=True,
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return (
            f"command: {' '.join(command)}\n"
            f"exit_code: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout


def _subprocess_env(env_path: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    python_bin = str((env_path / "bin").resolve()) if env_path else str(Path(sys.executable).resolve().parent)
    env["PATH"] = f"{python_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def _timeout_output_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""
