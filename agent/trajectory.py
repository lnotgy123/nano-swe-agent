from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from agent.types import AgentRunResult
from tools.executor import ToolResult


def save_trajectory(
    result: AgentRunResult,
    repo_root: str | Path,
    output_dir: str | Path,
    run_name: str,
    evaluation: dict | None = None,
    run_default_tests: bool = True,
) -> Path:
    repo_path = Path(repo_root).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    tests_passed = _command_ok(repo_path, "python -m pytest -q") if run_default_tests else None
    diff = _command_output(repo_path, "git diff -- .")
    record = {
        "run_id": _run_id(run_name),
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_path),
        "task": result.task,
        "initial_repo_context": _tool_result_to_dict(result.repo_context),
        "steps": [_step_to_dict(index, step) for index, step in enumerate(result.steps, start=1)],
        "messages": result.messages,
        "final": {
            "finished": result.finished,
            "summary": result.summary,
            "tests_passed": tests_passed,
            "diff": diff,
        },
    }
    if evaluation is not None:
        record["evaluation"] = evaluation

    file_path = output_path / f"{record['run_id']}.jsonl"
    with file_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return file_path


def _run_id(run_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name)
    return f"{safe_name}_{stamp}"


def _step_to_dict(index: int, step: object) -> dict:
    data = asdict(step)
    data["index"] = index
    if step.result is not None:
        data["result"] = _tool_result_to_dict(step.result)
    return data


def _tool_result_to_dict(result: ToolResult) -> dict:
    return {
        "name": result.name,
        "ok": result.ok,
        "output": result.output,
        "error_type": result.error_type,
    }


def _command_ok(repo_root: Path, command: str) -> bool:
    env = _subprocess_env()
    proc = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        shell=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    return proc.returncode == 0


def _command_output(repo_root: Path, command: str) -> str:
    env = _subprocess_env()
    proc = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        shell=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    output = "\n".join(
        part
        for part in [
            f"exit_code: {proc.returncode}",
            "stdout:",
            proc.stdout,
            "stderr:",
            proc.stderr,
        ]
        if part is not None
    )
    return output


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    python_bin = str(Path(sys.executable).resolve().parent)
    env["PATH"] = f"{python_bin}{os.pathsep}{env.get('PATH', '')}"
    return env
