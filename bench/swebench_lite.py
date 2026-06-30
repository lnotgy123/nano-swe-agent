from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASET_NAME = "SWE-bench/SWE-bench_Lite"


@dataclass(frozen=True)
class SWEBenchInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    test_patch: str
    raw: dict[str, Any]


def load_swebench_lite(split: str = "test", index: int = 0, offline: bool = False) -> SWEBenchInstance:
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, split=split)
    row = dict(dataset[index])
    return parse_instance(row)


def parse_instance(row: dict[str, Any]) -> SWEBenchInstance:
    return SWEBenchInstance(
        instance_id=row["instance_id"],
        repo=row["repo"],
        base_commit=row["base_commit"],
        problem_statement=row["problem_statement"],
        fail_to_pass=_parse_test_list(row.get("FAIL_TO_PASS", [])),
        pass_to_pass=_parse_test_list(row.get("PASS_TO_PASS", [])),
        test_patch=str(row.get("test_patch", "") or ""),
        raw=row,
    )


def prepare_repo_workspace(
    instance: SWEBenchInstance,
    workspace_root: str | Path,
    reuse_existing: bool = False,
    reset_existing: bool = False,
    repo_cache_root: str | Path | None = None,
) -> Path:
    workspace_root = Path(workspace_root).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    safe_id = instance.instance_id.replace("/", "__")
    repo_dir = workspace_root / safe_id
    if reuse_existing and repo_dir.exists():
        if not _has_valid_head(repo_dir):
            shutil.rmtree(repo_dir)
        else:
            if reset_existing:
                _run(["git", "reset", "--hard", "HEAD"], cwd=repo_dir)
                _run(["git", "clean", "-fd"], cwd=repo_dir)
            return repo_dir

    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    clone_source = _prepare_repo_cache(instance.repo, repo_cache_root) if repo_cache_root else f"https://github.com/{instance.repo}.git"
    _run(["git", "clone", "--no-tags", str(clone_source), str(repo_dir)], cwd=workspace_root, timeout=1800)
    _checkout_base_commit(repo_dir, instance.base_commit, instance.repo, repo_cache_root)
    _apply_test_patch(instance, repo_dir)
    _run(["git", "status", "--short"], cwd=repo_dir)
    return repo_dir


def _has_valid_head(repo_dir: Path) -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repo_dir,
        text=True,
        capture_output=True,
    )
    return proc.returncode == 0


def _checkout_base_commit(
    repo_dir: Path,
    base_commit: str,
    repo: str,
    repo_cache_root: str | Path | None,
) -> None:
    try:
        _run(["git", "checkout", base_commit], cwd=repo_dir)
        return
    except RuntimeError:
        if repo_cache_root is None:
            raise

    mirror_dir = _prepare_repo_cache(repo, repo_cache_root, update_existing=True)
    _run(["git", "fetch", "--no-tags", str(mirror_dir), base_commit], cwd=repo_dir, timeout=1800)
    _run(["git", "checkout", base_commit], cwd=repo_dir)


def _prepare_repo_cache(repo: str, cache_root: str | Path | None, update_existing: bool = False) -> Path:
    if cache_root is None:
        raise ValueError("cache_root is required.")
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    safe_name = repo.replace("/", "__")
    mirror_dir = cache_root / f"{safe_name}.git"
    repo_url = f"https://github.com/{repo}.git"
    if mirror_dir.exists() and not _is_valid_git_mirror(mirror_dir):
        shutil.rmtree(mirror_dir)
    if not mirror_dir.exists():
        _run(["git", "clone", "--mirror", repo_url, str(mirror_dir)], cwd=cache_root, timeout=1800)
    elif update_existing:
        _run(["git", "remote", "update", "--prune"], cwd=mirror_dir, timeout=1800)
    return mirror_dir


def _is_valid_git_mirror(mirror_dir: Path) -> bool:
    proc = subprocess.run(
        ["git", "--git-dir", str(mirror_dir), "rev-parse", "--is-bare-repository"],
        text=True,
        capture_output=True,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _apply_test_patch(instance: SWEBenchInstance, repo_dir: Path) -> None:
    if not instance.test_patch.strip():
        return

    proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=repo_dir,
        input=instance.test_patch,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to apply SWE-bench test patch.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    _run(["git", "config", "user.email", "swe-agent@example.local"], cwd=repo_dir)
    _run(["git", "config", "user.name", "swe-agent"], cwd=repo_dir)
    _run(["git", "add", "."], cwd=repo_dir)
    _run(["git", "commit", "-m", "Apply SWE-bench test patch"], cwd=repo_dir)


def instance_to_task(instance: SWEBenchInstance) -> str:
    fail_to_pass = "\n".join(f"- {test}" for test in instance.fail_to_pass) or "- <none provided>"
    pass_to_pass = "\n".join(f"- {test}" for test in instance.pass_to_pass) or "- <none provided>"
    return f"""You are working on SWE-bench Lite instance {instance.instance_id}.

Repository: {instance.repo}
Base commit: {instance.base_commit}

Problem statement:
{instance.problem_statement}

Known fail-to-pass tests:
{fail_to_pass}

Known pass-to-pass tests:
{pass_to_pass}

Find the relevant source files, make a minimal code fix, run focused tests when possible, inspect the diff, and finish with a concise summary."""


def print_instance_summary(instance: SWEBenchInstance) -> None:
    print(f"instance_id: {instance.instance_id}")
    print(f"repo: {instance.repo}")
    print(f"base_commit: {instance.base_commit}")
    print("FAIL_TO_PASS:")
    for test in instance.fail_to_pass:
        print(f"  - {test}")
    print("PASS_TO_PASS:")
    for test in instance.pass_to_pass:
        print(f"  - {test}")
    print("problem_statement:")
    print(instance.problem_statement)


def _parse_test_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]


def _run(command: list[str], cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        raise RuntimeError(
            f"Command timed out after {timeout}s: {' '.join(command)}\n"
            f"cwd: {cwd}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n"
            f"cwd: {cwd}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc
