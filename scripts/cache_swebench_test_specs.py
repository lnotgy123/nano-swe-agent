from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import requests
from datasets import load_dataset
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.test_spec import python as swebench_python_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache official SWE-bench TestSpecs for offline execution.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--repo-cache-root", default=str(PROJECT_ROOT / "data" / "repo_cache"))
    parser.add_argument("--offline-dataset", action="store_true")
    args = parser.parse_args()

    if args.offline_dataset:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    _patch_swebench_file_fetches(Path(args.repo_cache_root))

    dataset = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    output = Path(args.output).resolve()
    progress = output.with_suffix(output.suffix + ".progress.jsonl")
    progress.parent.mkdir(parents=True, exist_ok=True)
    specs = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()] if progress.exists() else []
    next_index = args.start_index + len(specs)
    for index in range(next_index, args.end_index):
        row = dict(dataset[index])
        print(f"[{index + 1 - args.start_index}/{args.end_index - args.start_index}] {row['instance_id']}", flush=True)
        for attempt in range(1, args.retries + 1):
            try:
                spec = asdict(make_test_spec(row, namespace=args.namespace))
                break
            except requests.RequestException as exc:
                if attempt == args.retries:
                    raise
                delay = min(5 * attempt, 30)
                print(f"  network retry {attempt}/{args.retries} in {delay}s: {exc}", flush=True)
                time.sleep(delay)
        specs.append(spec)
        with progress.open("a", encoding="utf-8") as file:
            file.write(json.dumps(spec, ensure_ascii=False) + "\n")

    output.write_text(json.dumps(specs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(specs)} TestSpecs to {output}", flush=True)


def _patch_swebench_file_fetches(repo_cache_root: Path) -> None:
    def show_file(repo: str, commit: str, path: str) -> str | None:
        mirror = repo_cache_root / f"{repo.replace('/', '__')}.git"
        if not mirror.exists():
            return None
        proc = subprocess.run(
            ["git", "--git-dir", str(mirror), "show", f"{commit}:{path}"],
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout

    def get_environment_yml_by_commit(repo: str, commit: str, env_name: str) -> str:
        for req_path in swebench_python_spec.MAP_REPO_TO_ENV_YML_PATHS[repo]:
            text = show_file(repo, commit, req_path)
            if text is not None:
                lines = text.split("\n")
                cleaned = [f"name: {env_name}" if line.startswith("name:") else line for line in lines]
                return "\n".join(cleaned)
        raise ValueError(
            f"Could not find environment.yml at paths {swebench_python_spec.MAP_REPO_TO_ENV_YML_PATHS[repo]} "
            f"for repo {repo} at commit {commit}"
        )

    def get_requirements_by_commit(repo: str, commit: str) -> str:
        req_path = None
        lines = None
        for candidate in swebench_python_spec.MAP_REPO_TO_REQS_PATHS[repo]:
            text = show_file(repo, commit, candidate)
            if text is not None:
                req_path = candidate
                lines = text
                break
        if req_path is None or lines is None:
            raise ValueError(
                f"Could not find requirements.txt at paths {swebench_python_spec.MAP_REPO_TO_REQS_PATHS[repo]} "
                f"for repo {repo} at commit {commit}"
            )

        original_req = []
        additional_reqs = []
        req_dir = "/".join(req_path.split("/")[:-1])

        def exclude_line(line: str) -> bool:
            return any(line.strip().startswith(prefix) for prefix in ["-e .", "#", ".[test"])

        for line in lines.split("\n"):
            if line.strip().startswith("-r"):
                file_name = line[len("-r") :].strip()
                nested_path = f"{req_dir}/{file_name}" if req_dir else file_name
                nested = show_file(repo, commit, nested_path)
                if nested is not None:
                    for line_extra in nested.split("\n"):
                        if not exclude_line(line_extra):
                            additional_reqs.append(line_extra)
            elif not exclude_line(line):
                original_req.append(line)

        additional_reqs.append("\n".join(original_req))
        return "\n".join(additional_reqs)

    swebench_python_spec.get_environment_yml_by_commit = get_environment_yml_by_commit
    swebench_python_spec.get_requirements_by_commit = get_requirements_by_commit


if __name__ == "__main__":
    main()
