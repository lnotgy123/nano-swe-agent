from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from swebench.harness.test_spec.test_spec import TestSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm_client import GenerationConfig, QwenLLMClient
from agent.sweagent_xml_workflow import SWEAgentXMLConfig, SWEAgentXMLWorkflow
from agent.trajectory import save_trajectory
from bench.swebench_lite import parse_instance
from tools.docker_executor import DockerMountedToolExecutor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SWE-bench rollouts with shared official Docker environments.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "qwen_local.yaml"))
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument(
        "--spec-cache",
        default=str(PROJECT_ROOT / "data" / "cache" / "swebench" / "lite_test_specs_0_100.json"),
    )
    parser.add_argument(
        "--workspace-root",
        default=str(Path.home() / ".cache" / "swe-agent" / "workspaces" / "shared_docker_0_100"),
    )
    parser.add_argument(
        "--repo-cache-root",
        default=str(PROJECT_ROOT / "data" / "repo_cache"),
    )
    parser.add_argument(
        "--trajectory-dir",
        default=str(PROJECT_ROOT / "data" / "trajectories" / "shared_docker_v2_0_100"),
    )
    parser.add_argument(
        "--summary-file",
        default=str(PROJECT_ROOT / "data" / "runs" / "shared_docker_v2_0_100.jsonl"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-output-chars", type=int, default=16000)
    args = parser.parse_args()

    specs = [TestSpec(**item) for item in json.loads(Path(args.spec_cache).read_text())]
    specs_by_id = {spec.instance_id: spec for spec in specs}
    representatives = _cached_representatives(specs)
    covered_envs = set(representatives)

    dataset = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    selected = []
    for index in range(args.start_index, min(args.end_index, len(dataset))):
        row = dict(dataset[index])
        spec = specs_by_id[row["instance_id"]]
        if spec.env_image_key in covered_envs:
            selected.append((index, row, spec, representatives[spec.env_image_key]))
    print(f"covered_instances: {len(selected)}", flush=True)

    cfg = yaml.safe_load(Path(args.config).read_text())
    client = QwenLLMClient(
        model_path=cfg["model"]["path"],
        adapter_path=args.adapter_path,
        generation_config=GenerationConfig(**cfg.get("generation", {})),
        torch_dtype=cfg["model"].get("torch_dtype", "auto"),
        device_map=cfg["model"].get("device_map", "auto"),
    )

    summary_path = Path(args.summary_file).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    for index, row, spec, image in selected:
        instance_id = row["instance_id"]
        if args.skip_existing and _has_trajectory(args.trajectory_dir, instance_id):
            print(f"index={index} instance_id={instance_id} status=skipped_existing", flush=True)
            continue
        record = _run_one(index, row, spec, image, args, client)
        with summary_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"index={index} instance_id={instance_id} status={record['status']} "
            f"has_patch={record.get('has_patch')} steps={record.get('steps')}",
            flush=True,
        )


def _run_one(
    index: int,
    row: dict[str, Any],
    spec: TestSpec,
    image: str,
    args: argparse.Namespace,
    client: QwenLLMClient,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    instance = parse_instance(row)
    record: dict[str, Any] = {
        "index": index,
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "base_commit": instance.base_commit,
        "env_image_key": spec.env_image_key,
        "representative_image": image,
        "started_at": started.isoformat(),
    }
    workspace = None
    try:
        workspace = _prepare_workspace(instance, args.workspace_root, args.repo_cache_root)
        record["workspace"] = str(workspace)
        with DockerMountedToolExecutor(
            workspace,
            image=image,
            instance_id=instance.instance_id,
            max_output_chars=args.max_output_chars,
            repo_cache_root=args.repo_cache_root,
        ) as tools:
            result = SWEAgentXMLWorkflow(
                llm=client,
                tools=tools,
                config=SWEAgentXMLConfig(
                    fail_to_pass=instance.fail_to_pass,
                    pass_to_pass=instance.pass_to_pass,
                    max_steps=args.max_steps,
                ),
            ).run(instance.problem_statement)

        trajectory = save_trajectory(
            result=result,
            repo_root=workspace,
            output_dir=args.trajectory_dir,
            run_name=instance.instance_id,
            run_default_tests=False,
        )
        patch = _git_diff(workspace)
        record.update(
            status="completed",
            finished=result.finished,
            steps=len(result.steps),
            has_patch=bool(patch.strip()),
            trajectory_path=str(trajectory),
        )
    except Exception as exc:
        record.update(
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    finished = datetime.now(timezone.utc)
    record["finished_at"] = finished.isoformat()
    record["duration_seconds"] = round((finished - started).total_seconds(), 3)
    return record


def _cached_representatives(specs: list[TestSpec]) -> dict[str, str]:
    grouped: dict[str, list[TestSpec]] = defaultdict(list)
    for spec in specs:
        grouped[spec.env_image_key].append(spec)
    representatives: dict[str, str] = {}
    for env_key, group in grouped.items():
        for spec in group:
            if subprocess.run(
                ["docker", "image", "inspect", spec.instance_image_key],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0:
                representatives[env_key] = spec.instance_image_key
                break
    return representatives


def _prepare_workspace(instance, workspace_root: str, repo_cache_root: str) -> Path:
    root = Path(workspace_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / instance.instance_id
    if target.exists():
        subprocess.run(["git", "reset", "--hard"], cwd=target, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "clean", "-fdx"], cwd=target, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "checkout", "--detach", instance.base_commit], cwd=target, check=True, stdout=subprocess.DEVNULL)
        return target

    mirror = Path(repo_cache_root).resolve() / f"{instance.repo.replace('/', '__')}.git"
    if not mirror.exists():
        raise FileNotFoundError(f"Repository mirror not found: {mirror}")
    subprocess.run(
        ["git", "clone", "--no-tags", "--reference-if-able", str(mirror), str(mirror), str(target)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", instance.base_commit],
        cwd=target,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return target


def _git_diff(workspace: Path) -> str:
    proc = subprocess.run(["git", "diff", "--", "."], cwd=workspace, text=True, capture_output=True, check=True)
    return proc.stdout


def _has_trajectory(directory: str, instance_id: str) -> bool:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in instance_id)
    return any(Path(directory).glob(f"{safe}_*.jsonl"))


if __name__ == "__main__":
    main()
