from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm_client import GenerationConfig, QwenLLMClient
from agent.sweagent_xml_workflow import SWEAgentXMLConfig, SWEAgentXMLWorkflow
from agent.trajectory import save_trajectory
from tools.executor import ToolExecutor


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_workspace(source: Path, workdir: Path) -> None:
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(
        source,
        workdir,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
    )
    subprocess.run(["git", "init"], cwd=workdir, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=workdir, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=swe-agent",
            "-c",
            "user.email=swe-agent@example.local",
            "commit",
            "-m",
            "initial buggy calculator",
        ],
        cwd=workdir,
        check=True,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal agent on a toy repo.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "qwen_local.yaml"),
    )
    parser.add_argument(
        "--source",
        default=str(PROJECT_ROOT / "data" / "toy_repos" / "calculator_buggy"),
    )
    parser.add_argument(
        "--workdir",
        default=str(PROJECT_ROOT / "data" / "workspaces" / "calculator_buggy_run"),
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--trajectory-dir",
        default=str(PROJECT_ROOT / "data" / "trajectories" / "toy"),
    )
    parser.add_argument("--no-save-trajectory", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    workdir = Path(args.workdir).resolve()
    prepare_workspace(source, workdir)

    cfg = load_config(args.config)
    gen_cfg = GenerationConfig(**cfg.get("generation", {}))
    model_cfg = cfg["model"]
    client = QwenLLMClient(
        model_path=model_cfg["path"],
        generation_config=gen_cfg,
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        device_map=model_cfg.get("device_map", "auto"),
    )

    task = (
        "The repository contains a small calculator module. "
        "One test is failing. Find the bug, fix the code, run the tests, "
        "inspect the diff, and finish with a short summary."
    )
    loop = SWEAgentXMLWorkflow(
        llm=client,
        tools=ToolExecutor(workdir),
        config=SWEAgentXMLConfig(
            fail_to_pass=[],
            pass_to_pass=[],
            max_steps=args.max_steps,
        ),
    )
    result = loop.run(task)

    trajectory_path = None
    if not args.no_save_trajectory:
        trajectory_path = save_trajectory(
            result=result,
            repo_root=workdir,
            output_dir=args.trajectory_dir,
            run_name="calculator_buggy",
        )

    print(f"finished: {result.finished}")
    print(f"summary: {result.summary}")
    if trajectory_path is not None:
        print(f"trajectory: {trajectory_path}")
    print()
    for index, step in enumerate(result.steps, start=1):
        print(f"step {index}: {step.tool} {step.args}")
        if step.result is not None:
            print(f"ok: {step.result.ok}")
            print(step.result.output)
        print("-" * 80)


if __name__ == "__main__":
    main()
