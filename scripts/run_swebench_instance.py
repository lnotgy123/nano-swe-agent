from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm_client import GenerationConfig, QwenLLMClient
from agent.sweagent_xml_workflow import SWEAgentXMLConfig, SWEAgentXMLWorkflow
from agent.trajectory import save_trajectory
from bench.evaluator import evaluate_patch
from bench.env_manager import prepare_instance_env
from bench.swebench_lite import load_swebench_lite, prepare_repo_workspace
from tools.executor import ToolExecutor


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent on one SWE-bench Lite instance.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--dataset-offline", action="store_true")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--agent-mode", choices=["sweagent_xml"], default="sweagent_xml")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "qwen_local.yaml"),
    )
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument(
        "--workspace-root",
        default=str(PROJECT_ROOT / "data" / "workspaces" / "swebench_lite"),
    )
    parser.add_argument(
        "--repo-cache-root",
        default=str(PROJECT_ROOT / "data" / "repo_cache"),
    )
    parser.add_argument("--reuse-existing-workspace", action="store_true")
    parser.add_argument("--reset-existing-workspace", action="store_true")
    parser.add_argument(
        "--trajectory-dir",
        default=str(PROJECT_ROOT / "data" / "trajectories" / "swebench_lite"),
    )
    parser.add_argument(
        "--env-root",
        default=str(PROJECT_ROOT / "data" / "envs" / "swebench_lite"),
    )
    parser.add_argument("--install-env", action="store_true")
    parser.add_argument("--recreate-env", action="store_true")
    parser.add_argument("--reuse-existing-env", action="store_true")
    parser.add_argument("--env-python", default=None)
    parser.add_argument("--install-timeout", type=int, default=1200)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-timeout", type=int, default=120)
    parser.add_argument("--command-timeout", type=int, default=300)
    parser.add_argument("--no-save-trajectory", action="store_true")
    args = parser.parse_args()

    instance = load_swebench_lite(split=args.split, index=args.index, offline=args.dataset_offline)
    repo_dir = prepare_repo_workspace(
        instance,
        args.workspace_root,
        reuse_existing=args.reuse_existing_workspace,
        reset_existing=args.reset_existing_workspace,
        repo_cache_root=args.repo_cache_root,
    )
    print(f"instance_id: {instance.instance_id}")
    print(f"repo: {instance.repo}")
    print(f"workspace: {repo_dir}")
    env_result = None
    env_path = None
    if args.install_env:
        env_result = prepare_instance_env(
            repo_root=repo_dir,
            env_root=args.env_root,
            instance_id=instance.instance_id,
            recreate=args.recreate_env,
            reuse_existing=args.reuse_existing_env,
            timeout=args.install_timeout,
            python_executable=args.env_python,
        )
        env_path = env_result.env_path
        print(f"env_path: {env_result.env_path}")
        print(f"env_install_ok: {env_result.ok}")
        print(f"env_install_log: {env_result.log_path}")

    if args.prepare_only:
        return

    cfg = load_config(args.config)
    gen_cfg = GenerationConfig(**cfg.get("generation", {}))
    model_cfg = cfg["model"]
    client = QwenLLMClient(
        model_path=model_cfg["path"],
        adapter_path=args.adapter_path,
        generation_config=gen_cfg,
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        device_map=model_cfg.get("device_map", "auto"),
    )

    tools = ToolExecutor(repo_dir, max_output_chars=8000, env_path=env_path)
    loop = SWEAgentXMLWorkflow(
        llm=client,
        tools=tools,
        config=SWEAgentXMLConfig(
            fail_to_pass=instance.fail_to_pass,
            pass_to_pass=instance.pass_to_pass,
            max_steps=args.max_steps,
            test_timeout=args.command_timeout,
        ),
    )
    result = loop.run(instance.problem_statement)
    evaluation = None
    if not args.skip_eval:
        evaluation_result = evaluate_patch(
            repo_root=repo_dir,
            fail_to_pass=instance.fail_to_pass,
            pass_to_pass=instance.pass_to_pass,
            timeout=args.eval_timeout,
            env_path=env_path,
        )
        evaluation = evaluation_result.to_dict()
        if env_result is not None:
            evaluation["environment"] = env_result.to_dict()

    trajectory_path = None
    if not args.no_save_trajectory:
        trajectory_path = save_trajectory(
            result=result,
            repo_root=repo_dir,
            output_dir=args.trajectory_dir,
            run_name=instance.instance_id,
            evaluation=evaluation,
            run_default_tests=False,
        )

    print(f"finished: {result.finished}")
    print(f"summary: {result.summary}")
    if evaluation is not None:
        print(f"resolved: {evaluation['resolved']}")
        print(f"fail_to_pass_passed: {evaluation['fail_to_pass_passed']}")
        print(f"pass_to_pass_passed: {evaluation['pass_to_pass_passed']}")
    if trajectory_path is not None:
        print(f"trajectory: {trajectory_path}")
    for index, step in enumerate(result.steps, start=1):
        print(f"step {index}: {step.tool} {step.args}")
        if step.result is not None:
            print(f"ok: {step.result.ok}")
            print(step.result.output)
        print("-" * 80)


if __name__ == "__main__":
    main()
