from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent on a range of SWE-bench Lite instances.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-offline", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, required=True, help="Exclusive end index.")
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
    parser.add_argument(
        "--summary-file",
        default=None,
        help="JSONL summary path. Defaults to data/runs/swebench_lite_batch_<timestamp>.jsonl.",
    )
    parser.add_argument("--install-env", action="store_true")
    parser.add_argument("--recreate-env", action="store_true")
    parser.add_argument("--reuse-existing-env", action="store_true")
    parser.add_argument(
        "--continue-on-env-failed",
        action="store_true",
        help="Continue running the agent without an env when env installation fails; evaluation is skipped.",
    )
    parser.add_argument("--env-python", default=None)
    parser.add_argument("--install-timeout", type=int, default=1200)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-timeout", type=int, default=120)
    parser.add_argument("--command-timeout", type=int, default=300)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-output-chars", type=int, default=16000)
    args = parser.parse_args()

    if args.end_index <= args.start_index:
        raise ValueError("--end-index must be greater than --start-index.")

    summary_path = Path(args.summary_file).resolve() if args.summary_file else _default_summary_path()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.prepare_only:
        cfg = _load_config(args.config)
        gen_cfg = GenerationConfig(**cfg.get("generation", {}))
        model_cfg = cfg["model"]
        client = QwenLLMClient(
            model_path=model_cfg["path"],
            adapter_path=args.adapter_path,
            generation_config=gen_cfg,
            torch_dtype=model_cfg.get("torch_dtype", "auto"),
            device_map=model_cfg.get("device_map", "auto"),
        )

    print(f"summary_file: {summary_path}")
    for index in range(args.start_index, args.end_index):
        record = _run_one(index=index, args=args, client=client)
        _append_jsonl(summary_path, record)
        print(
            "index={index} instance_id={instance_id} status={status} "
            "has_patch={has_patch} local_resolved={local_resolved} trajectory={trajectory_path}".format(
                index=record.get("index"),
                instance_id=record.get("instance_id"),
                status=record.get("status"),
                has_patch=record.get("has_patch"),
                local_resolved=record.get("local_resolved"),
                trajectory_path=record.get("trajectory_path"),
            )
        )


def _run_one(index: int, args: argparse.Namespace, client: QwenLLMClient | None) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    base_record: dict[str, Any] = {
        "index": index,
        "split": args.split,
        "started_at": started_at.isoformat(),
        "status": "started",
    }
    try:
        instance = load_swebench_lite(split=args.split, index=index, offline=args.dataset_offline)
        base_record.update(
            {
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "base_commit": instance.base_commit,
            }
        )

        if args.skip_existing and _has_existing_trajectory(args.trajectory_dir, instance.instance_id):
            return _finish_record(base_record, status="skipped_existing", started_at=started_at)

        repo_dir = prepare_repo_workspace(
            instance,
            args.workspace_root,
            reuse_existing=args.reuse_existing_workspace,
            reset_existing=args.reset_existing_workspace,
            repo_cache_root=args.repo_cache_root,
        )
        base_record["workspace"] = str(repo_dir)

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
            base_record["env"] = env_result.to_dict()
            if not env_result.ok:
                if not args.continue_on_env_failed:
                    return _finish_record(base_record, status="env_failed", started_at=started_at)
                base_record["env_failed_continued"] = True
                env_path = None

        if args.prepare_only:
            return _finish_record(base_record, status="prepared", started_at=started_at)

        if client is None:
            raise RuntimeError("Model client was not initialized.")

        tools = ToolExecutor(repo_dir, max_output_chars=args.max_output_chars, env_path=env_path)
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
        if not args.skip_eval and not base_record.get("env_failed_continued"):
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

        trajectory_path = save_trajectory(
            result=result,
            repo_root=repo_dir,
            output_dir=args.trajectory_dir,
            run_name=instance.instance_id,
            evaluation=evaluation,
            run_default_tests=False,
        )
        final_diff = (
            result.steps[-1].result.output
            if result.steps
            and result.steps[-1].tool in {"git_diff", "submit"}
            and result.steps[-1].result
            else ""
        )
        patch = evaluation.get("patch", "") if evaluation else _read_git_diff(repo_dir)

        base_record.update(
            {
                "status": "completed",
                "finished": result.finished,
                "summary": result.summary,
                "steps": len(result.steps),
                "agent_mode": args.agent_mode,
                "trajectory_path": str(trajectory_path),
                "has_patch": bool(patch.strip()),
                "local_resolved": evaluation.get("resolved") if evaluation else None,
                "fail_to_pass_passed": evaluation.get("fail_to_pass_passed") if evaluation else None,
                "pass_to_pass_passed": evaluation.get("pass_to_pass_passed") if evaluation else None,
                "last_git_diff_output_chars": len(final_diff),
            }
        )
        return _finish_record(base_record, status="completed", started_at=started_at)
    except Exception as exc:
        base_record.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return _finish_record(base_record, status="error", started_at=started_at)


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _has_existing_trajectory(trajectory_dir: str | Path, instance_id: str) -> bool:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in instance_id)
    return any(Path(trajectory_dir).glob(f"{safe}_*.jsonl"))


def _read_git_diff(repo_dir: Path) -> str:
    import subprocess

    proc = subprocess.run(["git", "diff", "--", "."], cwd=repo_dir, text=True, capture_output=True, timeout=60)
    return proc.stdout if proc.returncode == 0 else ""


def _finish_record(record: dict[str, Any], status: str, started_at: datetime) -> dict[str, Any]:
    record["status"] = status
    finished_at = datetime.now(timezone.utc)
    record["finished_at"] = finished_at.isoformat()
    record["duration_seconds"] = round((finished_at - started_at).total_seconds(), 3)
    record.setdefault("has_patch", None)
    record.setdefault("local_resolved", None)
    record.setdefault("trajectory_path", None)
    return record


def _default_summary_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "data" / "runs" / f"swebench_lite_batch_{stamp}.jsonl"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
