from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bench.env_manager import prepare_instance_env
from bench.swebench_lite import load_swebench_lite, prepare_repo_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and install dependencies for one SWE-bench Lite env.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--workspace-root",
        default=str(PROJECT_ROOT / "data" / "workspaces" / "swebench_lite"),
    )
    parser.add_argument("--reuse-existing-workspace", action="store_true")
    parser.add_argument(
        "--env-root",
        default=str(PROJECT_ROOT / "data" / "envs" / "swebench_lite"),
    )
    parser.add_argument("--recreate-env", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--env-python", default=None)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    instance = load_swebench_lite(split=args.split, index=args.index)
    repo_dir = prepare_repo_workspace(
        instance,
        args.workspace_root,
        reuse_existing=args.reuse_existing_workspace,
    )
    result = prepare_instance_env(
        repo_root=repo_dir,
        env_root=args.env_root,
        instance_id=instance.instance_id,
        recreate=args.recreate_env,
        reuse_existing=args.reuse_existing,
        timeout=args.timeout,
        python_executable=args.env_python,
    )
    print(f"instance_id: {instance.instance_id}")
    print(f"workspace: {repo_dir}")
    print(f"env_path: {result.env_path}")
    print(f"ok: {result.ok}")
    print(f"log_path: {result.log_path}")
    for command in result.commands:
        print(f"command: {command}")


if __name__ == "__main__":
    main()
