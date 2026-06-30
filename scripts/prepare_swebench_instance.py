from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bench.swebench_lite import load_swebench_lite, prepare_repo_workspace, print_instance_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one SWE-bench Lite workspace.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--workspace-root",
        default=str(PROJECT_ROOT / "data" / "workspaces" / "swebench_lite"),
    )
    parser.add_argument("--reuse-existing-workspace", action="store_true")
    args = parser.parse_args()

    instance = load_swebench_lite(split=args.split, index=args.index)
    print_instance_summary(instance)
    repo_dir = prepare_repo_workspace(
        instance,
        args.workspace_root,
        reuse_existing=args.reuse_existing_workspace,
    )
    print(f"workspace: {repo_dir}")


if __name__ == "__main__":
    main()
