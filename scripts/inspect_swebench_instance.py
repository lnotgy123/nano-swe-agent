from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bench.swebench_lite import load_swebench_lite, print_instance_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one SWE-bench Lite instance.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    instance = load_swebench_lite(split=args.split, index=args.index)
    print_instance_summary(instance)


if __name__ == "__main__":
    main()
