from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bench.predictions import import_official_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Import official SWE-bench results into trajectory JSONL files.")
    parser.add_argument(
        "--trajectory-glob",
        default=str(PROJECT_ROOT / "data" / "trajectories" / "swebench_lite" / "*.jsonl"),
    )
    parser.add_argument("--results-path", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "trajectories_with_official_eval"),
    )
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    stats = import_official_results(
        trajectory_glob=args.trajectory_glob,
        results_path=args.results_path,
        output_dir=None if args.in_place else args.output_dir,
        in_place=args.in_place,
    )
    print(f"results_loaded: {stats.results_loaded}")
    print(f"trajectories_scanned: {stats.trajectories_scanned}")
    print(f"trajectories_updated: {stats.trajectories_updated}")
    print(f"output_dir: {stats.output_dir or '<in-place>'}")


if __name__ == "__main__":
    main()
