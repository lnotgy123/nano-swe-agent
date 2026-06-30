from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bench.predictions import export_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trajectory patches to SWE-bench predictions JSONL.")
    parser.add_argument(
        "--input-glob",
        default=str(PROJECT_ROOT / "data" / "trajectories" / "swebench_lite" / "*.jsonl"),
    )
    parser.add_argument(
        "--output-file",
        default=str(PROJECT_ROOT / "data" / "predictions" / "swebench_lite_predictions.jsonl"),
    )
    parser.add_argument("--model-name", default="qwen2.5-coder-7b-swe-agent")
    parser.add_argument("--allow-empty-patch", action="store_true")
    parser.add_argument("--dedupe", choices=["latest", "none"], default="latest")
    args = parser.parse_args()

    stats = export_predictions(
        input_glob=args.input_glob,
        output_file=args.output_file,
        model_name_or_path=args.model_name,
        require_patch=not args.allow_empty_patch,
        dedupe=args.dedupe,
    )
    print(f"scanned: {stats.scanned}")
    print(f"exported: {stats.exported}")
    print(f"skipped_empty_patch: {stats.skipped_empty_patch}")
    print(f"deduplicated: {stats.deduplicated}")
    print(f"output_file: {stats.output_file}")


if __name__ == "__main__":
    main()
