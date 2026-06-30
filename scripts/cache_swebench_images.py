from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def image_name(instance_id: str, namespace: str) -> str:
    suffix = f"sweb.eval.x86_64.{instance_id.lower()}:latest".replace("__", "_1776_")
    return f"{namespace.rstrip('/')}/{suffix}"


def run(command: list[str], timeout: int) -> None:
    subprocess.run(command, check=True, timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache SWE-bench instance images through a registry mirror.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions")
    source.add_argument("--end-index", type=int)
    source.add_argument("--instance-id", action="append", dest="instance_ids")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--mirror", default="docker.1ms.run/swebench")
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--pull-timeout", type=int, default=300)
    args = parser.parse_args()

    if args.instance_ids:
        instance_ids = list(dict.fromkeys(args.instance_ids))
    elif args.predictions:
        rows = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
        instance_ids = list(dict.fromkeys(row["instance_id"] for row in rows))
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        from datasets import load_dataset

        dataset = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
        instance_ids = [dataset[index]["instance_id"] for index in range(args.start_index, args.end_index)]
    for index, instance_id in enumerate(instance_ids, start=1):
        source = image_name(instance_id, args.mirror)
        target = image_name(instance_id, args.namespace)
        inspect = subprocess.run(["docker", "image", "inspect", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if inspect.returncode == 0:
            print(f"[{index}/{len(instance_ids)}] cached {target}", flush=True)
            continue

        for candidate in (source, target):
            pulled = False
            candidate_retries = min(args.retries, 2) if candidate != target else args.retries
            for attempt in range(1, candidate_retries + 1):
                try:
                    print(
                        f"[{index}/{len(instance_ids)}] pull {candidate} "
                        f"(attempt {attempt}/{candidate_retries})",
                        flush=True,
                    )
                    run(["docker", "pull", candidate], timeout=args.pull_timeout)
                    if candidate != target:
                        run(["docker", "tag", candidate, target], timeout=30)
                    pulled = True
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    if attempt < candidate_retries:
                        time.sleep(5 * attempt)
            if pulled:
                break
        else:
            raise RuntimeError(f"Failed to cache image for {instance_id} from mirror and official registry.")


if __name__ == "__main__":
    main()
