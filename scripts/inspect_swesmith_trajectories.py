from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "SWE-bench/SWE-smith-trajectories"
DEFAULT_OUTPUT_DIR = Path("data/open_trajectories/swesmith")


def main() -> None:
    parser = argparse.ArgumentParser(description="抽样检查 SWE-smith trajectories 的真实字段结构。")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HuggingFace dataset id。")
    parser.add_argument("--split", default="tool", help="要读取的数据 split。SWE-smith 当前包含 tool/xml/ticks。")
    parser.add_argument(
        "--local-data-files",
        default="",
        help="本地 parquet glob，例如 data/open_trajectories/swesmith/raw/data/tool-*.parquet。设置后不访问远端 dataset。",
    )
    parser.add_argument("--limit", type=int, default=5, help="保存的样本数量。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录。")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="关闭 streaming。默认使用 streaming，避免为了看 schema 就下载完整数据。",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "缺少依赖 datasets。请先在项目虚拟环境中运行：\n"
            "pip install -r requirements.txt"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.local_data_files:
        dataset = load_dataset(
            "parquet",
            data_files=args.local_data_files,
            split="train",
            streaming=not args.no_streaming,
        )
    else:
        dataset = load_dataset(args.dataset, split=args.split, streaming=not args.no_streaming)

    samples = []
    for index, item in enumerate(dataset):
        if index >= args.limit:
            break
        samples.append(_jsonable(item))

    if not samples:
        raise SystemExit("没有读取到样本，请检查 dataset/split 是否正确。")

    schema = _schema(samples[0])
    sample_path = args.output_dir / "sample.jsonl"
    schema_path = args.output_dir / "schema.json"

    with sample_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved_samples={len(samples)}")
    print(f"sample_path={sample_path}")
    print(f"schema_path={schema_path}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _schema(value: Any, depth: int = 0) -> Any:
    if depth >= 4:
        return type(value).__name__
    if isinstance(value, Mapping):
        return {str(key): _schema(item, depth + 1) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return []
        return [_schema(value[0], depth + 1)]
    return type(value).__name__


if __name__ == "__main__":
    main()
