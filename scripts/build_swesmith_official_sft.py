from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "SWE-bench/SWE-smith-trajectories"
DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    str(PROJECT_ROOT / "Qwen2.5-Coder-7B-Instruct"),
)
DEFAULT_OUTPUT_DIR = Path("data/official_sft")
DEFAULT_LLAMAFATORY_DIR = Path("data/llamafactory")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build full-trajectory SWE-smith SFT data close to the public "
            "SWE-agent-LM training recipe: resolved trajectories only, no step chunking. "
            "Use the xml split by default because the student model is trained to emit XML actions, "
            "not Claude/OpenAI function calls."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="xml", choices=["tool", "xml", "ticks"])
    parser.add_argument(
        "--local-data-files",
        default="",
        help="Optional local parquet/json/jsonl glob. If set, the HuggingFace dataset is not used.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--llamafactory-dir", type=Path, default=DEFAULT_LLAMAFATORY_DIR)
    parser.add_argument("--dataset-name", default="swesmith_official_xml_resolved_32k")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument(
        "--rough-char-token-ratio",
        type=float,
        default=6.0,
        help=(
            "Skip tokenization when serialized trajectory chars are clearly too long. "
            "A value of 6 means chars > max_tokens * 6 are treated as overlength."
        ),
    )
    parser.add_argument("--tokenizer-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--limit", type=int, default=0, help="Stop after N accepted samples. 0 means all.")
    parser.add_argument("--scan-limit", type=int, default=0, help="Stop after scanning N raw rows. 0 means all.")
    parser.add_argument("--include-unresolved", action="store_true")
    parser.add_argument("--keep-overlength", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--write-sharegpt", action="store_true", default=True)
    parser.add_argument("--no-write-sharegpt", dest="write_sharegpt", action="store_false")
    parser.add_argument("--write-raw-messages", action="store_true", default=True)
    parser.add_argument("--no-write-raw-messages", dest="write_raw_messages", action="store_false")
    parser.add_argument("--reset-dataset-info", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.llamafactory_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(args.tokenizer_path)
    records = _iter_records(
        dataset=args.dataset,
        split=args.split,
        local_data_files=args.local_data_files,
        streaming=not args.no_streaming,
    )

    raw_path = args.output_dir / f"{args.dataset_name}.messages.jsonl"
    sharegpt_path = args.llamafactory_dir / f"{args.dataset_name}.sharegpt.jsonl"
    stats_path = args.output_dir / f"{args.dataset_name}.stats.json"

    raw_file = raw_path.open("w", encoding="utf-8") if args.write_raw_messages else None
    sharegpt_file = sharegpt_path.open("w", encoding="utf-8") if args.write_sharegpt else None

    stats: dict[str, Any] = {
        "dataset": args.dataset,
        "split": args.split,
        "max_tokens": args.max_tokens,
        "scanned": 0,
        "accepted": 0,
        "skipped_unresolved": 0,
        "skipped_bad_messages": 0,
        "skipped_overlength": 0,
        "models": Counter(),
        "message_counts": [],
        "token_counts": [],
        "raw_messages_file": str(raw_path) if args.write_raw_messages else None,
        "sharegpt_file": str(sharegpt_path) if args.write_sharegpt else None,
    }

    try:
        for record in records:
            stats["scanned"] += 1
            if args.scan_limit and stats["scanned"] > args.scan_limit:
                break

            if not args.include_unresolved and not record.get("resolved"):
                stats["skipped_unresolved"] += 1
                continue

            try:
                messages = _parse_messages(record)
            except (TypeError, ValueError, json.JSONDecodeError):
                stats["skipped_bad_messages"] += 1
                continue

            text = _messages_text(messages)
            rough_char_limit = int(args.max_tokens * args.rough_char_token_ratio)
            if len(text) > rough_char_limit and not args.keep_overlength:
                stats["skipped_overlength"] += 1
                continue

            token_count = _count_tokens(tokenizer, text)
            if token_count is not None and token_count > args.max_tokens and not args.keep_overlength:
                stats["skipped_overlength"] += 1
                continue

            metadata = {
                "instance_id": record.get("instance_id"),
                "traj_id": record.get("traj_id"),
                "model": record.get("model"),
                "resolved": record.get("resolved"),
                "patch_chars": len(record.get("patch") or ""),
                "message_count": len(messages),
                "token_count": token_count,
            }

            if raw_file is not None:
                raw_file.write(
                    json.dumps(
                        {
                            "messages": _to_openai_messages(messages),
                            "metadata": metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if sharegpt_file is not None:
                sharegpt_file.write(
                    json.dumps(
                        _to_sharegpt(messages=messages, metadata=metadata),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            stats["accepted"] += 1
            stats["models"][str(record.get("model"))] += 1
            stats["message_counts"].append(len(messages))
            if token_count is not None:
                stats["token_counts"].append(token_count)

            if args.limit and stats["accepted"] >= args.limit:
                break

            if args.progress_every and stats["scanned"] % args.progress_every == 0:
                print(
                    "progress scanned={scanned} accepted={accepted} unresolved={unresolved} overlength={overlength}".format(
                        scanned=stats["scanned"],
                        accepted=stats["accepted"],
                        unresolved=stats["skipped_unresolved"],
                        overlength=stats["skipped_overlength"],
                    ),
                    flush=True,
                )
    finally:
        if raw_file is not None:
            raw_file.close()
        if sharegpt_file is not None:
            sharegpt_file.close()

    stats_out = _finalize_stats(stats)
    stats_path.write_text(json.dumps(stats_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.write_sharegpt:
        _write_dataset_info(
            dataset_dir=args.llamafactory_dir,
            dataset_name=args.dataset_name,
            file_name=sharegpt_path.name,
            reset=args.reset_dataset_info,
        )

    print(f"scanned={stats_out['scanned']}")
    print(f"accepted={stats_out['accepted']}")
    print(f"skipped_unresolved={stats_out['skipped_unresolved']}")
    print(f"skipped_overlength={stats_out['skipped_overlength']}")
    if args.write_raw_messages:
        print(f"raw_messages={raw_path}")
    if args.write_sharegpt:
        print(f"sharegpt={sharegpt_path}")
    print(f"stats={stats_path}")


def _iter_records(
    dataset: str,
    split: str,
    local_data_files: str,
    streaming: bool,
) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "缺少 datasets。请使用项目环境运行：\n"
            "/environment/miniconda3/envs/swe-agent-lf/bin/python scripts/build_swesmith_official_sft.py"
        ) from exc

    if local_data_files:
        suffix = local_data_files.rsplit(".", 1)[-1].lower()
        loader = "parquet" if suffix == "parquet" else "json"
        return load_dataset(loader, data_files=local_data_files, split="train", streaming=streaming)
    return load_dataset(dataset, split=split, streaming=streaming)


def _load_tokenizer(path: str) -> Any | None:
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        return None
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception:
        return None


def _parse_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if isinstance(messages, str):
        parsed = json.loads(messages)
    else:
        parsed = messages
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("messages is empty or not a list")
    return parsed


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue

        item: dict[str, Any] = {"role": role, "content": _content_to_text(message.get("content"))}

        if role == "assistant" and message.get("tool_calls"):
            item["tool_calls"] = message["tool_calls"]
        if role == "tool":
            tool_call_ids = message.get("tool_call_ids")
            if isinstance(tool_call_ids, list) and tool_call_ids:
                item["tool_call_id"] = tool_call_ids[0]
            elif message.get("tool_call_id"):
                item["tool_call_id"] = message["tool_call_id"]

        normalized.append(item)
    return normalized


def _to_sharegpt(messages: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    system = ""
    conversations: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        text = _content_to_text(message.get("content"))
        if role == "system":
            if not system:
                system = text
        elif role == "user":
            conversations.append({"from": "human", "value": text})
        elif role == "tool":
            conversations.append({"from": "observation", "value": text})
        elif role == "assistant":
            value = text
            action = message.get("action")
            tool_calls = message.get("tool_calls")
            # The xml split already stores the action inside assistant content, e.g.
            # <function=bash>...</function>. The tool split stores function calls
            # separately; keep this fallback only for inspection or ablations.
            if tool_calls:
                value = _append_tool_calls(value, tool_calls)
            elif action:
                value = _append_action(value, action)
            if value.strip():
                conversations.append({"from": "gpt", "value": value})

    return {
        "conversations": conversations,
        "system": system,
        "metadata": metadata,
    }


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _append_tool_calls(text: str, tool_calls: Any) -> str:
    return (
        text.rstrip()
        + "\n\n<tool_calls>\n"
        + json.dumps(tool_calls, ensure_ascii=False)
        + "\n</tool_calls>"
    ).strip()


def _append_action(text: str, action: Any) -> str:
    return (
        text.rstrip()
        + "\n\n<action>\n"
        + str(action)
        + "\n</action>"
    ).strip()


def _messages_text(messages: list[dict[str, Any]]) -> str:
    text = "\n".join(
        f"{message.get('role', '')}: {_content_to_text(message.get('content'))}"
        for message in messages
    )
    for message in messages:
        if message.get("tool_calls"):
            text += "\n" + json.dumps(message["tool_calls"], ensure_ascii=False)
        elif message.get("action"):
            text += "\n" + str(message["action"])
    return text


def _count_tokens(tokenizer: Any | None, text: str) -> int | None:
    if tokenizer is None:
        return max(1, len(text) // 4)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    token_counts = stats.pop("token_counts")
    message_counts = stats.pop("message_counts")
    models = stats.pop("models")
    stats["models"] = dict(models)
    stats["token_count_summary"] = _summary(token_counts)
    stats["message_count_summary"] = _summary(message_counts)
    return stats


def _summary(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(ordered: list[int], q: float) -> int:
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[index]


def _write_dataset_info(dataset_dir: Path, dataset_name: str, file_name: str, reset: bool) -> None:
    path = dataset_dir / "dataset_info.json"
    dataset_info = {} if reset or not path.exists() else json.loads(path.read_text(encoding="utf-8"))
    dataset_info[dataset_name] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "observation_tag": "observation",
            "system_tag": "system",
        },
    }
    path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
