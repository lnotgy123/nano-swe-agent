from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data/llamafactory/swesmith_official_xml_resolved_32k.sharegpt.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/llamafactory/swesmith_stage2_key_actions_16k.sharegpt.jsonl"
DEFAULT_STATS = PROJECT_ROOT / "data/official_sft/swesmith_stage2_key_actions_16k.stats.json"
DEFAULT_MODEL = Path(
    os.environ.get(
        "QWEN_MODEL_PATH",
        str(PROJECT_ROOT / "Qwen2.5-Coder-7B-Instruct"),
    )
)

CATEGORY_RATIOS = {
    "source_edit": 0.45,
    "validation": 0.20,
    "recovery": 0.25,
    "submit": 0.05,
    "localization": 0.05,
}
EDITOR_FAILURE_MARKERS = (
    "no replacement was performed",
    "did not appear verbatim",
    "multiple occurrences",
    "invalid command",
    "invalid path",
    "file not found",
    "does not exist",
)
TEST_FAILURE_MARKERS = (
    "traceback (most recent call last)",
    "test failed",
    "tests failed",
    "failed =",
    " failed,",
    " failures ",
    " errors ",
    "assertionerror",
    "syntaxerror",
    "indentationerror",
)
REPO_TEST_RE = re.compile(r"(?:pytest|unittest|tox(?:\s|$)|nox(?:\s|$)|runtests)", re.IGNORECASE)
DIFF_SYNTAX_RE = re.compile(r"(?:py_compile|compileall|git\s+diff)", re.IGNORECASE)
REPRO_RE = re.compile(r"python\s+[^\n]*(?:repro|debug|check|test)[^\n]*\.py", re.IGNORECASE)
LOCALIZATION_RE = re.compile(r"(?:^|[;&|\s])(?:rg|grep|find)\s", re.IGNORECASE)
ACTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)


@dataclass(frozen=True)
class Action:
    tool: str
    params: dict[str, str]

    @property
    def signature(self) -> str:
        return json.dumps([self.tool, sorted(self.params.items())], ensure_ascii=False)


@dataclass(frozen=True)
class Candidate:
    line_no: int
    turn_index: int
    category: str
    subtype: str = ""
    trajectory_key: str = ""


def parse_action(text: str) -> Action | None:
    match = ACTION_RE.search(text)
    if not match:
        return None
    return Action(
        tool=match.group(1).strip(),
        params={name.strip(): value.strip() for name, value in PARAM_RE.findall(match.group(2))},
    )


def is_source_path(path: str) -> bool:
    normalized = path.lower().replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    if not normalized or normalized.endswith("/"):
        return False
    if any(part in normalized for part in ("/tests/", "/test/", "/testing/")):
        return False
    if name.startswith(("test_", "repro", "debug")) or name.endswith(("_test.py", ".patch", ".diff")):
        return False
    if name in {"patch.txt", "tmp.py", "temp.py"}:
        return False
    return normalized.startswith("/testbed/")


def is_source_edit(action: Action | None) -> bool:
    if action is None or action.tool != "str_replace_editor":
        return False
    return action.params.get("command") in {"str_replace", "insert", "create"} and is_source_path(
        action.params.get("path", "")
    )


def has_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def validation_subtype(action: Action | None) -> str | None:
    if action is None or action.tool != "bash":
        return None
    command = action.params.get("command", "")
    if REPO_TEST_RE.search(command):
        return "repo_test"
    if DIFF_SYNTAX_RE.search(command):
        return "diff_syntax"
    if REPRO_RE.search(command):
        return "repro_script"
    return None


def is_source_view(action: Action | None) -> bool:
    return bool(
        action
        and action.tool == "str_replace_editor"
        and action.params.get("command") == "view"
        and is_source_path(action.params.get("path", ""))
    )


def is_recovery_action(previous_action: Action | None, observation: str, action: Action) -> bool:
    target_is_source_action = is_source_edit(action) or is_source_view(action)
    if not target_is_source_action or previous_action is None:
        return False
    if previous_action.tool == "str_replace_editor":
        return has_marker(observation, EDITOR_FAILURE_MARKERS)
    if validation_subtype(previous_action):
        return has_marker(observation, TEST_FAILURE_MARKERS)
    return False


def classify_candidates(
    conversations: list[dict[str, str]], line_no: int, trajectory_key: str | None = None
) -> tuple[list[Candidate], int]:
    candidates: list[Candidate] = []
    source_edit_seen = False
    submit_seen = False
    previous_action: Action | None = None
    previous_observation = ""
    rejected_repeats = 0

    for index, turn in enumerate(conversations):
        role = turn.get("from")
        text = turn.get("value", "")
        if role != "gpt":
            if role in {"human", "observation"}:
                previous_observation = text
            continue

        action = parse_action(text)
        if action is None:
            previous_observation = ""
            continue

        failed = has_marker(previous_observation, EDITOR_FAILURE_MARKERS + TEST_FAILURE_MARKERS)
        repeated_failure = failed and previous_action is not None and action.signature == previous_action.signature
        category: str | None = None
        subtype = ""

        if not repeated_failure and is_recovery_action(previous_action, previous_observation, action):
            category = "recovery"
            subtype = "editor_failure" if previous_action and previous_action.tool == "str_replace_editor" else "test_failure"
        elif repeated_failure:
            rejected_repeats += 1
        elif is_source_edit(action):
            category = "source_edit"
            subtype = action.params.get("command", "")
        elif action.tool == "submit" and source_edit_seen and not failed:
            category = "submit"
        elif action.tool == "bash":
            command = action.params.get("command", "")
            subtype = validation_subtype(action) or ""
            if source_edit_seen and subtype:
                category = "validation"
            elif not source_edit_seen and LOCALIZATION_RE.search(command):
                category = "localization"
        elif (
            action.tool == "str_replace_editor"
            and action.params.get("command") == "view"
            and not source_edit_seen
            and is_source_path(action.params.get("path", ""))
        ):
            category = "localization"

        if category:
            candidates.append(
                Candidate(
                    line_no=line_no,
                    turn_index=index,
                    category=category,
                    subtype=subtype,
                    trajectory_key=trajectory_key or str(line_no),
                )
            )

        if is_source_edit(action):
            source_edit_seen = True
        if action.tool == "submit":
            submit_seen = True
        previous_action = action
        previous_observation = ""

    return candidates, rejected_repeats


def select_candidates(
    candidates: list[Candidate], total: int, seed: int, max_per_trajectory: int = 2
) -> list[Candidate]:
    rng = random.Random(seed)
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.category].append(candidate)
    for values in grouped.values():
        rng.shuffle(values)

    selected: list[Candidate] = []
    selected_set: set[Candidate] = set()
    trajectory_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    def take(pool: list[Candidate], count: int) -> None:
        for candidate in pool:
            if category_counts[candidate.category] >= count:
                break
            if candidate in selected_set or trajectory_counts[candidate.trajectory_key] >= max_per_trajectory:
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            trajectory_counts[candidate.trajectory_key] += 1
            category_counts[candidate.category] += 1

    for category, ratio in CATEGORY_RATIOS.items():
        quota = round(total * ratio)
        if category == "validation":
            subtype_ratios = {"repo_test": 0.50, "diff_syntax": 0.25, "repro_script": 0.25}
            subtype_pools: dict[str, list[Candidate]] = {}
            for subtype, subtype_ratio in subtype_ratios.items():
                pool = [item for item in grouped[category] if item.subtype == subtype]
                subtype_pools[subtype] = pool
                take(pool, category_counts[category] + round(quota * subtype_ratio))
            # The official trajectories contain very few explicit git diff or
            # syntax-check actions. Fill that shortage with real repository
            # tests before allowing more reproduction scripts.
            take(
                subtype_pools["repo_test"]
                + subtype_pools["diff_syntax"]
                + subtype_pools["repro_script"],
                quota,
            )
        else:
            take(grouped[category], quota)

    if len(selected) < total:
        leftovers = [candidate for candidate in candidates if candidate not in selected_set]
        rng.shuffle(leftovers)
        for candidate in leftovers:
            if len(selected) >= total:
                break
            if trajectory_counts[candidate.trajectory_key] < max_per_trajectory:
                selected.append(candidate)
                trajectory_counts[candidate.trajectory_key] += 1
                category_counts[candidate.category] += 1
    rng.shuffle(selected)
    return selected[:total]


def load_tokenizer(path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


def count_tokens(tokenizer: Any, system: str, conversations: list[dict[str, str]]) -> int:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    for turn in conversations:
        role = "assistant" if turn["from"] == "gpt" else "user"
        messages.append({"role": role, "content": turn.get("value", "")})
    try:
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    except Exception:
        text = system + "\n" + "\n".join(turn.get("value", "") for turn in conversations)
        return len(tokenizer.encode(text, add_special_tokens=False))


def trim_prefix(
    tokenizer: Any,
    system: str,
    conversations: list[dict[str, str]],
    max_tokens: int,
    min_history_pairs: int = 0,
) -> tuple[list[dict[str, str]], int, bool]:
    original = [dict(turn) for turn in conversations]
    token_count = count_tokens(tokenizer, system, original)
    if token_count <= max_tokens:
        return original, token_count, False

    # Preserve the issue and target. Binary-search the minimum number of oldest
    # action/observation pairs to remove; long prefixes otherwise require many
    # expensive tokenizer passes.
    max_pairs = max(0, (len(original) - 2) // 2 - min_history_pairs)
    if max_pairs == 0:
        raise ValueError(f"prefix cannot fit in {max_tokens} tokens while preserving causal history")
    low, high = 1, max_pairs
    best: tuple[list[dict[str, str]], int] | None = None
    while low <= high:
        removed_pairs = (low + high) // 2
        prefix = [original[0], *original[1 + 2 * removed_pairs :]]
        current_tokens = count_tokens(tokenizer, system, prefix)
        if current_tokens <= max_tokens:
            best = (prefix, current_tokens)
            high = removed_pairs - 1
        else:
            low = removed_pairs + 1

    if best is None:
        raise ValueError(f"prefix cannot fit in {max_tokens} tokens")
    prefix, token_count = best
    if not prefix or prefix[-1].get("from") != "gpt":
        raise ValueError("prefix does not end with an assistant target")
    return prefix, token_count, True


def summary(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "max": None}
    values = sorted(values)
    at = lambda q: values[min(len(values) - 1, round((len(values) - 1) * q))]
    return {"min": values[0], "p50": at(0.5), "p90": at(0.9), "max": values[-1]}


def register_dataset(dataset_info_path: Path, dataset_name: str, file_name: str) -> None:
    info = json.loads(dataset_info_path.read_text(encoding="utf-8")) if dataset_info_path.exists() else {}
    info[dataset_name] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "observation_tag": "observation",
            "system_tag": "system",
        },
    }
    dataset_info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage-2 key-action prefixes from resolved SWE-smith XML trajectories.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-name", default="swesmith_stage2_key_actions_16k")
    parser.add_argument("--total", type=int, default=5000)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidates: list[Candidate] = []
    rejected_repeats = 0
    scanned = 0
    with args.source.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source):
            record = json.loads(line)
            metadata = record.get("metadata") or {}
            trajectory_key = str(metadata.get("traj_id") or metadata.get("instance_id") or line_no)
            found, rejected = classify_candidates(record["conversations"], line_no, trajectory_key)
            candidates.extend(found)
            rejected_repeats += rejected
            scanned += 1

    selected = select_candidates(candidates, args.total + 50, args.seed)
    category_limits = {category: round(args.total * ratio) for category, ratio in CATEGORY_RATIOS.items()}
    selected_by_line: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in selected:
        selected_by_line[candidate.line_no].append(candidate)

    tokenizer = load_tokenizer(args.tokenizer_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    token_counts: list[int] = []
    output_counts: Counter[str] = Counter()
    output_subtypes: Counter[str] = Counter()
    source_trajectories: set[str] = set()
    trimmed_count = 0
    skipped_overlength = 0

    with args.source.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as output:
        for line_no, line in enumerate(source):
            if line_no not in selected_by_line:
                continue
            record = json.loads(line)
            for candidate in selected_by_line[line_no]:
                if output_counts[candidate.category] >= category_limits[candidate.category]:
                    continue
                try:
                    prefix, token_count, trimmed = trim_prefix(
                        tokenizer,
                        record.get("system", ""),
                        record["conversations"][: candidate.turn_index + 1],
                        args.max_tokens,
                        min_history_pairs=1 if candidate.category in {"recovery", "validation"} else 0,
                    )
                except ValueError:
                    skipped_overlength += 1
                    continue
                metadata = dict(record.get("metadata") or {})
                metadata.update(
                    {
                        "stage2_category": candidate.category,
                        "stage2_subtype": candidate.subtype,
                        "target_turn_index": candidate.turn_index,
                        "prefix_token_count": token_count,
                        "prefix_trimmed": trimmed,
                    }
                )
                output.write(
                    json.dumps(
                        {"conversations": prefix, "system": record.get("system", ""), "metadata": metadata},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_counts[candidate.category] += 1
                output_subtypes[f"{candidate.category}:{candidate.subtype}"] += 1
                token_counts.append(token_count)
                trimmed_count += int(trimmed)
                source_trajectories.add(str(metadata.get("traj_id") or metadata.get("instance_id") or line_no))

    stats = {
        "source": str(args.source),
        "output": str(args.output),
        "scanned_trajectories": scanned,
        "candidate_count": len(candidates),
        "candidate_categories": dict(Counter(candidate.category for candidate in candidates)),
        "candidate_subtypes": dict(Counter(f"{candidate.category}:{candidate.subtype}" for candidate in candidates)),
        "requested_samples": args.total,
        "written_samples": sum(output_counts.values()),
        "selected_categories": dict(output_counts),
        "selected_subtypes": dict(output_subtypes),
        "unique_source_trajectories": len(source_trajectories),
        "rejected_exact_repeats_after_failure": rejected_repeats,
        "trimmed_prefixes": trimmed_count,
        "skipped_overlength": skipped_overlength,
        "max_tokens": args.max_tokens,
        "max_samples_per_trajectory": 2,
        "token_count_summary": summary(token_counts),
        "seed": args.seed,
    }
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    register_dataset(args.output.parent / "dataset_info.json", args.dataset_name, args.output.name)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
