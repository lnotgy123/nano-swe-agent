from __future__ import annotations

import glob
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ExportStats:
    scanned: int
    exported: int
    skipped_empty_patch: int
    deduplicated: int
    output_file: str


@dataclass(frozen=True)
class ImportStats:
    trajectories_scanned: int
    trajectories_updated: int
    results_loaded: int
    output_dir: str | None


def export_predictions(
    input_glob: str,
    output_file: str | Path,
    model_name_or_path: str,
    require_patch: bool = True,
    dedupe: str = "latest",
) -> ExportStats:
    paths = sorted(Path(path) for path in glob.glob(input_glob, recursive=True))
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(_iter_jsonl(paths))
    scanned = len(records)
    selected = _select_records(records, dedupe)
    deduplicated = scanned - len(selected)

    exported = 0
    skipped_empty_patch = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in selected:
            patch = extract_patch(record)
            if require_patch and not patch.strip():
                skipped_empty_patch += 1
                continue

            instance_id = record.get("run_name") or record.get("instance_id") or record.get("run_id")
            if not instance_id:
                raise ValueError("Trajectory record is missing run_name/instance_id/run_id.")

            prediction = {
                "instance_id": instance_id,
                "model_name_or_path": model_name_or_path,
                "model_patch": patch,
            }
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            exported += 1

    return ExportStats(
        scanned=scanned,
        exported=exported,
        skipped_empty_patch=skipped_empty_patch,
        deduplicated=deduplicated,
        output_file=str(output_path),
    )


def import_official_results(
    trajectory_glob: str,
    results_path: str | Path,
    output_dir: str | Path | None = None,
    in_place: bool = False,
) -> ImportStats:
    result_map = load_result_map(results_path)
    trajectory_paths = sorted(Path(path) for path in glob.glob(trajectory_glob, recursive=True))
    if not trajectory_paths:
        raise FileNotFoundError(f"No trajectory files matched: {trajectory_glob}")

    output_root = Path(output_dir).resolve() if output_dir else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    scanned = 0
    updated = 0
    for path in trajectory_paths:
        records = list(_iter_jsonl([path]))
        if not records:
            continue

        changed = False
        for record in records:
            scanned += 1
            instance_id = record.get("run_name") or record.get("instance_id") or record.get("run_id")
            if instance_id not in result_map:
                continue
            official = result_map[instance_id]
            record["official_evaluation"] = {
                "resolved": official["resolved"],
                "source": str(Path(results_path).resolve()),
            }
            if "raw" in official:
                record["official_evaluation"]["raw"] = official["raw"]
            changed = True
            updated += 1

        if not changed:
            continue

        target = path if in_place else _output_path(path, output_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not in_place and target != path:
            shutil.copy2(path, target)
        _write_jsonl(target, records)

    return ImportStats(
        trajectories_scanned=scanned,
        trajectories_updated=updated,
        results_loaded=len(result_map),
        output_dir=str(output_root) if output_root else None,
    )


def load_result_map(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path).resolve()
    records: list[Any] = []
    if source.is_dir():
        for candidate in sorted(source.rglob("*.json")) + sorted(source.rglob("*.jsonl")):
            records.extend(_read_json_records(candidate))
    else:
        records.extend(_read_json_records(source))

    result_map: dict[str, dict[str, Any]] = {}
    for record in records:
        _collect_results(record, result_map)
    return result_map


def extract_patch(record: dict[str, Any]) -> str:
    evaluation_patch = record.get("evaluation", {}).get("patch")
    if isinstance(evaluation_patch, str):
        return evaluation_patch

    final_diff = record.get("final", {}).get("diff")
    if not isinstance(final_diff, str):
        return ""
    return _extract_stdout(final_diff)


def _select_records(records: list[dict[str, Any]], dedupe: str) -> list[dict[str, Any]]:
    if dedupe == "none":
        return records
    if dedupe != "latest":
        raise ValueError("dedupe must be one of: latest, none")

    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        instance_id = record.get("run_name") or record.get("instance_id") or record.get("run_id")
        if not instance_id:
            raise ValueError("Trajectory record is missing run_name/instance_id/run_id.")
        current = selected.get(str(instance_id))
        if current is None or str(record.get("created_at", "")) >= str(current.get("created_at", "")):
            selected[str(instance_id)] = record
    return list(selected.values())


def _collect_results(value: Any, result_map: dict[str, dict[str, Any]]) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_results(item, result_map)
        return

    if not isinstance(value, dict):
        return

    resolved_ids = value.get("resolved_ids")
    if isinstance(resolved_ids, list):
        for instance_id in resolved_ids:
            result_map[str(instance_id)] = {"resolved": True}

    unresolved_ids = value.get("unresolved_ids")
    if isinstance(unresolved_ids, list):
        for instance_id in unresolved_ids:
            result_map[str(instance_id)] = {"resolved": False}

    instance_id = _instance_id(value)
    resolved = _resolved_value(value)
    if instance_id is not None and resolved is not None:
        result_map[instance_id] = {"resolved": resolved, "raw": value}

    for key, item in value.items():
        if isinstance(item, dict):
            item_instance_id = _instance_id(item) or (key if "__" in str(key) else None)
            item_resolved = _resolved_value(item)
            if item_instance_id is not None and item_resolved is not None:
                result_map[str(item_instance_id)] = {"resolved": item_resolved, "raw": item}
            else:
                _collect_results(item, result_map)
        elif isinstance(item, list):
            _collect_results(item, result_map)


def _instance_id(record: dict[str, Any]) -> str | None:
    for key in ("instance_id", "instance", "id"):
        value = record.get(key)
        if value:
            return str(value)
    return None


def _resolved_value(record: dict[str, Any]) -> bool | None:
    for key in ("resolved", "is_resolved"):
        value = record.get(key)
        if isinstance(value, bool):
            return value
    for key in ("status", "eval_status"):
        value = record.get(key)
        if isinstance(value, str):
            normalized = value.lower().replace("-", "_")
            if normalized in {"resolved", "pass", "passed", "success"}:
                return True
            if normalized in {"unresolved", "fail", "failed", "error"}:
                return False
    return None


def _extract_stdout(command_output: str) -> str:
    marker = "\nstdout:\n"
    if marker not in command_output:
        return command_output
    stdout = command_output.split(marker, 1)[1]
    stderr_marker = "\nstderr:\n"
    if stderr_marker in stdout:
        stdout = stdout.split(stderr_marker, 1)[0]
    return stdout


def _read_json_records(path: Path) -> list[Any]:
    if path.suffix == ".jsonl":
        return list(_iter_jsonl([path]))
    with path.open("r", encoding="utf-8") as f:
        return [json.load(f)]


def _iter_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected object record in {path}:{line_no}.")
                yield record


def _output_path(path: Path, output_root: Path | None) -> Path:
    if output_root is None:
        return path
    return output_root / path.name


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
