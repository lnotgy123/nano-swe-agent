from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import docker

from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.docker_utils import clean_images, list_images
from swebench.harness.run_evaluation import run_instances
from swebench.harness.test_spec.test_spec import TestSpec


def load_predictions(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row[KEY_INSTANCE_ID]: row for row in rows}


def load_specs(path: Path, prediction_ids: set[str]) -> list[TestSpec]:
    rows = json.loads(path.read_text())
    specs = [TestSpec(**row) for row in rows if row["instance_id"] in prediction_ids]
    return specs


def write_report(
    *,
    report_dir: Path,
    run_id: str,
    predictions: dict[str, dict],
    specs: list[TestSpec],
    client: docker.DockerClient,
) -> Path:
    completed_ids: set[str] = set()
    resolved_ids: set[str] = set()
    unresolved_ids: set[str] = set()
    error_ids: set[str] = set()
    empty_patch_ids: set[str] = set()

    for spec in specs:
        pred = predictions[spec.instance_id]
        if pred.get(KEY_PREDICTION) in ["", None]:
            empty_patch_ids.add(spec.instance_id)
            continue
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / pred[KEY_MODEL].replace("/", "__")
            / spec.instance_id
            / LOG_REPORT
        )
        if not report_file.exists():
            error_ids.add(spec.instance_id)
            continue
        try:
            report = json.loads(report_file.read_text())
            completed_ids.add(spec.instance_id)
            if report[spec.instance_id]["resolved"]:
                resolved_ids.add(spec.instance_id)
            else:
                unresolved_ids.add(spec.instance_id)
        except Exception:
            error_ids.add(spec.instance_id)

    images = list_images(client)
    unremoved_images = sorted(
        spec.instance_image_key for spec in specs if spec.instance_image_key in images
    )
    containers = client.containers.list(all=True)
    unstopped_containers = sorted(c.name for c in containers if run_id in c.name)

    report = {
        "total_instances": len(specs),
        "submitted_instances": len(predictions),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "completed_ids": sorted(completed_ids),
        "submitted_ids": sorted(predictions),
        "resolved_ids": sorted(resolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "empty_patch_ids": sorted(empty_patch_ids),
        "error_ids": sorted(error_ids),
        "unstopped_instances": len(unstopped_containers),
        "unstopped_containers": unstopped_containers,
        "unremoved_images": unremoved_images,
        "schema_version": 2,
    }
    report_path = report_dir / f"{next(iter(predictions.values()))[KEY_MODEL].replace('/', '__')}.{run_id}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"Total instances: {report['total_instances']}")
    print(f"Instances submitted: {report['submitted_instances']}")
    print(f"Instances completed: {report['completed_instances']}")
    print(f"Instances resolved: {report['resolved_instances']}")
    print(f"Instances unresolved: {report['unresolved_instances']}")
    print(f"Instances with empty patches: {report['empty_patch_instances']}")
    print(f"Instances with errors: {report['error_instances']}")
    print(f"Report written to {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SWE-bench official local evaluation with cached TestSpec JSON."
    )
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--test-specs-path", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cache-level", choices=["none", "base", "env", "instance"], default="instance")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    predictions_path = Path(args.predictions_path).resolve()
    test_specs_path = Path(args.test_specs_path).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(report_dir)

    predictions = load_predictions(predictions_path)
    specs = load_specs(test_specs_path, set(predictions))
    missing = sorted(set(predictions) - {spec.instance_id for spec in specs})
    if missing:
        raise SystemExit(f"Missing cached TestSpecs for predictions: {missing}")

    client = docker.from_env()
    existing_images = list_images(client)
    run_instances(
        predictions=predictions,
        instances=specs,
        cache_level=args.cache_level,
        clean=args.clean,
        force_rebuild=args.force_rebuild,
        max_workers=args.max_workers,
        run_id=args.run_id,
        timeout=args.timeout,
        rewrite_reports=False,
    )
    clean_images(client, existing_images, args.cache_level, args.clean)
    write_report(
        report_dir=report_dir,
        run_id=args.run_id,
        predictions=predictions,
        specs=specs,
        client=client,
    )


if __name__ == "__main__":
    main()
