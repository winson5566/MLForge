"""Validate and normalize a Label Studio export into an immutable release."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def active_annotations(tasks: list[dict]) -> list[dict]:
    return [
        annotation
        for task in tasks
        for annotation in task.get("annotations", [])
        if not annotation.get("was_cancelled", False)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--release-id-output", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    tasks = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit("Label Studio export must be a JSON list")
    task_ids = [task.get("id") for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise SystemExit("duplicate task IDs in Label Studio export")
    annotations = active_annotations(tasks)
    if not annotations:
        raise SystemExit("no active annotations in Label Studio export")

    canonical = json.dumps(tasks, sort_keys=True, separators=(",", ":")).encode()
    release_id = hashlib.sha256(canonical).hexdigest()[:16]
    target = Path(args.output_root) / f"project-{args.project_id}" / release_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "annotations.json").write_bytes(canonical)

    label_counts: Counter[str] = Counter(
        label
        for annotation in annotations
        for result in annotation.get("result", [])
        for label in result.get("value", {}).get("rectanglelabels", [])
    )
    annotated_task_ids = {
        task.get("id")
        for task in tasks
        if any(not annotation.get("was_cancelled", False) for annotation in task.get("annotations", []))
    }
    manifest = {
        "schema_version": 1,
        "project_id": args.project_id,
        "release_id": release_id,
        "task_count": len(tasks),
        "active_annotation_count": len(annotations),
        "annotated_task_count": len(annotated_task_ids),
        "unannotated_task_count": len(tasks) - len(annotated_task_ids),
        "empty_annotation_count": sum(not annotation.get("result") for annotation in annotations),
        "label_counts": dict(sorted(label_counts.items())),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Path(args.release_id_output).write_text(release_id, encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
