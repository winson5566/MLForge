"""Train a YOLO detector from an immutable Label Studio data release."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from urllib.parse import urlparse

import boto3
import mlflow
import mlflow.pytorch
import torch
from mlflow import MlflowClient
from ultralytics import YOLO, settings

# The workflow owns MLflow lineage and model registration. Disable Ultralytics'
# implicit run so one immutable data release maps to one governed MLflow run.
settings.update({"mlflow": False})


def active_results(task: dict) -> list[dict]:
    for annotation in task.get("annotations", []):
        if not annotation.get("was_cancelled", False):
            return annotation.get("result", [])
    return []


def image_source(task: dict) -> tuple[str, str]:
    image = task.get("data", {}).get("image", "")
    parsed = urlparse(image)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"task {task.get('id')} has no s3 image URI")
    return parsed.netloc, parsed.path.lstrip("/")


def prepare_dataset(release_dir: Path, output: Path) -> tuple[Path, list[str]]:
    tasks = json.loads((release_dir / "annotations.json").read_text())
    labels = sorted({
        label
        for task in tasks
        for result in active_results(task)
        for label in result.get("value", {}).get("rectanglelabels", [])
    })
    if len(tasks) < 2 or len(labels) < 2:
        raise SystemExit("YOLO training needs at least two tasks and two classes")

    class_ids = {label: index for index, label in enumerate(labels)}
    client = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    random.Random(42).shuffle(tasks)
    split = max(1, int(len(tasks) * 0.8))
    for subset, subset_tasks in (("train", tasks[:split]), ("val", tasks[split:])):
        if not subset_tasks:
            subset_tasks = tasks[:1]
        image_dir = output / "images" / subset
        label_dir = output / "labels" / subset
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for task in subset_tasks:
            bucket, key = image_source(task)
            suffix = Path(key).suffix.lower() or ".jpg"
            stem = str(task["id"])
            image_path = image_dir / f"{stem}{suffix}"
            client.download_file(bucket, key, str(image_path))
            rows: list[str] = []
            for result in active_results(task):
                value = result.get("value", {})
                rectangle_labels = value.get("rectanglelabels", [])
                if not rectangle_labels:
                    continue
                try:
                    x, y = float(value["x"]), float(value["y"])
                    width, height = float(value["width"]), float(value["height"])
                except (KeyError, TypeError, ValueError):
                    continue
                label = rectangle_labels[0]
                if label not in class_ids:
                    continue
                rows.append(
                    f"{class_ids[label]} {(x + width / 2) / 100:.6f} "
                    f"{(y + height / 2) / 100:.6f} {width / 100:.6f} {height / 100:.6f}"
                )
            (label_dir / f"{stem}.txt").write_text("\n".join(rows))

    dataset = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "names": labels,
    }
    dataset_path = output / "dataset.yaml"
    dataset_path.write_text("\n".join([
        f"path: {dataset['path']}",
        "train: images/train",
        "val: images/val",
        "names:",
        *[f"  {index}: {label}" for index, label in enumerate(labels)],
        "",
    ]))
    return dataset_path, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--result-path", required=True, type=Path)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    manifest = json.loads((args.release_dir / "manifest.json").read_text())
    dataset_dir = args.result_path.parent / "yolo-dataset"
    dataset_path, classes = prepare_dataset(args.release_dir, dataset_dir)
    # Use COCO-pretrained weights for transfer learning. A `.yaml` file only
    # defines the network and would train from random initialization.
    model = YOLO("yolo11n.pt")
    run_dir = args.result_path.parent / "yolo-runs" / manifest["release_id"]
    model.train(
        data=str(dataset_path), epochs=args.epochs, imgsz=args.imgsz, batch=1,
        device="cpu", workers=0, project=str(args.result_path.parent / "yolo-runs"),
        name=manifest["release_id"], patience=args.patience, exist_ok=True, verbose=False,
    )
    metrics = model.val(data=str(dataset_path), imgsz=args.imgsz, batch=1, device="cpu", workers=0, verbose=False)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    best_weights = run_dir / "weights" / "best.pt"
    registered_model = YOLO(str(best_weights)).model

    mlflow.set_experiment("label-studio-yolo-training")
    with mlflow.start_run(run_name=f"yolo-{manifest['release_id']}") as run:
        mlflow.set_tags({
            "dataset_version": manifest["release_id"],
            "git_sha": args.git_sha,
            "training_kind": "yolo11n-object-detection",
        })
        mlflow.log_params({"epochs": args.epochs, "imgsz": args.imgsz, "patience": args.patience, "batch": 1, "class_count": len(classes)})
        mlflow.log_metrics({"validation_map50": map50, "validation_map50_95": map5095})
        mlflow.log_artifact(str(dataset_path), artifact_path="dataset")
        mlflow.log_artifact(str(best_weights), artifact_path="weights")
        mlflow.pytorch.log_model(
            registered_model,
            name="model",
            input_example=torch.zeros((1, 3, args.imgsz, args.imgsz)),
            serialization_format="pickle",
        )
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", args.model_name)

    MlflowClient().set_registered_model_alias(args.model_name, "candidate", version.version)
    result = {
        "model_name": args.model_name,
        "model_version": version.version,
        "model_alias": "candidate",
        "run_id": run.info.run_id,
        "dataset_version": manifest["release_id"],
        "git_sha": args.git_sha,
        "metrics": {"validation_map50": map50, "validation_map50_95": map5095},
    }
    args.result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
