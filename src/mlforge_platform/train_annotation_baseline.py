"""Train and register a small PyTorch baseline from a Label Studio release.

This is deliberately an annotation-geometry baseline, not an object detector.
It verifies the production-shaped training path while a project-specific detector
(for example YOLO or Detectron) can be substituted behind the same contract.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from mlflow import MlflowClient
from torch import nn


def examples(tasks: list[dict]) -> tuple[torch.Tensor, list[str]]:
    features: list[list[float]] = []
    labels: list[str] = []
    for task in tasks:
        for annotation in task.get("annotations", []):
            if annotation.get("was_cancelled", False):
                continue
            for result in annotation.get("result", []):
                value = result.get("value", {})
                rectangle_labels = value.get("rectanglelabels", [])
                if not rectangle_labels:
                    continue
                try:
                    features.append([
                        float(value["x"]) / 100.0,
                        float(value["y"]) / 100.0,
                        float(value["width"]) / 100.0,
                        float(value["height"]) / 100.0,
                    ])
                except (KeyError, TypeError, ValueError):
                    continue
                labels.append(str(rectangle_labels[0]))
    if len(features) < 2 or len(set(labels)) < 2:
        raise SystemExit("baseline needs at least two boxes from two classes")
    return torch.tensor(features, dtype=torch.float32), labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--result-path", required=True, type=Path)
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()

    manifest = json.loads((args.release_dir / "manifest.json").read_text())
    tasks = json.loads((args.release_dir / "annotations.json").read_text())
    x, raw_labels = examples(tasks)
    classes = sorted(set(raw_labels))
    lookup = {label: index for index, label in enumerate(classes)}
    y = torch.tensor([lookup[label] for label in raw_labels], dtype=torch.long)

    torch.manual_seed(42)
    random.seed(42)
    order = torch.randperm(len(x))
    split = max(1, int(len(x) * 0.8))
    train_indices, test_indices = order[:split], order[split:]
    if len(test_indices) == 0:
        test_indices = train_indices

    model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, len(classes)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_function = nn.CrossEntropyLoss()
    for _ in range(120):
        optimizer.zero_grad()
        loss = loss_function(model(x[train_indices]), y[train_indices])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        accuracy = (model(x[test_indices]).argmax(dim=1) == y[test_indices]).float().mean().item()
        final_loss = loss_function(model(x[test_indices]), y[test_indices]).item()

    mlflow.set_experiment("label-studio-candidate-training")
    with mlflow.start_run(run_name=f"annotation-baseline-{manifest['release_id']}") as run:
        mlflow.set_tags({
            "dataset_version": manifest["release_id"],
            "git_sha": args.git_sha,
            "training_kind": "annotation-geometry-baseline",
        })
        mlflow.log_params({"epochs": 120, "input_features": 4, "class_count": len(classes)})
        mlflow.log_metrics({"validation_accuracy": accuracy, "validation_loss": final_loss})
        classes_path = args.result_path.parent / "classes.json"
        classes_path.write_text(json.dumps(classes, indent=2))
        mlflow.log_artifact(str(classes_path))
        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=x[:1],
            serialization_format="pickle",
        )
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", args.model_name)

    client = MlflowClient()
    client.set_registered_model_alias(args.model_name, "candidate", version.version)
    result = {
        "model_name": args.model_name,
        "model_version": version.version,
        "model_alias": "candidate",
        "run_id": run.info.run_id,
        "dataset_version": manifest["release_id"],
        "git_sha": args.git_sha,
        "metrics": {"validation_accuracy": accuracy, "validation_loss": final_loss},
    }
    args.result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
