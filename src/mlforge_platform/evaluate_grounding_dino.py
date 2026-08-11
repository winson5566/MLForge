"""Evaluate Grounding DINO zero-shot detection against a Label Studio release."""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import boto3
import mlflow
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def active_results(task: dict) -> list[dict]:
    for annotation in task.get("annotations", []):
        if not annotation.get("was_cancelled", False):
            return annotation.get("result", [])
    return []


def s3_image(task: dict) -> tuple[str, str]:
    parsed = urlparse(task.get("data", {}).get("image", ""))
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"task {task.get('id')} has no s3 image URI")
    return parsed.netloc, parsed.path.lstrip("/")


def iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def average_precision(predictions: list[dict], ground_truth: dict[str, list[list[float]]]) -> float:
    total = sum(len(boxes) for boxes in ground_truth.values())
    if total == 0:
        return 0.0
    matched: dict[str, set[int]] = defaultdict(set)
    true_positives: list[float] = []
    false_positives: list[float] = []
    for prediction in sorted(predictions, key=lambda value: value["score"], reverse=True):
        boxes = ground_truth.get(prediction["task_id"], [])
        best_index, best_iou = -1, 0.0
        for index, box in enumerate(boxes):
            if index not in matched[prediction["task_id"]] and iou(prediction["box"], box) > best_iou:
                best_index, best_iou = index, iou(prediction["box"], box)
        if best_index >= 0 and best_iou >= 0.5:
            matched[prediction["task_id"]].add(best_index)
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)
    precision, recall, ap = [], [], 0.0
    tp, fp = 0.0, 0.0
    for true_positive, false_positive in zip(true_positives, false_positives):
        tp += true_positive
        fp += false_positive
        precision.append(tp / (tp + fp))
        recall.append(tp / total)
    previous_recall = 0.0
    for index, current_recall in enumerate(recall):
        ap += precision[index] * max(0.0, current_recall - previous_recall)
        previous_recall = current_recall
    return ap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--result-path", required=True, type=Path)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    args = parser.parse_args()

    manifest = json.loads((args.release_dir / "manifest.json").read_text())
    tasks = json.loads((args.release_dir / "annotations.json").read_text())
    labels = sorted({
        label for task in tasks for result in active_results(task)
        for label in result.get("value", {}).get("rectanglelabels", [])
    })
    prompt = " . ".join(labels) + " ."
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id)
    model.eval()
    client = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    predictions: dict[str, list[dict]] = defaultdict(list)
    truths: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    evaluation_started = time.monotonic()

    for task in tasks:
        task_id = str(task["id"])
        bucket, key = s3_image(task)
        response = client.get_object(Bucket=bucket, Key=key)
        image = Image.open(io.BytesIO(response["Body"].read())).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=args.box_threshold,
            text_threshold=args.text_threshold, target_sizes=[(image.height, image.width)],
        )[0]
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            normalized = str(label).strip().lower()
            if normalized in labels:
                predictions[normalized].append({
                    "task_id": task_id,
                    "box": [float(value) for value in box.tolist()],
                    "score": float(score),
                })
        for annotation in active_results(task):
            value = annotation.get("value", {})
            for label in value.get("rectanglelabels", []):
                truths[label][task_id].append([
                    float(value["x"]) * image.width / 100,
                    float(value["y"]) * image.height / 100,
                    float(value["x"] + value["width"]) * image.width / 100,
                    float(value["y"] + value["height"]) * image.height / 100,
                ])

    per_class = {label: average_precision(predictions[label], truths[label]) for label in labels}
    map50 = sum(per_class.values()) / len(per_class)
    evaluation_seconds = time.monotonic() - evaluation_started
    mlflow.set_experiment("label-studio-grounding-dino-evaluation")
    with mlflow.start_run(run_name=f"grounding-dino-{manifest['release_id']}") as run:
        mlflow.set_tags({"dataset_version": manifest["release_id"], "git_sha": args.git_sha, "evaluation_kind": "zero-shot-grounding-dino"})
        mlflow.log_params({"model_id": args.model_id, "class_count": len(labels), "box_threshold": args.box_threshold, "text_threshold": args.text_threshold})
        mlflow.log_metrics({"zero_shot_map50": map50, "evaluation_seconds": evaluation_seconds})
        report = {"dataset_version": manifest["release_id"], "model_id": args.model_id, "prompt": prompt, "zero_shot_map50": map50, "evaluation_seconds": evaluation_seconds, "per_class_ap50": per_class}
        report_path = args.result_path.parent / "grounding-dino-report.json"
        report_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(report_path))
    args.result_path.write_text(json.dumps({"run_id": run.info.run_id, **report}, indent=2))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
