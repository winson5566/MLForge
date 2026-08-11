"""Compare live Canary prediction mix with the immutable training release."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import mlflow


def prom(query: str) -> list[dict]:
    url = f"{os.environ['PROMETHEUS_URL']}/api/v1/query?{urlencode({'query': query})}"
    with urlopen(url, timeout=30) as response:  # noqa: S310 - in-cluster URL
        body = json.loads(response.read())
    return body["data"]["result"]


def value(query: str) -> float:
    results = prom(query)
    return sum(float(item["value"][1]) for item in results)


def jsd(reference: dict[str, int], observed: dict[str, float]) -> float:
    labels = set(reference) | set(observed)
    r_total, o_total = sum(reference.values()), sum(observed.values())
    if not r_total or not o_total:
        return 0.0
    divergence = 0.0
    for label in labels:
        p, q = reference.get(label, 0) / r_total, observed.get(label, 0) / o_total
        midpoint = (p + q) / 2
        if p:
            divergence += 0.5 * p * math.log2(p / midpoint)
        if q:
            divergence += 0.5 * q * math.log2(q / midpoint)
    return divergence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=Path, required=True)
    parser.add_argument("--minimum-requests", type=int, default=8)
    parser.add_argument("--maximum-jsd", type=float, default=0.35)
    parser.add_argument("--maximum-empty-rate", type=float, default=0.75)
    parser.add_argument("--result-path", type=Path, required=True)
    args = parser.parse_args()
    baseline = Counter(json.loads(args.baseline_path.read_text())["class_counts"])
    request_count = value('sum(mlforge_yolo_canary_requests_total{status="success"})')
    empty_count = value("sum(mlforge_yolo_canary_empty_predictions_total)")
    observed = {item["metric"].get("class_name", "unknown"): float(item["value"][1]) for item in prom("sum by (class_name) (mlforge_yolo_canary_detections_total)")}
    empty_rate = empty_count / request_count if request_count else 0.0
    divergence = jsd(dict(baseline), observed)
    enough = request_count >= args.minimum_requests
    drifted = enough and (divergence > args.maximum_jsd or empty_rate > args.maximum_empty_rate)
    result = {"result": "needs_labeling" if drifted else "healthy" if enough else "insufficient_data", "request_count": request_count, "empty_rate": empty_rate, "class_jsd": divergence, "baseline_classes": dict(baseline), "observed_classes": observed}
    args.result_path.write_text(json.dumps(result, indent=2))
    mlflow.set_experiment("label-studio-prediction-monitoring")
    with mlflow.start_run(run_name="yolo-local-canary-drift"):
        mlflow.log_metrics({"request_count": request_count, "empty_rate": empty_rate, "class_jsd": divergence})
        mlflow.set_tag("result", result["result"])
        mlflow.log_artifact(str(args.result_path), artifact_path="drift")
    print(json.dumps(result, sort_keys=True))
    if drifted:
        raise SystemExit("prediction drift requires new labeling")


if __name__ == "__main__":
    main()
