"""Exercise a deployed Canary and enforce serving SLOs with an audit trail."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path
from statistics import quantiles
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import boto3
import mlflow


def active_image(tasks: list[dict]) -> tuple[str, str]:
    for task in tasks:
        if not any(not annotation.get("was_cancelled", False) for annotation in task.get("annotations", [])):
            continue
        uri = task.get("data", {}).get("image", "")
        parsed = urlparse(uri)
        if parsed.scheme == "s3" and parsed.netloc and parsed.path:
            return parsed.netloc, parsed.path.lstrip("/")
    raise SystemExit("no active Label Studio task with an s3 image is available for Canary validation")


def get_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:  # noqa: S310 - in-cluster service URL
        return json.loads(response.read())


def percentile_95(samples: list[float]) -> float:
    if len(samples) == 1:
        return samples[0]
    return quantiles(samples, n=100, method="inclusive")[94]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--minimum-success-rate", type=float, default=1.0)
    parser.add_argument("--maximum-p95-seconds", type=float, default=10.0)
    parser.add_argument("--result-path", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 1:
        raise SystemExit("requests must be positive")

    # Workflows receive the Git-tracked DVC pointer for lineage, whereas DVC
    # materializes the payload in the sibling directory without `.dvc`.
    release_dir = args.release_dir.with_suffix("") if args.release_dir.suffix == ".dvc" else args.release_dir
    tasks = json.loads((release_dir / "annotations.json").read_text())
    bucket, key = active_image(tasks)
    s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    image = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    payload = json.dumps({"image_b64": base64.b64encode(image).decode(), "confidence": 0.25}).encode()
    health = get_json(f"{args.endpoint.rstrip('/')}/health")

    successes, errors, durations = 0, 0, []
    for _ in range(args.requests):
        started = time.monotonic()
        try:
            request = Request(
                f"{args.endpoint.rstrip('/')}/predict", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urlopen(request, timeout=60) as response:  # noqa: S310 - controlled in-cluster endpoint
                json.loads(response.read())
            successes += 1
            durations.append(time.monotonic() - started)
        except Exception as exc:  # Keep test failures in the audit output.
            errors += 1
            print(f"request failed: {exc}")

    success_rate = successes / args.requests
    p95_seconds = percentile_95(durations) if durations else float("inf")
    passed = success_rate >= args.minimum_success_rate and p95_seconds <= args.maximum_p95_seconds
    result = {
        "result": "passed" if passed else "failed",
        "endpoint": args.endpoint,
        "model_name": health.get("model_name"),
        "model_alias": health.get("model_alias"),
        "model_version": health.get("model_version"),
        "requests": args.requests,
        "successes": successes,
        "errors": errors,
        "success_rate": success_rate,
        "p95_seconds": p95_seconds,
        "minimum_success_rate": args.minimum_success_rate,
        "maximum_p95_seconds": args.maximum_p95_seconds,
    }
    mlflow.set_experiment("label-studio-canary-validation")
    with mlflow.start_run(run_name=f"canary-{health.get('model_alias', 'unknown')}-{health.get('model_version', 'unknown')}"):
        mlflow.set_tags({key: str(value) for key, value in result.items() if key in {"model_name", "model_alias", "model_version", "endpoint", "result"}})
        mlflow.log_params({"requests": args.requests, "minimum_success_rate": args.minimum_success_rate, "maximum_p95_seconds": args.maximum_p95_seconds})
        mlflow.log_metrics({"success_rate": success_rate, "p95_latency_seconds": p95_seconds, "request_errors": errors})
        args.result_path.write_text(json.dumps(result, indent=2))
        mlflow.log_artifact(str(args.result_path), artifact_path="canary")
    print(json.dumps(result, sort_keys=True))
    if not passed:
        raise SystemExit("Canary SLO gate rejected the deployment")


if __name__ == "__main__":
    main()
