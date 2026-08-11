"""Promote an MLflow model only when its declared metric meets a policy."""
from __future__ import annotations

import argparse
import json
import os

from mlflow import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--minimum", type=float, required=True)
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    candidate = client.get_model_version_by_alias(args.model_name, "candidate")
    if str(candidate.version) != str(args.version):
        raise SystemExit(
            f"promotion rejected: {args.model_name}@candidate is version "
            f"{candidate.version}, not requested version {args.version}"
        )
    run = client.get_run(candidate.run_id)
    if args.metric_name not in run.data.metrics:
        raise SystemExit(f"promotion rejected: metric {args.metric_name} is absent from candidate run")
    metric_value = run.data.metrics[args.metric_name]
    if metric_value < args.minimum:
        raise SystemExit(
            f"quality gate rejected: {args.metric_name}={metric_value} < {args.minimum}"
        )

    client.set_registered_model_alias(args.model_name, args.alias, args.version)
    print(json.dumps({
        "model_name": args.model_name,
        "version": str(args.version),
        "source_alias": "candidate",
        "target_alias": args.alias,
        "metric_name": args.metric_name,
        "metric_value": metric_value,
        "minimum": args.minimum,
        "result": "promoted",
    }))


if __name__ == "__main__":
    main()
