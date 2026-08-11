"""Promote an MLflow model only when its declared metric meets a policy."""
from __future__ import annotations

import argparse
import os

import mlflow
from mlflow import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--metric-value", type=float, required=True)
    parser.add_argument("--minimum", type=float, required=True)
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    if args.metric_value < args.minimum:
        raise SystemExit(
            f"quality gate rejected: {args.metric_name}={args.metric_value} < {args.minimum}"
        )

    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    client.set_registered_model_alias(args.model_name, args.alias, args.version)
    print(f"promoted {args.model_name}@{args.alias} -> version {args.version}")


if __name__ == "__main__":
    main()
