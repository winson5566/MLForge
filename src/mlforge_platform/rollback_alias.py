"""Move a local Canary alias back to its recorded predecessor."""
from __future__ import annotations

import argparse
import json
import os

from mlflow import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--alias", default="demo-champion")
    args = parser.parse_args()
    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    previous_alias = f"{args.alias}-previous"
    current = client.get_model_version_by_alias(args.model_name, args.alias)
    try:
        previous = client.get_model_version_by_alias(args.model_name, previous_alias)
    except Exception:
        print(json.dumps({"result": "skipped", "reason": "no_previous_version", "current_version": str(current.version)}))
        return
    client.set_registered_model_alias(args.model_name, args.alias, previous.version)
    print(json.dumps({"result": "rolled_back", "model_name": args.model_name, "alias": args.alias, "from_version": str(current.version), "to_version": str(previous.version)}))


if __name__ == "__main__":
    main()
