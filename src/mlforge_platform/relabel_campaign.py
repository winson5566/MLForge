"""Turn an approved prediction-drift demand into a Label Studio task campaign.

The local VOC project already contains its unlabelled source images.  Selecting
existing task IDs is intentional: importing the same S3 URI again would create
duplicate training examples and corrupt data lineage.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from urllib.request import Request, urlopen

from mlforge_platform.pending_release import kubernetes_request
from mlforge_platform.release_gate import access_token


def label_studio_get(url: str, token: str) -> object:
    request = Request(url, headers={"Authorization": f"Bearer {token}"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()
    namespace, name = os.environ["POD_NAMESPACE"], "labeling-demand"
    path = f"/api/v1/namespaces/{namespace}/configmaps/{name}"
    demand = kubernetes_request("GET", path)
    status = demand.get("data", {}).get("status")
    if status != "needs-labeling":
        print(json.dumps({"result": "skipped", "status": status or "missing"}))
        return

    base_url = os.environ["LABEL_STUDIO_URL"].rstrip("/")
    token = access_token(base_url, os.environ["LABEL_STUDIO_TOKEN"])
    # Fetch a modest window: this project keeps the hand-off deliberately
    # bounded to a reviewable batch, not a surprise bulk-labeling operation.
    tasks = label_studio_get(
        f"{base_url}/api/projects/{args.project_id}/tasks?page_size=200", token
    )
    selected = [
        {"id": task["id"], "image": task.get("data", {}).get("image", "")}
        for task in tasks if not task.get("is_labeled", False)
    ][:args.batch_size]
    if not selected:
        raise SystemExit("no unlabelled Label Studio tasks available for this campaign")
    campaign = {
        "project_id": args.project_id,
        "task_count": len(selected),
        "task_ids": [task["id"] for task in selected],
        "tasks": selected,
        "source": demand["data"].get("source", "prediction-drift"),
        "created_at": datetime.now(UTC).isoformat(),
    }
    demand.setdefault("data", {})["status"] = "campaign-ready"
    demand["data"]["campaign.json"] = json.dumps(campaign, sort_keys=True)
    kubernetes_request("PUT", path, demand)
    print(json.dumps({"result": "campaign-ready", **campaign}, sort_keys=True))


if __name__ == "__main__":
    main()
