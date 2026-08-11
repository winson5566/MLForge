"""Record Label Studio changes as pending data-release state in Kubernetes."""
from __future__ import annotations

import argparse
import json
import os
import ssl
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from mlforge_platform.release_gate import access_token


def kubernetes_request(method: str, path: str, payload: dict | None = None) -> dict:
    token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
    request = Request(
        f"https://kubernetes.default.svc{path}",
        data=json.dumps(payload).encode() if payload else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method=method,
    )
    context = ssl.create_default_context(cafile="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
    with urlopen(request, context=context, timeout=15) as response:
        return json.loads(response.read().decode() or "{}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--event-id", required=True)
    args = parser.parse_args()

    base_url = os.environ["LABEL_STUDIO_URL"].rstrip("/")
    token = access_token(base_url, os.environ["LABEL_STUDIO_TOKEN"])
    request = Request(
        f"{base_url}/api/projects/{args.project_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urlopen(request, timeout=30) as response:
        project = json.load(response)

    namespace = os.environ["POD_NAMESPACE"]
    name = "label-release-status"
    key = f"project-{args.project_id}.json"
    state = {
        "project_id": args.project_id,
        "pending": True,
        "annotated_task_count": int(project.get("num_tasks_with_annotations") or 0),
        "total_task_count": int(project.get("task_number") or 0),
        "last_event_id": args.event_id,
        "last_change_observed_at": datetime.now(UTC).isoformat(),
    }
    path = f"/api/v1/namespaces/{namespace}/configmaps/{name}"
    try:
        existing = kubernetes_request("GET", path)
        existing.setdefault("data", {})[key] = json.dumps(state, sort_keys=True)
        kubernetes_request("PUT", path, existing)
        action = "updated"
    except HTTPError as error:
        if error.code != 404:
            raise
        kubernetes_request("POST", f"/api/v1/namespaces/{namespace}/configmaps", {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": name, "namespace": namespace},
            "data": {key: json.dumps(state, sort_keys=True)},
        })
        action = "created"
    print(json.dumps({"action": action, **state}, sort_keys=True))


if __name__ == "__main__":
    main()
