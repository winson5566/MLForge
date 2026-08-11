"""Decide whether a Label Studio event is allowed to start a dataset release."""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen


def access_token(base_url: str, refresh_token: str) -> str:
    request = Request(
        f"{base_url.rstrip('/')}/api/token/refresh",
        data=json.dumps({"refresh": refresh_token}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)["access"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--minimum-annotations", type=int, required=True)
    parser.add_argument("--quiet-period-seconds", type=int, default=300)
    parser.add_argument("--decision-output", required=True)
    args = parser.parse_args()

    base_url = os.environ["LABEL_STUDIO_URL"].rstrip("/")
    token = access_token(base_url, os.environ["LABEL_STUDIO_TOKEN"])
    request = Request(
        f"{base_url}/api/projects/{args.project_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urlopen(request, timeout=30) as response:
        project = json.load(response)

    annotated = int(project.get("num_tasks_with_annotations") or 0)
    observed_at = datetime.now(UTC)
    project_updated_at = project.get("updated_at")
    quiet = True
    if project_updated_at:
        updated_at = datetime.fromisoformat(project_updated_at.replace("Z", "+00:00"))
        quiet = (observed_at - updated_at).total_seconds() >= args.quiet_period_seconds
    release = annotated >= args.minimum_annotations and quiet
    # Label Studio OSS does not expose a per-review webhook action in this local
    # version. The downstream transform still snapshots immutable data; a
    # production deployment must set an approval/review policy in that gate.
    decision = {
        "project_id": args.project_id,
        "annotated_tasks": annotated,
        "minimum_annotations": args.minimum_annotations,
        "quiet_period_seconds": args.quiet_period_seconds,
        "project_updated_at": project_updated_at,
        "observed_at": observed_at.isoformat(),
        "release": release,
        "reason": (
            "threshold-met"
            if release
            else "below-threshold"
            if annotated < args.minimum_annotations
            else "quiet-period-not-met"
        ),
    }
    target = Path(args.decision_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
