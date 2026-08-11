"""Export a completed Label Studio project into the project DVC data area."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def _access_token(base_url: str, personal_access_token: str) -> str:
    """Exchange a Label Studio PAT refresh token for a short-lived API token."""
    request = Request(
        f"{base_url}/api/token/refresh",
        data=json.dumps({"refresh": personal_access_token}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)["access"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output", default="data/annotations/label-studio-export.json")
    args = parser.parse_args()

    base_url = os.environ["LABEL_STUDIO_URL"].rstrip("/")
    personal_access_token = os.environ["LABEL_STUDIO_TOKEN"]
    token = _access_token(base_url, personal_access_token)
    request = Request(
        f"{base_url}/api/projects/{args.project_id}/export?exportType=JSON",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"exported {len(payload)} tasks to {target}")


if __name__ == "__main__":
    main()
