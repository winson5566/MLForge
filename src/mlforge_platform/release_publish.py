"""Version a validated release with DVC and open an idempotent GitHub PR."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def run(*command: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--base-branch", default="main")
    args = parser.parse_args()

    token = os.environ["GITHUB_TOKEN"]
    branch = f"data/label-studio-project-{args.project_id}-{args.release_id}"
    workspace = Path("/workspace/repository")
    run("git", "clone", "--depth", "1", "--branch", args.base_branch,
        f"https://x-access-token:{token}@github.com/{args.repository}.git", str(workspace))
    run("git", "checkout", "-B", branch, cwd=workspace)
    run("git", "config", "user.name", "mlforge-data-bot", cwd=workspace)
    run("git", "config", "user.email", "mlforge-data-bot@users.noreply.github.com", cwd=workspace)
    run("dvc", "remote", "modify", "--local", "minio", "endpointurl", os.environ["DVC_ENDPOINT_URL"], cwd=workspace)

    source = Path(args.release_root) / f"project-{args.project_id}" / args.release_id
    destination = workspace / "data" / "releases" / f"project-{args.project_id}" / args.release_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(("cp", "-R", str(source), str(destination)))
    run("dvc", "add", str(destination.relative_to(workspace)), cwd=workspace)
    run("dvc", "push", cwd=workspace)
    run("git", "add", "data", ".dvc/config", cwd=workspace)
    run("git", "commit", "-m", f"data: release Label Studio project {args.project_id} ({args.release_id})", cwd=workspace)
    run("git", "push", "--force-with-lease", "origin", branch, cwd=workspace)

    body = (
        "## Label Studio data release\n\n"
        f"- Project: `{args.project_id}`\n"
        f"- Immutable release ID: `{args.release_id}`\n"
        "- Data blobs: local MinIO via DVC\n"
        "- Next gate: CML data validation and baseline comparison\n"
    )
    existing = run("gh", "pr", "list", "--repo", args.repository, "--head", branch,
                   "--state", "open", "--json", "url")
    if json.loads(existing):
        print(existing)
        return
    url = run("gh", "pr", "create", "--repo", args.repository, "--base", args.base_branch,
              "--head", branch, "--title", f"data: Label Studio project {args.project_id} {args.release_id}",
              "--body", body)
    print(url)


if __name__ == "__main__":
    main()
