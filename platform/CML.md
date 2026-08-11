# CML / GitHub Actions hand-off

The workflow at `.github/workflows/cml-train.yaml` is deliberately configured
for a `self-hosted` runner with the `mlforge` label. This is required because
the local K3d cluster and MinIO endpoint are not reachable from GitHub-hosted
runners.

Before pushing this repository to GitHub:

1. Create a GitHub repository and add it as `origin`.
2. Install a GitHub Actions self-hosted runner on this Mac and assign it the
   `mlforge` label.
3. Add repository secrets `DVC_S3_ACCESS_KEY_ID` and
   `DVC_S3_SECRET_ACCESS_KEY` using the MinIO DVC credentials.
4. Commit the DVC metadata, platform manifests, and workflow. Never commit
   `.dvc/config.local`.

The workflow pulls DVC data, submits the project manifest specified by the
repository variable `TRAINING_MANIFEST` (or the manual-dispatch input), waits
for completion, and comments the tail of the training log on pull requests.
