# Data release policy

The Label Studio webhook does not publish data. It only records pending state
in the `label-release-status` ConfigMap. An algorithm engineer uses the GitHub
Actions **Create data release** button to export an immutable snapshot,
validate it, push the data blobs to MinIO through DVC, and create a candidate
data-release PR.

Before training can start, configure the repository's `main` branch protection
in GitHub to require:

1. one approving review;
2. review from a code owner (see `.github/CODEOWNERS`);
3. the `CML training / train-and-report` status check.

Algorithm engineers review the PR's CML report: annotation count, empty or
duplicate data, class distribution, and baseline comparison. Merge is the
audited data-publication decision. Only the merge event starts full training.

For a team deployment, replace `@winson5566` in `CODEOWNERS` with an algorithm
engineering team and use a GitHub App, not a personal access token, for the
`github-bot` Kubernetes secret.
