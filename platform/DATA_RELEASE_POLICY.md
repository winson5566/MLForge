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
3. the `CML data validation / validate-and-report` status check.

Algorithm engineers review the PR's CML report: annotation count, empty or
duplicate data, class distribution, and baseline comparison. Merge is the
audited data-publication decision. A merge that changes exactly one
`data/releases/project-*/<release-id>.dvc` pointer triggers the **Train MLflow
Candidate** GitHub workflow. It submits the `train-mlflow-candidate` Argo
WorkflowTemplate, which performs DVC data preparation, PyTorch training, and
MLflow Candidate registration inside the workload cluster.

Candidate promotion is a separate, manually initiated **Promote MLflow
Candidate** workflow. It targets the `model-production` GitHub Environment;
configure required reviewers there. The Argo promotion gate reads the metric
from the Candidate's MLflow run itself and only assigns the `champion` alias
when it meets the declared threshold. A rejected Candidate must never reach
the serving deployment.

For a team deployment, replace `@winson5566` in `CODEOWNERS` with an algorithm
engineering team and use a GitHub App, not a personal access token, for the
`github-bot` Kubernetes secret.
