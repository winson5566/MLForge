# Drift-to-labeling campaign

When `monitor-yolo-local-predictions` detects drift, it creates
`ml-platform/labeling-demand` with `status=needs-labeling`. The campaign
workflow runs every five minutes and selects a bounded batch of existing,
unlabelled Label Studio tasks. It never imports duplicate S3 image URIs.

Inspect the selected batch:

```sh
kubectl --context k3d-workload -n ml-platform get configmap labeling-demand \
  -o jsonpath='{.data.campaign\.json}' | jq
```

Annotate and review those task IDs in Label Studio project 1. Its existing
webhook records the change as pending data-release state. An algorithm engineer
then uses **Create data release** to make the governed DVC/GitHub data PR.
