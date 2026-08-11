# Local YOLO Canary runbook

`local-demo` is a deliberately separate policy. It promotes a qualifying
candidate to the `demo-champion` alias, while the production `champion` alias
remains subject to the `production` policy (mAP50 >= 0.70 and the
`model-production` GitHub environment).

## Promotion

Run **Promote MLflow Candidate** on `main` with:

- `model_name`: `label-studio-project-1-yolo-detector`
- `model_version`: the current `candidate` version
- `policy`: `local-demo`

The job pauses at the `model-staging` environment, then Argo records the
metric-based promotion. Verify in MLflow that `demo-champion` resolves to the
same version as `candidate`.

## Deploy and verify

Build/import the image in the local k3d cluster, then apply the manifest:

```sh
docker build -t mlforge/yolo-canary-server:0.1.0 images/yolo-canary-server
k3d image import -c workload mlforge/yolo-canary-server:0.1.0
kubectl --context k3d-workload apply -f platform/yolo-canary-serving.yaml
kubectl --context k3d-workload -n ml-platform rollout status deployment/yolo-local-canary --timeout=8m
kubectl --context k3d-workload -n ml-platform port-forward service/yolo-local-canary 8090:8080
```

`GET http://localhost:8090/health` must return the expected `demo-champion`
version. Send test images to `POST /predict` and inspect `GET /metrics`.

## Rollback

Rollback is an alias move, not an in-place image edit. Reassign
`demo-champion` to the last known-good registered version, then restart the
deployment. This leaves an MLflow audit trail. Production uses the same
contract but with the `champion` alias, Triton, and weighted traffic routing.
