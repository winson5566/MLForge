# Local Canary SLO gate

This is a **serving reliability** gate, not an online accuracy test. It sends
real images from the immutable DVC data release to the deployed
`demo-champion`, then writes its evidence to MLflow.

| Control | Local threshold | Failure action |
| --- | ---: | --- |
| Success rate | 100% of 8 requests | Do not retain or expand the Canary; roll `demo-champion` back. |
| p95 inference latency | <= 10 seconds | Do not retain or expand the Canary; investigate resources/model size. |
| Model provenance | `/health` must resolve `demo-champion` | Stop validation; repair alias/deployment mismatch. |

The generous latency budget is intentionally scoped to CPU-only Apple Silicon
local validation. A production GPU policy must be a separately versioned file
with realistic traffic, error, latency, saturation, and drift thresholds.
