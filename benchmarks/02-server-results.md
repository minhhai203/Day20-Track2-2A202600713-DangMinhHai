# 02 — llama-server Load Test Results

Server: native `llama-server` (Metal, `-ngl 99`, `-t 10`, `--parallel 4`, `--cont-batching`, `--metrics`)

## Locust summary

| Concurrency | Total reqs | RPS | E2E Median (ms) | E2E P95 (ms) | E2E P99 (ms) | Failures |
|--:|--:|--:|--:|--:|--:|--:|
| 10 | 75 | 1.29 | 6000 | 9600 | 11000 | 0 |
| 50 | 86 | 1.44 | 13000 | 22000 | 23000 | 0 |

## Prometheus metrics (during load-50)

Recorded via `make metrics` → `benchmarks/02-server-metrics.csv`

| Metric | Peak value |
|---|---:|
| `llamacpp:n_busy_slots_per_decode` | 3.71 |
| `llamacpp:requests_processing` | 4 |
| `llamacpp:requests_deferred` | 46 |
| `llamacpp:tokens_predicted_total` | 12,138 |

**Observation:** With `--parallel 4`, the server saturates all 4 decode slots under 50-user load (`requests_processing` pegged at 4). Deferred requests queue up to 46 — continuous batching is working but the laptop becomes the bottleneck at high concurrency.
