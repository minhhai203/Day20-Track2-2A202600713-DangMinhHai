# 01 — Quickstart Results

Settings: `n_threads=10`, `n_ctx=2048`, `n_batch=512`, `n_gpu_layers=99`.

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-3B-Instruct-Q4_K_M.gguf | 12818 | 46 / 147 | 11.4 / 12.0 | 773 / 852 / 869 | 87.9 |
| Llama-3.2-3B-Instruct-Q3_K_L.gguf | 1209 | 57 / 140 | 16.1 / 16.7 | 1069 / 1138 / 1153 | 62.3 |

## Observations

- TTFT is the prefill cost. With short prompts this is small; with long prompts it dominates.
- TPOT is per-token decode latency. The decode rate is `1000 / TPOT_p50`.
    - The primary quantization is usually only modestly slower than the smaller comparison quantization but produces noticeably better text. The smaller one is for *truly* tight RAM.
- `n_threads = physical_cores` is usually best on CPU. Hyperthreading (`logical_cores`) often hurts because the work is bandwidth-bound.

(Edit this file with your own observations before submitting.)
