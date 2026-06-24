# 01 — llama.cpp quickstart

A 30-minute tour of the deck's vocabulary on your own laptop. You'll measure **TTFT, TPOT, P50/P95/P99** and compare two GGUF quantizations of the same model.

## Goals

By the end of this track you should be able to answer:

1. What's the TTFT and TPOT for *your* laptop on a Q4_K_M model? Where does that put you on the latency taxonomy from §0?
2. How much does dropping from the primary quantization to a smaller comparison quantization save in RAM, and what does it cost you in latency / quality?
3. What does `n_threads` actually do? (Spoiler: more isn't always faster — physical cores beat logical cores for compute-bound work.)

## Run

```bash
# from repo root, with .venv activated
python 01-llama-cpp-quickstart/benchmark.py
```

This loads the primary GGUF model from `models/active.json`, runs:

- A **single-prompt latency** trace (TTFT + TPOT)
- A **20-request batch** to compute P50 / P95 / P99 over a realistic prompt distribution
- A **quantization comparison** loading the primary quantization and a smaller comparison quantization, same prompts, side-by-side numbers

Output is printed to stdout and written to `benchmarks/01-quickstart-results.md`.

## Reading the output

```
TTFT (Time To First Token):     128 ms     ← prefill latency
TPOT (Time Per Output Token):    34 ms     ← decode latency
Decode rate:                     29 tok/s
P50 / P95 / P99 e2e (32 tok):   1.21 / 1.45 / 1.62 s
```

**TTFT** is dominated by prefill compute — long prompts make this big. **TPOT** is dominated by KV-cache memory bandwidth — small models on fast RAM look great here.

## Knobs to try

Edit `benchmark.py` (or pass via env vars) and observe the change:

| Variable | Default | What it does |
|---|---|---|
| `LAB_N_CTX` | 2048 | Context window. Bigger = more KV-cache memory. |
| `LAB_N_THREADS` | physical-core count | Threads. **More ≠ faster.** Try `cores // 2`, `cores`, `cores * 2`. |
| `LAB_N_BATCH` | 512 | Prompt processing batch size (prefill). |
| `LAB_N_GPU_LAYERS` | 0 (or auto) | Layers offloaded to GPU. `99` = full offload. |
| `LAB_TEMPERATURE` | 0.7 | Sampling temp. |

## Don't be fooled by

- **Cold-start overhead.** First call includes model load (could be 10s+). Throw away the first run when computing P50/P95/P99.
- **OS file cache.** Second run loads the model 5–10× faster because it's already in page cache. Both numbers matter, but make sure you know which one you're reporting.
- **Background processes.** Browser, IDE, Slack — all eat CPU and memory bandwidth. Close them before benchmarking.

## Deck mapping

- §0 Latency Taxonomy → the TTFT/TPOT/P95 numbers you just measured
- §1 Quantization → the primary quantization vs a smaller comparison quantization
- §2 KV Cache & Attention → `LAB_N_CTX` controls how much KV cache you allocate
- §3 Production Tuning → `LAB_N_THREADS`, `LAB_N_BATCH`, `LAB_N_GPU_LAYERS` are the knobs
