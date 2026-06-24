# Reflection — Lab 20 (Personal Report)

> Báo cáo cá nhân — Đặng Minh Hải · M1 Max 32GB · so sánh before/after trên cùng máy.

---

**Họ Tên:** Đặng Minh Hải
**Cohort:** 2A202600713
**Ngày submit:** 2026-06-24

---

## 1. Hardware spec (từ `00-setup/detect-hardware.py`)

- **OS:** macOS 15 (Darwin 24.6.0, arm64)
- **CPU:** Apple M1 Max
- **Cores:** 10 physical / 10 logical
- **CPU extensions:** NEON, ARM_FMA, FP16_VA, DOTPROD
- **RAM:** 32 GB
- **Accelerator:** Apple Metal (unified memory)
- **llama.cpp backend đã chọn:** Metal (`-DGGML_METAL=on`)
- **Recommended model tier:** Llama-3.2-3B-Instruct (Q4_K_M) — đã dùng thay vì Qwen2.5-7B vì setup script auto-pick theo RAM

**Setup story:** Chạy `bash 00-setup/macos-setup.sh` — tạo `.venv`, build `llama-cpp-python` với Metal, download GGUF ~2GB. Không cần chỉnh gì thêm; chỉ set `LAB_N_GPU_LAYERS=99` khi benchmark để offload toàn bộ lên GPU.

---

## 2. Track 01 — Quickstart numbers

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|--:|--:|--:|--:|--:|
| Q4_K_M | 12818 | 46 / 147 | 11.4 / 12.0 | 773 / 852 / 869 | 87.9 |
| Q3_K_L | 1209 | 57 / 140 | 16.1 / 16.7 | 1069 / 1138 / 1153 | 62.3 |

**Một quan sát:** Q4_K_M nhanh hơn ~41% decode (87.9 vs 62.3 tok/s) dù file lớn hơn — trên Metal, quantization nhỏ hơn không luôn nhanh hơn vì kernel efficiency. Q4_K_M cho chất lượng tốt hơn với chi phí latency chấp nhận được trên M1 Max 32GB.

---

## 3. Track 02 — llama-server load test

| Concurrency | Total RPS | TTFB P50 (ms) | E2E P95 (ms) | E2E P99 (ms) | Failures |
|--:|--:|--:|--:|--:|--:|
| 10 | 1.29 | ~6000 | 9600 | 11000 | 0 |
| 50 | 1.44 | ~13000 | 22000 | 23000 | 0 |

**Batching observation:** Ở concurrency 50, peak `llamacpp:requests_processing` = **4** (bằng `--parallel 4`) và `n_busy_slots_per_decode` peak ≈ **3.71**. Có tới **46 requests deferred** — server batching hoạt động đúng nhưng 4 slot decode + Metal bandwidth là bottleneck; P95 tăng gấp ~2.3× so với 10 users.

---

## 4. Track 03 — Milestone integration

- **N16 (Cloud/IaC):** stub — localhost only, không cần K8s cho demo
- **N17 (Data pipeline):** stub — `TOY_DOCS` seed vào SQLite lúc startup
- **N18 (Lakehouse):** stub — SQLite `lakehouse_stub.db` (mô phỏng Delta table)
- **N19 (Vector + Feature Store):** stub — numpy cosine index + hash embedder (thay Feast/Qdrant)

**Nơi tốn nhiều ms nhất:**

- embed: ~0.1 ms (sau warmup)
- retrieve: ~0.02 ms
- llama-server: **960–4000 ms**

**Reflection:** Bottleneck 100% nằm ở LLM decode — đúng kỳ vọng cho RAG trên laptop. Với corpus thật, embed có thể lên 50–200ms nhưng vẫn nhỏ hơn generation.

---

## 5. Bonus — The single change that mattered most

**Change:** Bật Metal GPU offload (`-ngl 99`) thay vì CPU-only (`-ngl 0`) — đo bằng `gpu-offload-sweep.py` trên native `llama-bench`.

**Before vs after:**

```
before: -ngl 0  →  6.9 tok/s (CPU only)
after:  -ngl 99 → 64.4 tok/s (full Metal offload)
speedup: ~9.3×
```

**Tại sao nó work:** Llama-3.2-3B có ~3.2B params — matrix multiply trong decode là compute-bound trên GPU nhưng memory-bandwidth-bound trên CPU. M1 Max có unified memory: khi offload toàn bộ layers lên Metal, weights và activations nằm trên GPU path với SIMD matrix ops (Apple7 GPU family), tránh copy qua lại CPU↔GPU. CPU-only phải chạy GGML kernels trên 10 NEON cores — đủ cho edge nhưng không cạnh tranh được với 32-core GPU. Sweep cũng cho thấy `-t 2` (94.3 tok/s) nhanh hơn `-t 10` (62.1 tok/s) khi đã offload — trên Metal, quá nhiều CPU threads chỉ tạo contention cho memory bus thay vì giúp GPU.

**Challenge C8 (Semantic cache):** Hit rate 38% (3/8) với threshold 0.8 — tiết kiệm 3 LLM calls (~750ms decode). Đây là cache *trên* KV cache — bắt paraphrase mà prefix cache bỏ sót.

**MLX comparison:** MLX-LM decode **97.2 tok/s** vs llama.cpp Metal **61.6 tok/s** (~1.58×) — MLX tận dụng unified memory aggressive hơn; llama.cpp cross-platform hơn.

---

## 6. Điều ngạc nhiên nhất

`-ngl 32` (76.4 tok/s) lại nhanh hơn `-ngl 99` (64.4 tok/s) trong micro-benchmark — có thể do memory placement / kernel warmup khác nhau khi partial offload. Trong production vẫn chọn `-ngl 99` vì đơn giản và ổn định hơn.

---

## 7. Self-graded checklist

- [x] `hardware.json` đã commit
- [x] `models/active.json` đã commit
- [x] `benchmarks/01-quickstart-results.md` đã commit
- [x] `benchmarks/02-server-results.md` đã commit
- [x] `benchmarks/bonus-*.md` đã commit (gpu-offload, thread-sweep, mlx)
- [x] Ít nhất 6 screenshots trong `submission/screenshots/`
- [x] `make verify` exit 0
- [ ] Repo trên GitHub ở chế độ **public** (cần push)
- [ ] Đã paste public repo URL vào VinUni LMS
