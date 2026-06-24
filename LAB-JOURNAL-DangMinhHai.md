# Lab Journal — Day 20 Model Serving (Đặng Minh Hải)

> Nhật ký làm bài cá nhân: từng bước, giải thích dễ hiểu, kèm lệnh/code.
> Máy: **Apple M1 Max · 32 GB RAM · Metal**

---

## Mục lục

1. [Setup môi trường (Track 00)](#1-setup-môi-trường-track-00)
2. [Benchmark latency (Track 01)](#2-benchmark-latency-track-01)
3. [llama-server + load test (Track 02)](#3-llama-server--load-test-track-02)
4. [RAG pipeline (Track 03)](#4-rag-pipeline-track-03)
5. [Bonus: tối ưu tốc độ](#5-bonus-tối-ưu-tốc-độ)
6. [Kết quả tổng hợp](#6-kết-quả-tổng-hợp)

---

## 1. Setup môi trường (Track 00)

### Bước 1.1 — Probe phần cứng

**Làm gì:** Script đọc CPU, RAM, GPU backend và ghi `hardware.json`.

**Tại sao:** Mọi script sau đọc file này để auto-chọn Metal/CUDA/Vulkan và model tier phù hợp RAM.

```bash
cd Day20-Track2-2A202600713-DangMinhHai
python 00-setup/detect-hardware.py
```

**Kết quả trên máy tôi:**
- CPU: Apple M1 Max, 10 cores
- RAM: 32 GB
- Backend: **Metal** (`-DGGML_METAL=on`)
- Model gợi ý: Qwen2.5-7B — nhưng setup đã tải **Llama-3.2-3B-Instruct Q4_K_M**

### Bước 1.2 — Cài đặt đầy đủ

**Làm gì:** Tạo `.venv`, cài Python deps, build `llama-cpp-python` với Metal, download GGUF.

**Tại sao:** `llama-cpp-python` phải build native với flag Metal mới offload được lên GPU Apple Silicon.

```bash
bash 00-setup/macos-setup.sh
# hoặc: make setup
```

**Output quan trọng:**
- `hardware.json` — spec máy
- `models/active.json` — đường dẫn model chính + model so sánh
- `models/Llama-3.2-3B-Instruct-Q4_K_M.gguf` (~2 GB)
- `models/Llama-3.2-3B-Instruct-Q3_K_L.gguf` (~1.8 GB)

```bash
source .venv/bin/activate
cat models/active.json
```

---

## 2. Benchmark latency (Track 01)

### Bước 2.1 — Chạy benchmark với tối ưu Metal

**Làm gì:** Load model, đo TTFT (Time To First Token), TPOT (Time Per Output Token), P50/P95/P99 trên 10 prompts, so sánh 2 quantization.

**Tại sao tối ưu `-ngl 99`:** Offload toàn bộ layers lên GPU Metal — thay đổi lớn nhất trên Apple Silicon.

```bash
LAB_N_GPU_LAYERS=99 LAB_N_THREADS=10 LAB_N_BATCH=512 make bench
```

**Kết quả chính:**

| Model | TTFT P50 | TPOT P50 | Decode tok/s |
|---|---:|---:|---:|
| Q4_K_M | 46 ms | 11.4 ms | **87.9** |
| Q3_K_L | 57 ms | 16.1 ms | 62.3 |

**Giải thích:**
- **TTFT** = thời gian prefill (xử lý prompt trước token đầu tiên)
- **TPOT** = thời gian mỗi token decode — `1000/TPOT` ≈ tok/s
- Q4_K_M **nhanh hơn** Q3_K_L trên Metal — quantization nhỏ hơn không luôn nhanh hơn vì kernel efficiency

File output: `benchmarks/01-quickstart-results.md`

---

## 3. llama-server + load test (Track 02)

### Bước 3.1 — Build llama.cpp native (có `/metrics`)

**Làm gì:** Clone + cmake build `llama-server` binary từ source.

**Tại sao:** Python server (`make serve`) **không có** endpoint `/metrics`. Native server mới có Prometheus gauges (`n_busy_slots_per_decode`, `requests_processing`).

```bash
LLAMA_CMAKE_FLAGS="-DGGML_METAL=ON" make build-llama
# Kiểm tra:
BONUS-llama-cpp-optimization/llama.cpp/build/bin/llama-bench --version
```

### Bước 3.2 — Khởi động native server

**Làm gì:** Chạy `llama-server` với continuous batching + metrics.

**Tại sao `--parallel 4 --cont-batching`:** Cho phép 4 request decode đồng thời — analog của vLLM continuous batching trong deck §2.

```bash
# Terminal 1
LAB_N_GPU_LAYERS=99 LAB_PARALLEL=4 make serve-native
```

Flags quan trọng:
- `-ngl 99` — full Metal offload
- `-t 10` — 10 CPU threads (physical cores)
- `--parallel 4` — 4 decode slots
- `--metrics` — bật Prometheus

### Bước 3.3 — Smoke test

```bash
# Terminal 2
make smoke
curl -s http://localhost:8080/metrics | head -20
```

Xác nhận: `llamacpp:tokens_predicted_total` > 0 sau 1 request.

### Bước 3.4 — Load test Locust

**Làm gì:** Giả lập 10 rồi 50 user đồng thời, mỗi lần 60 giây.

**Tại sao:** Đo P95 dưới contention — metric production quan trọng hơn throughput peak.

```bash
make load-10    # 10 users, 1 phút
make load-50    # 50 users, 1 phút
```

**Kết quả:**

| Users | RPS | E2E P95 | Failures |
|--:|--:|--:|--:|
| 10 | 1.29 | 9600 ms | 0 |
| 50 | 1.44 | 22000 ms | 0 |

### Bước 3.5 — Ghi metrics Prometheus

```bash
# Chạy song song với load-50:
make metrics   # → benchmarks/02-server-metrics.csv
```

Peak quan sát được:
- `requests_processing` = **4** (đầy slot)
- `n_busy_slots_per_decode` ≈ **3.71**
- `requests_deferred` = **46** (hàng đợi)

---

## 4. RAG pipeline (Track 03)

### Bước 4.1 — Cải tiến `pipeline.py`

**Làm gì:** Thay keyword stub bằng pipeline có:
- N18: SQLite lakehouse stub
- N19: numpy cosine vector index
- N20: gọi `llama-server` qua OpenAI-compat API
- Đo timing từng stage bằng `time.perf_counter`

**Tại sao:** Chứng minh endpoint OpenAI-compat slot vào RAG stack thật; đo bottleneck.

```bash
make pipeline
```

**Cấu trúc code chính:**

```python
# Config
LLAMA_SERVER_BASE = "http://localhost:8080/v1"

def retrieve(query: str, k: int = 3) -> list[Doc]:
    query_vec = embed(query)
    return get_index().search(query_vec, k=k)

def answer(query: str) -> dict:
    # embed → retrieve → build_prompt → call_llm
    ...
```

**Timing (sau warmup):**

| Stage | ms |
|---|---:|
| embed | ~0.1 |
| retrieve | ~0.02 |
| llama-server | 960–4000 |

→ **LLM chiếm >99% thời gian** — đúng kỳ vọng RAG trên laptop.

Ghi chú: `benchmarks/03-integration-notes.md`

---

## 5. Bonus: tối ưu tốc độ

### 5.1 — GPU offload sweep (thay đổi quan trọng nhất)

**Làm gì:** Sweep `-ngl` từ 0 (CPU) đến 99 (full Metal).

```bash
make sweep-gpu
```

| -ngl | tok/s | Ý nghĩa |
|--:|--:|---|
| 0 | 6.9 | CPU only |
| 32 | 76.4 | Partial offload |
| 99 | 64.4 | Full Metal |

**Speedup CPU → Metal: ~9.3×** — đây là insight lớn nhất của lab.

### 5.2 — Thread sweep

```bash
make sweep-thread
```

| threads | tok/s |
|--:|--:|
| 2 | **94.3** (best) |
| 10 | 62.1 |

**Tại sao ít thread hơn lại nhanh hơn trên Metal:** Decode đã chạy trên GPU; CPU threads thừa chỉ tranh memory bus.

### 5.3 — MLX vs llama.cpp (Apple Silicon)

```bash
pip install mlx mlx-lm
make mlx-compare
```

| Runtime | TTFT P50 | Decode tok/s |
|---|---:|---:|
| llama.cpp Metal | 61 ms | 61.6 |
| MLX-LM | 10 ms | **97.2** |

MLX nhanh hơn ~1.58× decode — unified memory zero-copy aggressive hơn.

### 5.4 — Semantic cache (Challenge C8)

```bash
python BONUS-llama-cpp-optimization/semantic-cache-demo.py --offline --sweep
```

- Hit rate: **38%** (3/8 prompts)
- Tiết kiệm 3 LLM calls
- Cache layer *trên* KV cache — bắt paraphrase

### 5.5 — Embedding serving demo

```bash
python BONUS-llama-cpp-optimization/embedding-serving.py --offline
```

Embedding = prefill-bound, **không có KV cache/decode loop** — regime serving khác chat.

---

## 6. Kết quả tổng hợp

### Files đã tạo / cập nhật

| File | Mô tả |
|---|---|
| `hardware.json` | Spec M1 Max |
| `benchmarks/01-quickstart-results.md` | TTFT/TPOT baseline |
| `benchmarks/02-server-results.md` | Locust + metrics |
| `benchmarks/03-integration-notes.md` | Pipeline latency |
| `benchmarks/bonus-gpu-offload-sweep.md` | GPU sweep |
| `benchmarks/bonus-thread-sweep.md` | Thread sweep |
| `benchmarks/bonus-mlx-vs-llama-cpp.md` | MLX comparison |
| `submission/REFLECTION.md` | Báo cáo chấm điểm |
| `submission/screenshots/*.png` | 8 screenshots |

### Tối ưu đã áp dụng

1. **`-ngl 99`** — Metal full offload (~9× vs CPU)
2. **`-t 2`** trong micro-bench (tốt hơn `-t 10` khi GPU làm việc chính)
3. **`--parallel 4 --cont-batching`** — continuous batching cho server
4. **Q4_K_M** thay vì Q3_K_L — cân bằng quality/speed tốt hơn trên Metal

### Verify trước khi nộp

```bash
make verify    # phải exit 0
git add ... && git push   # repo PUBLIC trên GitHub
```

### Lệnh nhanh tái tạo toàn bộ lab

```bash
source .venv/bin/activate
make bench
make build-llama
make serve-native &    # terminal 1
make smoke
make load-10
make load-50
make metrics
make pipeline
make sweep-gpu
make sweep-thread
make mlx-compare
make verify
```

---

*Lab hoàn thành ngày 2026-06-24 · Đặng Minh Hải · 2A202600713*
