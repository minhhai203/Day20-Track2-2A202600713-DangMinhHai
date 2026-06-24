# 03 — Milestone Integration Notes

## N16–N19 pieces connected

| Day | Component | Implementation |
|---|---|---|
| **N16** Cloud/IaC | **Stub:** localhost-only — no K8s cluster needed for this demo |
| **N17** Data pipeline | **Stub:** static `TOY_DOCS` list seeded into SQLite at pipeline startup |
| **N18** Lakehouse | **Stub:** SQLite file `03-milestone-integration/lakehouse_stub.db` mimicking a doc table |
| **N19** Vector store | **Stub:** in-memory numpy cosine index over hash-based embeddings |
| **N20** Serving | **Real:** `llama-server` on `http://localhost:8080/v1` (OpenAI-compat) |

## Latency breakdown (warm run, `time.perf_counter`)

| Stage | Typical (ms) | Notes |
|---|---:|---|
| embed | ~0.1 | Hash-based embedder — instant after index warmup |
| retrieve | ~0.02 | Cosine search over 5 docs |
| llama-server | 960–4000 | Dominates end-to-end; scales with output length |

## Observation

The bottleneck is overwhelmingly **llama-server decode** (>99% of pipeline time). Embedding and retrieval are negligible with a toy corpus. In production with a real vector DB (Qdrant/Milvus) and a proper embedding model (BGE-M3), embed+retrieve might reach 50–200 ms — still usually less than LLM generation for RAG workloads.
