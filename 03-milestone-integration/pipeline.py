#!/usr/bin/env python3
"""RAG pipeline: vector retrieval + llama-server (N16–N19 integration).

Wires a minimal end-to-end flow:
  N18 (lakehouse) → SQLite doc store
  N19 (vector)    → in-memory cosine index with numpy embeddings
  N20 (serving)   → local llama-server OpenAI-compat API
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

# ── Config (N16–N19 stubs) ───────────────────────────────────────────────
LLAMA_SERVER_BASE = "http://localhost:8080/v1"
LAKEHOUSE_DB = Path(__file__).parent / "lakehouse_stub.db"
VECTOR_DIM = 64

SYSTEM_PROMPT = (
    "You are a serving-engineering tutor. Answer using only the documents provided. "
    "If the documents don't contain the answer, say so."
)

TOY_DOCS = [
    {"id": "n20-paged", "text": "PagedAttention treats KV cache like virtual memory pages, eliminating 60-80% fragmentation."},
    {"id": "n20-radix", "text": "RadixAttention stores KV in a prefix trie; cache hit on shared prefix lets the engine skip prefill."},
    {"id": "n20-disagg", "text": "Disaggregated serving (Mooncake, llm-d, Dynamo) splits prefill and decode onto separate GPU pools."},
    {"id": "n20-goodput", "text": "Goodput@SLO = req/s satisfying TTFT and TPOT SLOs. Throughput at saturation ignores SLO."},
    {"id": "n20-quant", "text": "GGUF Q4_K_M is the production-quality default for laptop/edge serving via llama.cpp."},
]


@dataclass
class Doc:
    id: str
    text: str
    score: float


# ── N18: Lakehouse stub (SQLite) ───────────────────────────────────────

def init_lakehouse() -> sqlite3.Connection:
    """N18 stub — SQLite table mimicking a Delta/Iceberg doc table."""
    conn = sqlite3.connect(LAKEHOUSE_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, text TEXT NOT NULL)"
    )
    for d in TOY_DOCS:
        conn.execute(
            "INSERT OR REPLACE INTO docs (id, text) VALUES (?, ?)",
            (d["id"], d["text"]),
        )
    conn.commit()
    return conn


# ── N19: Vector index (numpy cosine) ─────────────────────────────────────

def _hash_embed(text: str, dim: int = VECTOR_DIM) -> np.ndarray:
    """Lightweight deterministic embedder — no extra model download."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(dim)
    return v / (np.linalg.norm(v) + 1e-9)


class VectorIndex:
    """N19 stub — in-memory cosine similarity index."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.docs: list[dict] = []
        self.vectors: list[np.ndarray] = []
        for row in conn.execute("SELECT id, text FROM docs"):
            self.docs.append({"id": row[0], "text": row[1]})
            self.vectors.append(_hash_embed(row[1]))

    def search(self, query_vec: np.ndarray, k: int = 3) -> list[Doc]:
        scores = [float(np.dot(query_vec, v)) for v in self.vectors]
        ranked = sorted(
            zip(self.docs, scores), key=lambda x: x[1], reverse=True
        )[:k]
        return [Doc(d["id"], d["text"], s) for d, s in ranked]


_INDEX: VectorIndex | None = None


def get_index() -> VectorIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = VectorIndex(init_lakehouse())
    return _INDEX


def embed(query: str) -> np.ndarray:
    return _hash_embed(query)


def retrieve(query: str, k: int = 3) -> list[Doc]:
    """N19 vector retrieval — embed query, cosine top-K."""
    return get_index().search(embed(query), k=k)


# ── Prompt assembly + N20 llama-server ───────────────────────────────────

def build_prompt(query: str, contexts: list[Doc]) -> list[dict]:
    ctx_block = "\n".join(f"[{c.id}] {c.text}" for c in contexts)
    user = f"Context:\n{ctx_block}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def call_llm(messages: list[dict]) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = httpx.post(
        f"{LLAMA_SERVER_BASE}/chat/completions",
        json={"model": "local", "messages": messages, "max_tokens": 200, "temperature": 0.3},
        timeout=120.0,
    )
    r.raise_for_status()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return r.json()["choices"][0]["message"]["content"], elapsed_ms


def answer(query: str) -> dict:
    t_total = time.perf_counter()

    t = time.perf_counter()
    query_vec = embed(query)
    t_embed_ms = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    docs = get_index().search(query_vec, k=3)
    t_retrieve_ms = (time.perf_counter() - t) * 1000.0

    messages = build_prompt(query, docs)
    text, t_llm_ms = call_llm(messages)

    return {
        "query": query,
        "answer": text,
        "contexts": [{"id": d.id, "score": round(d.score, 3)} for d in docs],
        "timings_ms": {
            "embed": round(t_embed_ms, 2),
            "retrieve": round(t_retrieve_ms, 2),
            "llm": round(t_llm_ms, 1),
            "total": round((time.perf_counter() - t_total) * 1000.0, 1),
        },
    }


def main() -> None:
    # Warm up lakehouse + vector index (avoid cold-start skew in timings)
    get_index()

    queries = [
        "Why is goodput more useful than throughput?",
        "What problem does PagedAttention actually solve?",
        "When should I think about disaggregated serving?",
    ]
    for q in queries:
        print(f"\n=== {q} ===")
        result = answer(q)
        print(f"  contexts: {[c['id'] for c in result['contexts']]}")
        print(f"  scores  : {[c['score'] for c in result['contexts']]}")
        print(f"  timings : {result['timings_ms']}")
        print(f"  answer  : {result['answer'].strip()[:300]}")


if __name__ == "__main__":
    main()
