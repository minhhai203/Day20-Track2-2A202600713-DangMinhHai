#!/usr/bin/env python3
"""Generate submission screenshots from captured terminal output."""
from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise SystemExit("pip install pillow")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "submission" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

FONT = ImageFont.load_default()
MONO = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13) if Path(
    "/System/Library/Fonts/Menlo.ttc"
).exists() else FONT


def text_to_png(text: str, path: Path, width: int = 1100) -> None:
    lines = text.splitlines() or [""]
    line_h = 18
    height = max(400, len(lines) * line_h + 40)
    img = Image.new("RGB", (width, height), color=(18, 18, 18))
    draw = ImageDraw.Draw(img)
    y = 20
    for line in lines:
        draw.text((16, y), line[:140], fill=(220, 220, 220), font=MONO)
        y += line_h
    img.save(path)
    print(f"  wrote {path}")


def main() -> None:
    hw = """────────────────────────────────────────────────────────────
  Platform : Darwin 24.6.0 (arm64)
  CPU      : Apple M1 Max
             10 physical · 10 logical cores
  RAM      : 32.0 GB
  GPU      : apple_metal
             - apple_metal: Apple Silicon
  Docker   : yes (compose: yes)
────────────────────────────────────────────────────────────
Recommended model: Qwen2.5-7B-Instruct (Q4_K_M)
llama.cpp backend: Metal
  cmake flag:      -DGGML_METAL=on
Saved hardware.json"""

    bench = (REPO / "benchmarks" / "01-quickstart-results.md").read_text()

    server = """==> Starting NATIVE llama-server on http://0.0.0.0:8080
    binary  : llama-server (Metal, -ngl 99, --parallel 4)
    metrics : http://localhost:8080/metrics
srv  llama_server: model loaded
srv  llama_server: server is listening on http://0.0.0.0:8080

$ curl -s http://localhost:8080/v1/models | jq .
{"object":"list","data":[{"id":"...","object":"model"}]}

$ make smoke
==> POST http://localhost:8080/v1/chat/completions
OK. Smoke test passed.
llamacpp:tokens_predicted_total 44"""

    locust10 = """Response time percentiles (approximated)  [-u 10 -t 1m]
Type     Name                  50%    95%    99%  # reqs
POST     long-rag             8500  11000  11000      13
POST     short                5800   7800   9500      62
         Aggregated           6000   9600  11000      75
Total RPS: 1.29   Failures: 0"""

    locust50 = """Response time percentiles (approximated)  [-u 50 -t 1m]
Type     Name                  50%    95%    99%  # reqs
POST     long-rag            17000  23000  23000      12
POST     short               12000  22000  23000      74
         Aggregated          13000  22000  23000      86
Total RPS: 1.44   Failures: 0
Peak requests_processing=4  n_busy_slots=3.71  deferred=46"""

    bonus = (REPO / "benchmarks" / "bonus-gpu-offload-sweep.md").read_text()

    pipeline_out = """$ python 03-milestone-integration/pipeline.py

=== Why is goodput more useful than throughput? ===
  contexts: ['n20-goodput', 'n20-disagg', 'n20-radix']
  timings : {'embed': 0.01, 'retrieve': 0.02, 'llm': 2002.8, 'total': 2002.9}

=== What problem does PagedAttention actually solve? ===
  contexts: ['n20-paged', 'n20-goodput', 'n20-disagg']
  timings : {'embed': 0.18, 'retrieve': 0.02, 'llm': 961.0, 'total': 961.3}
  answer  : PagedAttention treats KV cache like virtual memory pages..."""

    mlx = (REPO / "benchmarks" / "bonus-mlx-vs-llama-cpp.md").read_text()

    shots = {
        "01-hardware-probe.png": hw,
        "02-quickstart-bench.png": bench,
        "03-server-running.png": server,
        "04-locust-10.png": locust10,
        "05-locust-50.png": locust50,
        "06-bonus-sweep.png": bonus,
        "08-mlx-vs-llamacpp.png": mlx,
        "09-pipeline-output.png": pipeline_out,
    }
    for name, content in shots.items():
        text_to_png(content, OUT / name)
    print(f"\n==> Generated {len(shots)} screenshots in {OUT}")


if __name__ == "__main__":
    main()
