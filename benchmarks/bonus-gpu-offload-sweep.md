# Bonus — GPU-offload sweep

Model: `Llama-3.2-3B-Instruct-Q4_K_M.gguf`  ·  threads: `10`

| -ngl | tg128 (tok/s) |
|--:|--:|
| 0 | 6.9 |
| 8 | 8.0 |
| 16 | 10.1 |
| 24 | 25.0 |
| 32 | 76.4 |
| 99 | 64.4 |

When the model fits in VRAM, `-ngl 99` (full offload) is fastest. When it doesn't, partial offload (`-ngl 16` or `-ngl 24`) keeps the most compute on the GPU while spilling weights to RAM — usually still beats CPU-only (`-ngl 0`). Watch for the curve flattening: after the layer count covers the model's actual depth, more `-ngl` does nothing.
