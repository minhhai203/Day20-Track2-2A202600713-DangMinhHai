# Bonus — Thread sweep

Model: `Llama-3.2-3B-Instruct-Q4_K_M.gguf`  ·  GPU layers: `99`

| threads | tg128 (tok/s) |
|---:|---:|
| 1 | 92.6 |
| 2 | 94.3 |
| 5 | 92.7 |
| 10 | 62.1 |
| 20 | 61.7 |

**Best**: `-t 2` at 94.3 tok/s.

Look at the curve. If it peaks around your **physical** core count and drops as you go higher, that's the memory-bandwidth ceiling: extra threads fight over the same memory channels and slow each other down.
