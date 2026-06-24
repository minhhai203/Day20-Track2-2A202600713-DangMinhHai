# Bonus — MLX vs llama.cpp Metal

Tier: `Llama-3.2-3B-Instruct`

| runtime | TTFT P50 (ms) | TTFT P95 (ms) | decode (tok/s) |
|---|--:|--:|--:|
| llama.cpp (Metal) | 61.1 | 86.6 | 61.6 |
| MLX-LM | 10.3 | 12.8 | 97.2 |

MLX TTFT numbers are approximations (mlx-lm doesn't expose first-token timing as readily as llama-cpp-python's stream API). Trust the decode tok/s for the head-to-head; trust both implementations' P95 only as rough indicators.
