# Manual model download

If `download-model.py` can't reach Hugging Face (university firewall, captive portal, slow network), grab the GGUF file in a browser and drop it into `models/` yourself.

## Steps

1. Look at `hardware.json` → `recommendation.recommended_model`. That tells you the tier.
2. Open the matching repo in a browser:

   | Tier | Hugging Face URL |
   |---|---|
   | `TinyLlama-1.1B` | https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tree/main |
   | `Qwen2.5-1.5B-Instruct` | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/main |
   | `Llama-3.2-3B-Instruct` | https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/tree/main |
   | `Qwen2.5-7B-Instruct` | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/main |

3. Download two files:
   - The primary GGUF listed in `download-model.py` for your tier
   - The smaller comparison GGUF listed in `download-model.py` for your tier

4. Place them under `models/` in the repo root. Any subdirectory layout is fine.

5. Re-run with `--skip-download` so it just writes the manifest:

   ```bash
   python 00-setup/download-model.py --skip-download
   ```

   This writes `models/active.json` pointing at the two `.gguf` files it found.

## Mirror fallbacks

The download script tries Hugging Face first, then `hf-mirror.com`. If both are blocked you can also try:

- ModelScope: https://modelscope.cn/models — mirrors many GGUF repos
- A local conference USB drive if your instructor brought one

If you've manually placed files but the `--skip-download` step doesn't find them, check that the filenames match the table in `download-model.py`'s `TIERS` dict.
