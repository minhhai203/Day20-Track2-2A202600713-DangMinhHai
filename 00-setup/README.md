# 00 — Setup

Probes your laptop, picks the right llama.cpp backend, installs Python deps, and downloads a GGUF model sized for your RAM.

## Run

```bash
# Linux
bash 00-setup/linux-setup.sh

# macOS (Apple Silicon or Intel)
bash 00-setup/macos-setup.sh

# Windows (PowerShell 7+)
pwsh -ExecutionPolicy Bypass -File 00-setup\windows-setup.ps1
```

The script:

1. Creates a `.venv/` virtualenv at the repo root.
2. Installs `requirements.txt`.
3. Builds **`llama-cpp-python`** with the right backend flag for your hardware:
   - NVIDIA → `CMAKE_ARGS="-DGGML_CUDA=on"` (env override: `LLAMA_CUDA=1`)
   - Apple Silicon → `CMAKE_ARGS="-DGGML_METAL=on"` (auto on `arm64` macOS)
   - Intel/AMD with Vulkan → `CMAKE_ARGS="-DGGML_VULKAN=on"` (env override: `LLAMA_VULKAN=1`)
   - Everything else → CPU prebuilt wheel
4. Runs `detect-hardware.py` and writes `hardware.json` at the repo root.
5. Runs `download-model.py` and writes `models/active.json` pointing at the GGUF files.

## Outputs

- `hardware.json` — read by every other track to branch behaviour
- `models/<repo>/<file>.gguf` — primary + comparison quantization
- `models/active.json` — paths to the downloaded files

## Override the auto-pick

Set env vars **before** running the setup script:

| Variable | Effect |
|---|---|
| `LLAMA_CUDA=1` | Force a CUDA build (Linux/Windows + NVIDIA) |
| `LLAMA_VULKAN=1` | Force a Vulkan build (cross-vendor) |
| `PYTHON=python3.11` | Use a specific Python interpreter (macOS) |
| `LAB_MODEL=Llama-3.2-3B-Instruct` | Force a specific GGUF tier to download |

## Manual model download

If your network blocks Hugging Face, follow [`MANUAL-DOWNLOAD.md`](MANUAL-DOWNLOAD.md) — drop the `.gguf` into `models/` yourself and re-run `python 00-setup/download-model.py --skip-download` to write `models/active.json`.

## What's next

```bash
source .venv/bin/activate          # Linux / macOS
# .\.venv\Scripts\Activate.ps1     # Windows

cd 01-llama-cpp-quickstart && cat README.md
```
