AirStudio
=========

Local Gradio UI + AirLLM engine for chat, per-request metrics, and optional benchmarks.


Whats in this repo
-------------------
- Gradio UI (chat + benchmark)
- AirLLM engine integration (layer-wise loading + optional 4/8-bit compression)
- Metrics per request (tokens/s, peak RAM, peak VRAM, decode time)
- Benchmark runner that exports CSV/JSON into `runs/`

Quick start (Windows)
---------------------
1) Install
```
.\install.bat
```

2) Download models
```
.\models.bat
```
Optional: choose specific models (comma-separated):
```
.\models.bat mistral7b
.\models.bat llama8b
.\models.bat mistral7b,llama8b
```
If you need a HF token for gated models:
```
set HF_TOKEN=hf_xxx
.\models.bat
```

3) Run UI
```
.\run.bat
```


CLI options
------------------------------------
`run.bat` calls `python -m airstudio.main` and supports:
- `--config` (default: `configs/airstudio.yaml`)
- `--host`
- `--port`
- `--title`
- `--concurrency-limit`
- `--sampling-interval-ms`
- `--gpu-index`
- `--offline`
- `--share`

Example:
```
.\run.bat --config configs\airstudio.yaml --host 127.0.0.1 --port 7860
```


Configuration
-------------
Main config: `configs/airstudio.yaml`

Current defaults:
- `generation_defaults.max_new_tokens: 16`
- `generation_defaults.max_context: 4096`
- `do_sample: false`, `temperature: 0.0` (deterministic)
- Profiles:
  - `low_vram`: 4-bit (requires bitsandbytes)
  - `balanced`: 8-bit (requires bitsandbytes)
  - `quality`: no compression

Models configured (local paths):
- Mistral 7B Instruct: `./models/Mistral-7B-Instruct-v0.3`
- Llama 3.1 8B Instruct: `./models/Llama-3.1-8B-Instruct`


Install flow notes
------------------
`install.bat`:
- Creates `venv/`
- Installs project dependencies
- Installs CUDA-enabled PyTorch from the cu128 index (unless `AIRSTUDIO_TORCH=cpu`)
- Ensures `protobuf` is installed
- Installs `nvidia-ml-py` for VRAM metrics
- Installs `bitsandbytes` (optional) unless `AIRSTUDIO_BNB=skip`
- Creates `models/`, `cache/`, and `runs/` if missing
wu