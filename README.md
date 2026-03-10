# heretic-docker

Docker container for running [Heretic](https://github.com/p-e-w/heretic) LLM abliteration on NVIDIA Blackwell GPUs (RTX 5090, RTX 5080, etc).

Produces ComfyUI-compatible text encoder formats (with vision preserved), FP8/NVFP4 quantized variants, and GGUF quants for llama.cpp.

## What it does

1. **Abliterate** any HuggingFace model using Heretic v1.2.0 (interactive, you pick the trial)
2. **Convert** to ComfyUI text encoder format (vision preserved, tokenizer embedded)
3. **Quantize** to FP8 (float8_e4m3fn), NVFP4 (ComfyUI-native, Blackwell-optimized)
4. **GGUF** conversion with multiple quantization levels via llama.cpp

## Requirements

- NVIDIA GPU with latest drivers (tested on RTX 5090)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- HuggingFace account with access to gated models (if targeting gated models like Gemma)

## Quick start

```bash
# Clone and enter the repo
git clone https://github.com/YOUR_USER/heretic-docker.git
cd heretic-docker

# Set up your HuggingFace token
cp .env.example .env
# Edit .env and add your HF token

# Create output directories
mkdir -p output models

# Build the container (first time takes a while - NGC base is ~20GB)
./heretic.sh build

# Run Heretic abliteration (interactive)
./heretic.sh abliterate google/gemma-3-12b-it
```

When Heretic finishes optimization and presents the Pareto menu:
1. Pick a trial (low refusals + low KL divergence)
2. Choose "Save the model to a local folder"
3. Enter `/output/hf-model` as the save path

Then run the conversion pipeline:

```bash
./heretic.sh convert /output/hf-model my-model-name
```

## Using with any HuggingFace model

Replace `google/gemma-3-12b-it` with any model ID:

```bash
./heretic.sh abliterate meta-llama/Llama-3.1-8B-Instruct
./heretic.sh abliterate Qwen/Qwen2.5-7B-Instruct
./heretic.sh abliterate mistralai/Mistral-7B-Instruct-v0.3
```

You can also pass Heretic CLI flags:

```bash
# Custom trial count
./heretic.sh abliterate --n-trials 100 google/gemma-3-12b-it

# Use LoRA-based abliteration with 4-bit quantization (for large models)
./heretic.sh abliterate --quantization BNB_4BIT google/gemma-3-27b-it
```

## Output formats

After running the conversion pipeline, `./output/` contains:

### ComfyUI safetensors (with vision)

| Path | Description | Size (12B) |
|------|-------------|------------|
| `comfyui/<name>.safetensors` | bf16 | ~23 GB |
| `comfyui/<name>_fp8_e4m3fn.safetensors` | fp8 | ~12 GB |
| `comfyui/<name>_nvfp4.safetensors` | nvfp4 | ~7.8 GB |

All ComfyUI formats strip the `language_model.*` prefix and embed the tokenizer as a `spiece_model` tensor. Vision weights (`vision_model.*` and `multi_modal_projector.*`) are preserved for I2V prompt enhancement. The vision weights add minimal overhead (~1 GB) and are simply unused during T2V.

### GGUF (for llama.cpp)

| Quant | Size (12B) | Notes |
|-------|------------|-------|
| F16 | ~22 GB | Lossless reference |
| Q8_0 | ~12 GB | Excellent quality |
| Q6_K | ~9 GB | Very good quality |
| Q5_K_M | ~8 GB | Good quality |
| Q5_K_S | ~7.7 GB | Slightly smaller Q5 |
| Q4_K_M | ~6.8 GB | Recommended balance |
| Q4_K_S | ~6.5 GB | Smaller Q4 variant |
| Q3_K_M | ~5.6 GB | For low VRAM only |

GGUF files are text-only (no vision). They work with llama.cpp directly and with ComfyUI via [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (Gemma 3 support merged in [PR #402](https://github.com/city96/ComfyUI-GGUF/pull/402)). When using GGUF in ComfyUI, load via the `DualClipLoader (GGUF)` node with embedding connectors from [Kijai/LTXV2_comfy](https://huggingface.co/Kijai/LTXV2_comfy/tree/main/text_encoders) (use the **dev** connectors, not distilled).

### Other outputs

| Path | Description |
|------|-------------|
| `<name>/` | Full HuggingFace model (shards + config + tokenizer) |
| `merged/<name>-full.safetensors` | Single merged safetensors with all keys |

### NVFP4 quantization

The NVFP4 (4-bit floating point, E2M1 format) variants use ComfyUI's native quantization format via [comfy_kitchen](https://pypi.org/project/comfy-kitchen/). Each quantized weight stores:

- Packed FP4 data (2 values per byte)
- Per-block FP8 scales (block size 16)
- Per-tensor float32 scale (double quantization)
- `comfy_quant` metadata for automatic ComfyUI detection

The quantized files are ~3x smaller than bf16 and load natively in ComfyUI without any plugins. Blackwell GPUs (RTX 5090/5080) use native FP4 tensor cores for best performance, but ComfyUI also supports software dequantization on older GPUs (tested working on RTX 4090).

## Running individual stages

You can run any conversion step independently:

```bash
# Just the GGUF conversion
./heretic.sh gguf /output/hf-model my-model-name

# Re-run ComfyUI variants only (bf16 + fp8 + nvfp4)
./heretic.sh comfyui /output/hf-model my-model-name

# Run an arbitrary command in the container
./heretic.sh run python3 /scripts/quantize_fp8.py /output/comfyui/input.safetensors /output/comfyui/output_fp8.safetensors

# Open a shell for debugging
./heretic.sh shell
```

## GPU configuration

By default, the container uses GPU 0. To change this, set `GPU_ID` in your `.env` file:

```bash
GPU_ID=1  # Use GPU 1 instead
```

## File permissions

The container matches your host user's UID/GID so all output files are owned by you. The `HOST_UID` and `HOST_GID` environment variables are passed automatically.

## How model downloads work

Models are downloaded to `./models/` (mounted as `/models` in the container, used as `HF_HOME`). This means:

- Models persist between runs (no re-downloading)
- Models are on your host filesystem, not buried in Docker layers
- You can pre-download models or share the cache between projects

## Blackwell GPU compatibility

This container includes patches for NVIDIA Blackwell architecture (sm_120) compatibility:

- **Base image**: `nvcr.io/nvidia/pytorch:26.02-py3` (NGC PyTorch with sm_120 CUDA kernels)
- **SDPA attention**: Patched into Heretic's model loading (default Gemma 3 attention kernels lack sm_120 support)
- **bitsandbytes stub**: Stubbed out since no CUDA 13.1 binary exists yet

These patches are transparent - the container also works on older GPUs (RTX 4090, 3090, etc.) since SDPA is universally supported.

## Project structure

```
.
├── heretic.sh              # Helper script (./heretic.sh --help)
├── Dockerfile              # NGC PyTorch base + heretic-llm + llama.cpp + patches
├── docker-compose.yml      # Services: heretic (interactive) + convert
├── entrypoint.sh           # UID/GID matching via gosu
├── .env.example            # HuggingFace token template
├── .dockerignore
├── .gitignore
├── patches/
│   ├── blackwell_compat.py # bitsandbytes stub for CUDA 13.1
│   └── patch_sdpa.py       # Injects SDPA attention into heretic
└── scripts/
    ├── convert_all.sh            # Orchestrates all conversion steps (5 stages)
    ├── merge_safetensors.py      # Merge shards, keep all keys (vision intact)
    ├── convert_comfyui_vision.py # ComfyUI format with vision preserved
    ├── quantize_fp8.py           # FP8 e4m3fn quantization
    ├── quantize_nvfp4.py         # NVFP4 quantization (ComfyUI-native, Blackwell)
    └── convert_gguf.sh           # GGUF conversion + quantization via llama.cpp
```

## Credits

- [Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF conversion and quantization
- [comfy_kitchen](https://github.com/Comfy-Org/comfy-kitchen) for NVFP4 quantization
