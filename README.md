# heretic-docker

Docker container for running [Heretic](https://github.com/p-e-w/heretic) LLM abliteration and fine-tuning on NVIDIA GPUs.

Produces ComfyUI-compatible text encoder formats (with vision preserved), FP8/NVFP4 quantized variants, and GGUF quants for llama.cpp. Includes [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for SFT fine-tuning.

## What it does

1. **Abliterate** any HuggingFace model using Heretic (git master, interactive, you pick the trial)
2. **Fine-tune** with Axolotl SFT (LoRA, Qwen3.5 support)
3. **Convert** to ComfyUI text encoder format (vision preserved, tokenizer embedded)
4. **Quantize** to FP8 (float8_e4m3fn), NVFP4 (ComfyUI-native, Blackwell-optimized)
5. **GGUF** conversion with multiple quantization levels via llama.cpp

## Requirements

- NVIDIA GPU with latest drivers
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- HuggingFace account with access to gated models (if targeting gated models)

## Quick start

```bash
# Clone and enter the repo
git clone https://github.com/dreamfast/heretic-docker.git
cd heretic-docker

# Set up your HuggingFace token
cp .env.example .env
# Edit .env and add your HF token

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

Replace the model ID as needed:

```bash
./heretic.sh abliterate meta-llama/Llama-3.1-8B-Instruct
./heretic.sh abliterate Qwen/Qwen3.5-9B
./heretic.sh abliterate mistralai/Mistral-7B-Instruct-v0.3
```

You can also pass Heretic CLI flags **after** the model name:

```bash
# Custom trial count
./heretic.sh abliterate google/gemma-3-12b-it --n-trials 100

# Use LoRA-based abliteration with 4-bit quantization (for large models)
./heretic.sh abliterate google/gemma-3-27b-it --quantization BNB_4BIT
```

**Important:** The model name must come first, flags come after.

## Fine-tuning with Axolotl

After abliteration, you can fine-tune the model with your own SFT data using Axolotl.

```bash
# 1. Place your ShareGPT-format data in ./data/
cp my_sft_data.jsonl ./data/

# 2. Fine-tune (model path optional, defaults to /output/hf-model)
./heretic.sh finetune sft.jsonl /output/hf-model

# 3. Merge LoRA adapter into base model
./heretic.sh merge
```

The default config uses LoRA (rank 16, alpha 32), bf16, with sample packing enabled. Edit `./configs/sft.yml` to customize.

For more details, see the [Axolotl documentation](https://github.com/axolotl-ai-cloud/axolotl).

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

GGUF files are text-only (no vision). They work with llama.cpp directly and with ComfyUI via [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

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

## GPU selection

Select a GPU with `CUDA_VISIBLE_DEVICES`:

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 ./heretic.sh abliterate Qwen/Qwen3.5-9B

# Use GPU 1
CUDA_VISIBLE_DEVICES=1 ./heretic.sh abliterate Qwen/Qwen3.5-9B
```

## File permissions

The container matches your host user's UID/GID so all output files are owned by you. The `HOST_UID` and `HOST_GID` environment variables are passed automatically.

## How model downloads work

Models are downloaded to `./models/` (mounted as `/models` in the container, used as `HF_HOME`). This means:

- Models persist between runs (no re-downloading)
- Models are on your host filesystem, not buried in Docker layers
- You can pre-download models or share the cache between projects

## Project structure

```
.
├── heretic.sh              # Helper script (./heretic.sh --help)
├── Dockerfile              # NGC PyTorch base + heretic (git master) + transformers (git master)
├── Dockerfile.axolotl      # Axolotl with Qwen3.5 support
├── docker-compose.yml      # Services: heretic (interactive) + convert + axolotl
├── entrypoint.sh           # UID/GID matching via gosu
├── .env.example            # HuggingFace token template
├── .dockerignore
├── .gitignore
├── patches/
│   └── blackwell_compat.py # bitsandbytes stub for CUDA 13.1
├── configs/                # Axolotl SFT configs (user-provided)
├── data/                   # Training data (user-provided)
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
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for SFT fine-tuning
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF conversion and quantization
- [comfy_kitchen](https://github.com/Comfy-Org/comfy-kitchen) for NVFP4 quantization
