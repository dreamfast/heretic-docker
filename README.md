# heretic-docker

Docker container for running [Heretic](https://github.com/p-e-w/heretic) LLM abliteration on NVIDIA GPUs.

Produces ComfyUI-compatible text encoder formats (with vision preserved) in five quantization levels — FP8, INT8 (ConvRot), NVFP4, MXFP8 — plus GGUF quants for llama.cpp.

## What it does

1. **Abliterate** any HuggingFace model using Heretic (git master, interactive, you pick the trial)
2. **Convert** to ComfyUI text encoder format (vision preserved, tokenizer embedded)
3. **Quantize** to five formats:
   - **FP8** (float8_e4m3fn, per-tensor scaled via convert-to-quant)
   - **INT8** (block-wise, ConvRot learned rounding — near-lossless, works on any GPU)
   - **NVFP4** (4-bit float E2M1, double quantization, Blackwell-optimized)
   - **MXFP8** (Microscaling FP8, OCP MX standard, E8M0 block scales, Blackwell)
4. **GGUF** conversion with multiple quantization levels via llama.cpp

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

## Automated sweep

Sometimes a single Heretic run doesn't find a good abliteration. The `sweep` command automates running Heretic multiple times with different random seeds and collecting the best LoRA adapters:

```bash
# Run 10 sweeps with 200 trials each, collect best adapters
./heretic.sh sweep Qwen/Qwen3-4B --runs 10 --trials 200 --batch-size 64

# Fewer trials per run (faster but less thorough)
./heretic.sh sweep Qwen/Qwen3-4B --runs 20 --trials 50 --batch-size 64

# With quantization for large models
./heretic.sh sweep google/gemma-3-27b-it --runs 10 --trials 100 --quantization BNB_4BIT
```

Each run uses a unique random seed, so the optimization explores different parameter
combinations. After all runs complete, you get:
- A ranked summary table of all adapters (sorted by refusals, then KL divergence)
- Each adapter named with model, seed, trial number, KL, and refusal count
- The checkpoint JSONL alongside each adapter for full reproducibility
- A prompt to merge the best adapter into the base model

Output structure:
```
output/sweep-20260714-120000/
  run-0/qwen3-4b-s12345-t78_kl0.0001_r2/
    adapter_model.safetensors
    adapter_config.json
    checkpoint.jsonl        # full Optuna study for reproducibility
    metrics.json            # trial scores
  run-1/qwen3-4b-s67890-t45_kl0.0003_r1/
    ...
  summary.txt               # ranked table of all adapters
  merged/                   # (if you chose to merge) full model
```

Sweep options:
- `--runs N` — Number of independent runs (default: 10)
- `--trials N` — Trials per run (default: 200)
- `--batch-size N` — Fixed batch size, skips auto-detection (default: 64)
- Extra flags pass through to heretic (e.g. `--quantization BNB_4BIT`)

## Output formats

After running the conversion pipeline, `./output/` contains:

### ComfyUI safetensors (with vision)

| Path | Format | Size (12B) | HW | Description |
|------|--------|------------|-----|-------------|
| `comfyui/<name>.safetensors` | bf16 | ~23 GB | Any | Full precision |
| `comfyui/<name>_fp8_e4m3fn.safetensors` | FP8 E4M3 | ~12 GB | Ada+ | Per-tensor scaled (convert-to-quant) |
| `comfyui/<name>_int8.safetensors` | INT8 | ~13 GB | Any | Block-wise with ConvRot learned rounding |
| `comfyui/<name>_nvfp4.safetensors` | NVFP4 E2M1 | ~7.8 GB | Blackwell | 4-bit float, double quantization |
| `comfyui/<name>_mxfp8.safetensors` | MXFP8 | ~13 GB | Blackwell | Microscaling FP8, E8M0 block scales |

All ComfyUI formats strip the `language_model.*` prefix and embed the tokenizer as a `spiece_model` tensor. Vision weights (`vision_model.*` and `multi_modal_projector.*`) are preserved for I2V prompt enhancement. The vision weights add minimal overhead (~1 GB) and are simply unused during T2V.

### Format details

**FP8 (E4M3)** — Per-tensor scaled quantization via [convert-to-quant](https://github.com/silveroxides/convert_to_quant). Falls back to naive cast if CTQ is unavailable. Works on Ada (RTX 4090) and newer.

**INT8 (ConvRot)** — Block-wise symmetric INT8 [-127, 127] with [ConvRot](https://github.com/silveroxides/convert_to_quant) learned rounding optimization. For each weight tensor, an SVD-guided gradient descent loop (Prodigy optimizer) learns the optimal rounding direction to minimize output error. This produces near-lossless INT8 quality — significantly better than naive round-to-nearest. Works on any modern GPU (Ampere+). Vision encoder weights are excluded due to non-standard dimensions.

**NVFP4 (E2M1)** — 4-bit floating point with double quantization (per-tensor f32 scale + per-block FP8 e4m3 scale, block size 16). Each quantized weight stores packed FP4 data, block scales, tensor scale, and `comfy_quant` metadata. Pure-PyTorch implementation with optional [comfy_kitchen](https://github.com/Comfy-Org/comfy-kitchen) CUDA acceleration. Requires SM100+ (Blackwell) for native FP4 tensor cores; software dequant works on older GPUs.

**MXFP8** — Microscaling FP8 (OCP MX standard). FP8 E4M3 data with E8M0 (power-of-2 exponent) per-block scales using 32-element blocks. Better dynamic range handling than per-tensor FP8. Requires SM100+ (Blackwell) for hardware-accelerated dequant. Quantized via convert-to-quant.

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

## Running individual stages

All stages support **skip-if-exists** caching — re-running the pipeline will skip any format that's already been generated.

```bash
# Full pipeline (ComfyUI formats + GGUF)
./heretic.sh convert /output/hf-model my-model-name

# ComfyUI formats only (bf16 + fp8 + int8 + nvfp4 + mxfp8, no GGUF)
./heretic.sh comfyui /output/hf-model my-model-name

# GGUF conversion only
./heretic.sh gguf /output/hf-model my-model-name

# Run individual quantization scripts
./heretic.sh run python3 /scripts/quantize_int8.py /output/comfyui/input.safetensors /output/comfyui/output_int8.safetensors
./heretic.sh run python3 /scripts/quantize_mxfp8.py /output/comfyui/input.safetensors /output/comfyui/output_mxfp8.safetensors

# Open a shell for debugging
./heretic.sh shell
```

## GPU selection

Select a GPU with the `GPU_ID` environment variable:

```bash
# Use GPU 0 (default)
./heretic.sh convert /output/hf-model my-model-name

# Use GPU 1
GPU_ID=1 ./heretic.sh convert /output/hf-model my-model-name
```

## INT8 ConvRot notes

The INT8 ConvRot stage is significantly slower than other formats because it runs a per-tensor optimization loop (up to 4000 iterations of Prodigy optimizer with SVD projection). For a 12B model with ~500 weight tensors, expect 30-60 minutes on GPU (vs seconds for FP8/MXFP8). Early stopping kicks in automatically when the learning rate bottoms out.

The optimization runs on CUDA when available, falling back to CPU (much slower). Vision encoder weights are excluded from INT8 quantization due to non-standard tensor dimensions.

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
├── Dockerfile              # NGC PyTorch base + heretic + transformers + convert-to-quant + comfy-kitchen
├── docker-compose.yml      # Single heretic service (GPU, volumes, UID/GID)
├── entrypoint.sh           # UID/GID matching via gosu
├── .env.example            # HuggingFace token template
├── .dockerignore
├── .gitignore
├── patches/
│   ├── blackwell_compat.py         # bitsandbytes stub for CUDA 13.1
│   ├── patch_hf_union_types.py     # huggingface_hub PEP 604 union type fix
│   ├── patch_hub_kernels.py        # transformers hub_kernels stub
│   └── patch_tokenizer_special_tokens.py
└── scripts/
    ├── convert_all.sh              # Full pipeline (7 stages, skip-if-exists)
    ├── convert_comfyui.sh          # ComfyUI stages only (6 stages, skip-if-exists)
    ├── merge_safetensors.py        # Merge shards, keep all keys (vision intact)
    ├── convert_comfyui_vision.py   # ComfyUI format with vision preserved
    ├── quantize_fp8.py             # FP8 e4m3fn (CTQ per-tensor scaled, naive fallback)
    ├── quantize_int8.py            # INT8 block-wise with ConvRot learned rounding (CTQ)
    ├── quantize_nvfp4.py           # NVFP4 E2M1 4-bit (double quantization, comfy_kitchen)
    ├── quantize_mxfp8.py           # MXFP8 microscaling FP8 (CTQ, E8M0 block scales)
    ├── convert_gguf.sh             # GGUF conversion + quantization via llama.cpp
    ├── compare_models.py           # Debug utility: compare tensor keys between files
    ├── parse_checkpoint.py         # Parse Optuna checkpoint JSONL for sweep metrics
    └── merge_lora.py               # Merge LoRA adapter into base model
```

## Credits

- [Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann
- [convert-to-quant](https://github.com/silveroxides/convert_to_quant) by silveroxides — INT8 ConvRot, FP8 scaling, MXFP8 quantization
- [comfy_kitchen](https://github.com/Comfy-Org/comfy-kitchen) by Comfy-Org — NVFP4/MXFP8 CUDA kernels
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF conversion and quantization
