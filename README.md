# heretic-docker

Docker container for running [Heretic](https://github.com/p-e-w/heretic) LLM abliteration on NVIDIA GPUs.

Outputs:

- ComfyUI text-encoder formats with vision preserved, in five quantization levels: FP8, INT8, INT4, NVFP4, MXFP8
- GGUF quants for llama.cpp, imatrix-calibrated, with a separate mmproj vision file for VLMs
- vLLM-ready NVFP4 checkpoints (compressed-tensors format, W4A4)

## What it does

1. **Abliterate** any HuggingFace model with Heretic v2 (interactive, you pick the trial).
   - v2 defaults to ARA (Arbitrary-Rank Abliteration). Classic directional-ablation abliteration is still available via config, see the [v1 to v2 migration guide](https://github.com/p-e-w/heretic/blob/master/docs/migration_v1_to_v2.md).
2. **Convert** to ComfyUI text-encoder format. Vision weights and tokenizer are kept.
3. **Quantize** to five formats:
   - **FP8** (e4m3fn, per-tensor scaled via convert-to-quant)
   - **INT8** (block-wise, ConvRot learned rounding, near-lossless, any GPU)
   - **INT4** (W4A4, ConvRot + AdaRound learned rounding, about half the size of INT8)
   - **NVFP4** (4-bit float E2M1, double quantization, Blackwell)
   - **MXFP8** (microscaling FP8, OCP MX standard, E8M0 block scales, Blackwell)
4. **GGUF** conversion with imatrix-calibrated quantization via llama.cpp.
   - VLMs additionally get an mmproj file so vision keeps working in llama.cpp.
5. **NVFP4 for vLLM**: compressed-tensors W4A4, with optional dynamic mixed precision (Unsloth-style).

## Requirements

- NVIDIA GPU with current drivers
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- HuggingFace account with access to gated models, if you target gated models

## Quick start

```bash
git clone https://github.com/dreamfast/heretic-docker.git
cd heretic-docker

cp .env.example .env
# edit .env and add your HF token

./heretic.sh build          # first build takes a while, the NGC base is ~20 GB
```

Run abliteration:

```bash
./heretic.sh abliterate google/gemma-3-12b-it
```

When optimization finishes:

1. Pick a trial from the Pareto menu (low refusals, low KL divergence).
2. Choose "Save the model to a local folder".
3. Pick an export strategy: merged model, or LoRA adapter only.
4. Enter `/output/hf-model` as the path.

Then run the conversion pipeline:

```bash
./heretic.sh convert /output/hf-model my-model-name
```

## Using with any HuggingFace model

Pass the model ID first, heretic flags after:

```bash
./heretic.sh abliterate meta-llama/Llama-3.1-8B-Instruct
./heretic.sh abliterate Qwen/Qwen3.5-9B
./heretic.sh abliterate google/gemma-3-12b-it --n-trials 100
./heretic.sh abliterate google/gemma-3-27b-it --quantization BNB_4BIT   # large models
```

## Automated sweep

A single run does not always find a good trial. `sweep` runs heretic multiple times with different seeds, exports the top trials headlessly, and ranks the adapters:

```bash
./heretic.sh sweep Qwen/Qwen3-4B --runs 10 --trials 200 --batch-size 64
./heretic.sh sweep google/gemma-3-27b-it --runs 10 --trials 100 --quantization BNB_4BIT
```

Options:

- `--runs N`: independent runs (default 10)
- `--trials N`: trials per run, maps to heretic `--n-trials` (default 200)
- `--batch-size N`: fixed batch size, skips auto-detection (default 64)
- `--n-top-trials N`: export top N Pareto trials per run (default 1; extra trials re-export from the finished study)
- `--keep N`: after the sweep, keep only the best N adapters and delete the rest
- Any other flags pass through to heretic, e.g. `--quantization BNB_4BIT`

Output:

```
output/sweep-<timestamp>/
  <model>-r<run>-s<seed>-t<trial>_kl<kl>_r<refusals>/
    adapter_model.safetensors
    adapter_config.json
    checkpoint.jsonl      # full Optuna study, for reproducibility
    metrics.json          # trial scores
  summary.txt             # ranked table of all adapters
  merged/                 # only if you merge at the prompt
```

- Adapters are ranked by `refusals + KL * 10` (lower is better).
- After the sweep you are prompted to merge the best adapter into the base model.
- `./heretic.sh prune <sweep_dir> [--keep N]` deletes all but the best N adapters.

## Output formats

After `convert`, `./output/` contains:

### ComfyUI safetensors (with vision)

| Path | Format | Size (12B) | HW | Description |
|------|--------|------------|-----|-------------|
| `comfyui/<name>.safetensors` | bf16 | ~23 GB | Any | Full precision |
| `comfyui/<name>_fp8_e4m3fn.safetensors` | FP8 E4M3 | ~12 GB | Ada+ | Per-tensor scaled (convert-to-quant) |
| `comfyui/<name>_int8.safetensors` | INT8 | ~13 GB | Any | Block-wise with ConvRot learned rounding |
| `comfyui/<name>_int4.safetensors` | INT4 W4A4 | ~7 GB | Blackwell (MMA) | ConvRot + AdaRound learned rounding, packed 2-per-byte |
| `comfyui/<name>_nvfp4.safetensors` | NVFP4 E2M1 | ~7.8 GB | Blackwell | 4-bit float, double quantization |
| `comfyui/<name>_mxfp8.safetensors` | MXFP8 | ~13 GB | Blackwell | Microscaling FP8, E8M0 block scales |

Notes:

- All formats strip the `language_model.*` prefix and embed the tokenizer as a `spiece_model` tensor.
- Vision weights (`vision_model.*`, `multi_modal_projector.*`) are preserved for I2V prompt enhancement.
- The vision weights add about 1 GB and are unused during T2V.

### Format details

- **FP8 (E4M3)**: per-tensor scaled via [convert-to-quant](https://github.com/silveroxides/convert_to_quant). Falls back to a naive cast if CTQ is unavailable. Works on Ada (RTX 4090) and newer.
- **INT8 (ConvRot)**: block-wise symmetric INT8 [-127, 127] with ConvRot learned rounding. An SVD-guided gradient descent loop (Prodigy optimizer) learns the rounding direction per tensor to minimize output error. Near-lossless, significantly better than round-to-nearest. Works on Ampere and newer. Vision encoder weights are excluded due to non-standard tensor dimensions.
- **INT4 (W4A4 ConvRot)**: signed INT4 [-7, 7], per-row scaling, packed two per byte. A group-wise Hadamard rotation (ConvRot) spreads outliers so 4-bit stays viable, and the same AdaRound loop as INT8 optimizes rounding (~30 dB vs ~20 dB for round-to-nearest). `--dynamic-convrot` picks a compatible group size per layer. About 50% smaller than INT8. ComfyUI 0.30.0+ accelerates W4A4 matmuls via [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) on Blackwell tensor cores; software dequant works elsewhere.
- **NVFP4 (E2M1)**: 4-bit float with double quantization (per-tensor f32 scale plus per-block FP8 e4m3 scale, block size 16). Packed FP4 data with `comfy_quant` metadata. Optional comfy-kitchen CUDA acceleration. Needs SM100+ (Blackwell) for FP4 tensor cores; software dequant works on older GPUs.
- **MXFP8**: FP8 E4M3 data with E8M0 per-block scales, 32-element blocks (OCP MX). Better dynamic range than per-tensor FP8. Needs SM100+ for accelerated dequant. Quantized via convert-to-quant.

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
| Q3_K_M | ~5.6 GB | Low VRAM only |

Notes:

- Text GGUFs exclude vision weights. Multimodal models additionally produce `gguf/<name>-mmproj-f16.gguf`, the vision projector. Point llama.cpp at it with `--mmproj`.
- The mmproj is exported at F16 and never quantized.
- Calibration data is not included in the repo. Set `CALIBRATION_FILE=<file>` to enable imatrix-calibrated quants; without it the script skips imatrix and produces plain RTN quants. `SKIP_IMATRIX=1` forces the same fallback.
- GGUFs work with llama.cpp directly and with ComfyUI via [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

### Other outputs

| Path | Description |
|------|-------------|
| `<name>/` | Full HuggingFace model (shards + config + tokenizer) |
| `merged/<name>-full.safetensors` | Single merged safetensors with all keys |

## Running individual stages

All stages skip-if-exists: re-running the pipeline only builds missing formats.

```bash
# Full pipeline (ComfyUI formats + GGUF)
./heretic.sh convert /output/hf-model my-model-name

# ComfyUI formats only (bf16 + fp8 + int8 + int4 + nvfp4 + mxfp8, no GGUF)
./heretic.sh comfyui /output/hf-model my-model-name

# GGUF conversion only
./heretic.sh gguf /output/hf-model my-model-name

# Individual quantization scripts
./heretic.sh run python3 /scripts/quantize_int8.py /output/comfyui/input.safetensors /output/comfyui/output_int8.safetensors
./heretic.sh run python3 /scripts/quantize_mxfp8.py /output/comfyui/input.safetensors /output/comfyui/output_mxfp8.safetensors

# Merge a LoRA adapter into a base model
./heretic.sh merge Qwen/Qwen3-4B /output/my-adapter /output/merged

# Debug shell
./heretic.sh shell
```

## NVFP4 for vLLM

Produces vLLM-compatible NVFP4 checkpoints (compressed-tensors, W4A4). Pass a HuggingFace model name; it downloads, quantizes, and saves a directory ready for `vllm serve`.

- Produce on any GPU (e.g. RTX 4090), serve on Blackwell (RTX 5090, B200) for FP4 tensor-core acceleration.

```bash
# Baseline NVFP4, fastest to produce
./heretic.sh nvfp4-vllm Qwen/Qwen3-8B

# Max quality: GPTQ + SmoothQuant + dynamic mixed precision + hermes-mix + FP8 KV cache
./heretic.sh nvfp4-vllm Qwen/Qwen3-8B \
    --mixed-precision dynamic --samples 512 --dataset hermes-mix --kv-cache-fp8
```

Output: `/output/nvfp4-vllm/<model>-NVFP4/`, a full HF model directory:

```bash
vllm serve /output/nvfp4-vllm/Qwen3-8B-NVFP4
```

### Quality ladder (composable flags)

| Flag | Effect |
|------|--------|
| *(default)* | Uniform NVFP4 W4A4 on all Linear layers |
| `--mixed-precision downproj` | Keep `down_proj` in FP8, rest NVFP4. Large quality win, small size cost. |
| `--mixed-precision dynamic` | Per-layer sensitivity analysis keeps the worst layers in FP8 (Unsloth-style) |
| `--samples 512` | More calibration samples, more stable activation scales |
| `--dataset hermes-mix` | OpenHermes + code + math + tool-calling blend, best for reasoning and code |
| `--kv-cache-fp8` | Calibrate KV cache to FP8, roughly 2x longer context |
| `--gptq` *(default on)* | Hessian-guided weight quantization. 2-5x slower, significantly better quality. |
| `--smoothquant` *(default on)* | Smooth outlier activation channels. Critical for W4A4. |
| `--no-gptq` | Round-to-nearest weights instead of GPTQ (faster, lower quality) |
| `--no-smoothquant` | Disable activation smoothing |

Flags compose freely:

```bash
# Balanced
./heretic.sh nvfp4-vllm Qwen/Qwen3-8B --mixed-precision downproj --samples 512 --dataset hermes-mix

# Down_proj FP8 + better calibration only
./heretic.sh nvfp4-vllm Llama-3-8B --mixed-precision downproj --samples 512

# Fastest (round-to-nearest)
./heretic.sh nvfp4-vllm Qwen/Qwen3-8B --no-gptq --no-smoothquant

# Dynamic sensitivity, top-20 layers in FP8
./heretic.sh nvfp4-vllm Qwen/Qwen3-8B --mixed-precision dynamic --dynamic-topk 20
```

Hardware:

- Produce: any NVIDIA GPU. Calibration runs BF16, FP4 packing is uint8. Tested on RTX 4090 (24 GB). Models larger than VRAM trigger automatic sequential onloading.
- Serve: SM100+ (RTX 5090, B200, DGX Spark) for W4A4 FP4 tensor cores, about 2.5x decode speedup. Pre-Blackwell vLLM falls back to W4A16.

## GPU selection

```bash
GPU_ID=1 ./heretic.sh convert /output/hf-model my-model-name   # default is 0
```

## ConvRot notes (INT8 and INT4)

- The ConvRot stages run a per-tensor optimization loop (up to 4000 Prodigy iterations with SVD projection), so they are much slower than FP8/MXFP8.
- A 12B model with ~500 weight tensors takes 30-60 minutes on GPU for INT8. INT4 is similar.
- Early stopping kicks in when the learning rate bottoms out.
- Optimization runs on CUDA when available and falls back to CPU (much slower).
- Vision encoder weights are excluded due to non-standard tensor dimensions.

## File permissions

- The container runs as your host UID/GID via gosu, so all output files are owned by you.

## Model cache

- Models download to `./models/` (mounted at `/models`, used as `HF_HOME`).
- Downloads persist between runs, so nothing is fetched twice.
- The cache sits on your host filesystem, not in Docker layers. Pre-download or share it between projects if you like.

## Project structure

```
.
├── heretic.sh              # Helper script (./heretic.sh --help)
├── Dockerfile              # NGC PyTorch base + heretic v2 + llama.cpp + quantization toolchain
├── docker-compose.yml      # Single heretic service (GPU, volumes, UID/GID)
├── entrypoint.sh           # UID/GID matching via gosu
├── .env.example            # HuggingFace token template
├── patches/
│   ├── convert_to_quant_int4.patch        # INT4 W4A4 ConvRot for convert-to-quant (PR #55)
│   ├── patch_hub_kernels.py               # transformers hub_kernels stub
│   ├── patch_tokenizer_special_tokens.py  # tokenizer extra_special_tokens as list
│   ├── blackwell_compat.py                # bitsandbytes stub for CUDA 13.1
│   └── patch_sdpa.py                      # optional Blackwell SDPA fix (disabled)
└── scripts/
    ├── convert_all.sh              # Full pipeline, skip-if-exists
    ├── convert_comfyui.sh          # ComfyUI stages only, skip-if-exists
    ├── convert_gguf.sh             # GGUF + imatrix quants + mmproj for VLMs
    ├── quantize_all.sh             # Quantize an existing safetensors into all formats
    ├── convert_comfyui_vision.py   # ComfyUI format with vision preserved
    ├── merge_safetensors.py        # Merge shards, keep all keys (vision intact)
    ├── quantize_fp8.py             # FP8 e4m3fn (CTQ per-tensor scaled, naive fallback)
    ├── quantize_int8.py            # INT8 block-wise, ConvRot learned rounding
    ├── quantize_int4.py            # INT4 W4A4, ConvRot + AdaRound learned rounding
    ├── quantize_nvfp4.py           # NVFP4 E2M1 (double quantization, comfy-kitchen)
    ├── quantize_mxfp8.py           # MXFP8 microscaling FP8 (CTQ, E8M0 block scales)
    ├── nvfp4_vllm.py               # NVFP4 W4A4 for vLLM (llm-compressor, compressed-tensors)
    ├── sensitivity.py              # Per-layer FP4 sensitivity analysis (dynamic mixed precision)
    ├── parse_checkpoint.py         # Parse Optuna checkpoint JSONL for sweep metrics
    ├── merge_lora.py               # Merge LoRA adapter into base model
    ├── average_loras.py            # Average multiple LoRA adapters
    ├── compare_models.py           # Compare tensor keys between files
    └── tensor_diff.py              # Numerical diff between safetensors
```

## Credits

- [Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann
- [convert-to-quant](https://github.com/silveroxides/convert_to_quant) by silveroxides: INT8/INT4 ConvRot, FP8 scaling, MXFP8
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) by Comfy-Org: NVFP4/MXFP8/INT4 CUDA kernels
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF conversion and quantization
- [llm-compressor](https://github.com/vllm-project/llm-compressor) by the vLLM Project: NVFP4 W4A4 for vLLM
