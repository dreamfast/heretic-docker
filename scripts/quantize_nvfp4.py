#!/usr/bin/env python3
"""
Quantize a ComfyUI-format safetensors model to NVFP4 (ComfyUI-native format).

This produces files compatible with ComfyUI's mixed_precision_ops / QuantizedTensor
system, matching the format used by comfy_kitchen's TensorCoreNVFP4Layout.

For each float weight tensor "layer.weight", produces:
  - "layer.weight":         uint8 packed FP4 data (two values per byte)
  - "layer.weight_scale":   uint8 (fp8 e4m3 block scales, 1 per block of 16 elements)
  - "layer.weight_scale_2": float32 per-tensor scale
  - "layer.comfy_quant":    uint8 JSON metadata '{"format": "nvfp4"}'

Non-weight tensors (embeddings, norms, bias) and small tensors are left unchanged.

Requires: comfy_kitchen (preferred) OR falls back to pure-PyTorch implementation.
NVFP4 inference requires SM100+ (Blackwell) GPU.

Usage:
    python3 quantize_nvfp4.py input.safetensors output.safetensors
"""

import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

BLOCK_SIZE = 16
FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

# FP4 E2M1 max representable value
FP4_E2M1_MAX = 6.0
# FP8 E4M3 max representable value
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

# FP4 E2M1 representable values (all 16 values for 4-bit encoding)
# Positive: 0, 0.5, 1, 1.5, 2, 3, 4, 6
# Negative: -0, -0.5, -1, -1.5, -2, -3, -4, -6
# Encoding: low nibble index [0..15]
FP4_LOOKUP = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _try_comfy_kitchen():
    """Try to import comfy_kitchen for native quantization."""
    try:
        import comfy_kitchen as ck
        return ck
    except ImportError:
        return None


def _pad_to_block(tensor_2d, block_size=BLOCK_SIZE):
    """Pad a 2D tensor so both dims are multiples of block_size."""
    rows, cols = tensor_2d.shape
    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size
    if pad_rows > 0 or pad_cols > 0:
        tensor_2d = torch.nn.functional.pad(tensor_2d, (0, pad_cols, 0, pad_rows))
    return tensor_2d


def quantize_nvfp4_comfy_kitchen(tensor_2d, ck):
    """Quantize using comfy_kitchen (native C++/CUDA implementation)."""
    # Per-tensor scale: max_abs / (FP8_MAX * FP4_MAX)
    tensor_scale = torch.amax(tensor_2d.abs()) / (FP8_E4M3_MAX * FP4_E2M1_MAX)
    tensor_scale = tensor_scale.to(dtype=torch.float32)

    orig_shape = tensor_2d.shape
    padded_rows = ((orig_shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padded_cols = ((orig_shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padded_shape = (padded_rows, padded_cols)
    needs_padding = padded_shape != orig_shape

    packed, block_scale = ck.quantize_nvfp4(tensor_2d, tensor_scale, pad_16x=needs_padding)

    # comfy_kitchen may return 1D packed tensors — reshape to 2D for ComfyUI compatibility
    # Packed FP4: 2 values per byte, so packed cols = padded_cols // 2
    if packed.ndim == 1:
        packed = packed.reshape(padded_rows, padded_cols // 2)
    if block_scale.ndim == 1:
        block_scale = block_scale.reshape(padded_rows, padded_cols // BLOCK_SIZE)

    return packed, block_scale, tensor_scale


def quantize_nvfp4_manual(tensor_2d):
    """
    Pure PyTorch NVFP4 quantization matching comfy_kitchen format.
    
    Double quantization:
    1. Per-tensor scale normalizes into FP8 range
    2. Per-block (16-element) FP8 scale handles local range
    3. Values quantized to FP4 E2M1 and packed into uint8
    """
    tensor_scale = torch.amax(tensor_2d.abs()).float() / (FP8_E4M3_MAX * FP4_E2M1_MAX)
    if tensor_scale == 0:
        tensor_scale = torch.tensor(1.0, dtype=torch.float32)

    # Pad to block-aligned shape
    padded = _pad_to_block(tensor_2d.float())
    rows, cols = padded.shape

    # Scale by per-tensor scale first
    scaled = padded / tensor_scale

    # Reshape into blocks of 16 along the column dimension
    # ComfyUI processes blocks along the innermost dimension
    flat = scaled.reshape(-1, BLOCK_SIZE)

    # Per-block scales (FP8)
    block_max = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    block_scales = (block_max / FP4_E2M1_MAX).to(torch.float8_e4m3fn)

    # Normalize by block scale and quantize to nearest FP4 value
    block_scales_f32 = block_scales.float()
    normalized = flat / block_scales_f32

    # Find nearest FP4 value for each element
    lookup = FP4_LOOKUP.to(normalized.device)
    diffs = (normalized.unsqueeze(-1) - lookup.unsqueeze(0).unsqueeze(0)).abs()
    indices = diffs.argmin(dim=-1).to(torch.uint8)  # [num_blocks, 16]

    # Pack pairs of FP4 into uint8: low nibble = even index, high nibble = odd index
    indices_flat = indices.reshape(-1)
    even = indices_flat[0::2]
    odd = indices_flat[1::2]
    packed = even | (odd << 4)

    # Reshape packed output to 2D: (padded_rows, padded_cols // 2)
    packed = packed.reshape(rows, cols // 2)

    # Block scales as uint8 (underlying storage of fp8), reshaped to 2D
    block_scales_out = block_scales.squeeze(1).view(torch.uint8)
    block_scales_out = block_scales_out.reshape(rows, cols // BLOCK_SIZE)

    return packed, block_scales_out, tensor_scale.to(torch.float32)


def make_comfy_quant_metadata(format_name="nvfp4"):
    """Create the comfy_quant metadata tensor."""
    json_bytes = json.dumps({"format": format_name}).encode("utf-8")
    return torch.tensor(list(json_bytes), dtype=torch.uint8)


def should_quantize(key, tensor):
    """Determine if a tensor should be NVFP4 quantized."""
    # Only quantize float weight tensors that are 2D and large enough
    if tensor.dtype not in FLOAT_DTYPES:
        return False
    if tensor.ndim != 2:
        return False
    if tensor.numel() < BLOCK_SIZE:
        return False
    # Only quantize actual weight matrices, not embeddings/norms/biases
    if not key.endswith(".weight"):
        return False
    # Skip embedding weights and norm weights (typically 1D but check anyway)
    for skip in ["embed", "norm", "lm_head"]:
        if skip in key.lower():
            return False
    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: quantize_nvfp4.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Try comfy_kitchen first
    ck = _try_comfy_kitchen()
    if ck is not None:
        print("Using comfy_kitchen for NVFP4 quantization (native)")
    else:
        print("comfy_kitchen not available, using pure PyTorch fallback")

    # Check if GPU is available and has enough memory
    use_gpu = ck is not None and torch.cuda.is_available()
    if use_gpu:
        try:
            torch.cuda.empty_cache()
            free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
            print(f"GPU free memory: {free_mem:.1f} GB")
            if free_mem < 1.0:
                print("Insufficient GPU memory, using CPU fallback")
                use_gpu = False
        except Exception:
            use_gpu = False

    print(f"Loading {input_file}...")
    tensors = load_file(input_file)

    output = {}
    converted = 0
    kept = 0

    for key in sorted(tensors.keys()):
        tensor = tensors[key]

        if should_quantize(key, tensor):
            print(f"  NVFP4: {key} {list(tensor.shape)} {tensor.dtype}")

            if ck is not None and use_gpu:
                try:
                    tensor_gpu = tensor.cuda()
                    packed, block_scale, tensor_scale = quantize_nvfp4_comfy_kitchen(
                        tensor_gpu, ck
                    )
                    packed = packed.cpu()
                    block_scale = block_scale.cpu()
                    tensor_scale = tensor_scale.cpu()
                    del tensor_gpu
                    torch.cuda.empty_cache()
                except Exception as e:
                    if converted == 0:
                        print(f"    GPU failed ({e}), falling back to CPU for all tensors")
                        use_gpu = False
                    else:
                        print(f"    GPU OOM on {key}, falling back to CPU")
                    torch.cuda.empty_cache()
                    packed, block_scale, tensor_scale = quantize_nvfp4_manual(tensor)
            else:
                packed, block_scale, tensor_scale = quantize_nvfp4_manual(tensor)

            # Store in ComfyUI format (ensure CPU for safetensors saving)
            output[key] = packed.cpu()
            block_scale_u8 = block_scale.view(torch.uint8) if block_scale.dtype != torch.uint8 else block_scale
            output[f"{key}_scale"] = block_scale_u8.cpu()
            output[f"{key}_scale_2"] = tensor_scale.cpu()

            # Add comfy_quant metadata for this layer
            # The key pattern: "model.layers.0.self_attn.q_proj.weight" -> "model.layers.0.self_attn.q_proj.comfy_quant"
            quant_key = key.rsplit(".", 1)[0] + ".comfy_quant"
            output[quant_key] = make_comfy_quant_metadata("nvfp4")

            converted += 1
        else:
            output[key] = tensor.cpu()
            kept += 1

    print(f"\nConverted {converted} tensors to NVFP4")
    print(f"Kept {kept} tensors unchanged")

    print(f"\nSaving NVFP4 model to {output_file}...")
    save_file(output, output_file)

    input_size = os.path.getsize(input_file) / (1024**3)
    output_size = os.path.getsize(output_file) / (1024**3)
    ratio = output_size / input_size * 100 if input_size > 0 else 0
    print(f"Done. {input_size:.2f} GB -> {output_size:.2f} GB ({ratio:.1f}%)")


if __name__ == "__main__":
    main()
