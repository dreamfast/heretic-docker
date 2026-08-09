#!/usr/bin/env python3
"""
Quantize a safetensors model to FP8 (float8_e4m3fn).

Primary: Uses convert_to_quant (CTQ) for FP8 with row-wise (per-row) scaling
          and SVD-guided learned rounding (AdaRound) that minimizes output error,
          with ComfyUI-compatible comfy_quant metadata (format=float8_e4m3fn_rowwise).
          Strictly better than naive cast: per-row scales preserve dynamic range,
          and learned rounding optimizes each weight's rounding direction.
Fallback: Pure PyTorch naive cast (.to(float8_e4m3fn)) when CTQ is unavailable.

Both modes leave biases, norms, embeddings, and small tensors in their original
dtype since ComfyUI performs element-wise ops (add) on those, and Blackwell GPUs
don't support ufunc_add for Float8_e4m3fn.

Based on: https://nathan.sapwell.net/posts/heretic-gemma-12b/

Usage:
    python3 quantize_fp8.py /output/comfyui/model.safetensors /output/comfyui/model_fp8.safetensors
"""

import os
import sys

import torch
from safetensors.torch import load_file, save_file

FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def should_quantize_fp8(key, tensor):
    """Only quantize large 2D weight matrices, not biases/norms/embeddings."""
    if tensor.dtype not in FLOAT_DTYPES:
        return False
    if tensor.ndim != 2:
        return False
    if tensor.numel() < 256:
        return False
    key_lower = key.lower()
    for skip in ["embed", "norm", "bias", "lm_head", "spiece_model"]:
        if skip in key_lower:
            return False
    return True


def quantize_ctq(input_file, output_file):
    """FP8 quantization via convert_to_quant (row-wise scaling + learned rounding)."""
    from convert_to_quant import quantize

    print("Using convert_to_quant for FP8 (row-wise scaling + learned rounding, comfy_quant)")
    quantize(
        input=input_file,
        output=output_file,
        scaling_mode="row",
        comfy_quant=True,
        save_quant_metadata=True,
        low_memory=True,
        device="cuda",
        exclude_layers=r"(embed|norm|bias|lm_head|spiece_model|multi_modal_projector|patch_embed|patch_conv)",
        fallback_simple=True,
        verbose="VERBOSE",
    )


def quantize_naive(input_file, output_file):
    """FP8 quantization via naive PyTorch cast (fallback when CTQ unavailable)."""
    print("WARNING: convert-to-quant not available, using naive FP8 cast (no scaling)")
    print(f"Loading {input_file}...")
    tensors = load_file(input_file)

    fp8_tensors = {}
    converted = 0
    kept = 0

    for k, v in tensors.items():
        if should_quantize_fp8(k, v):
            fp8_tensors[k] = v.cpu().to(torch.float8_e4m3fn)
            converted += 1
        else:
            fp8_tensors[k] = v.cpu()
            kept += 1

    print(f"Converted {converted} tensors to float8_e4m3fn")
    print(f"Kept {kept} tensors unchanged (biases, norms, embeddings, non-float)")

    print(f"\nSaving FP8 model to {output_file}...")
    save_file(fp8_tensors, output_file, metadata={"format": "pt"})


def main():
    if len(sys.argv) < 3:
        print("Usage: quantize_fp8.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    used_ctq = False
    try:
        quantize_ctq(input_file, output_file)
        used_ctq = True
    except ImportError:
        quantize_naive(input_file, output_file)
    except Exception as e:
        print(f"convert_to_quant failed ({e}), falling back to naive cast")
        quantize_naive(input_file, output_file)

    input_size = os.path.getsize(input_file) / (1024**3)
    output_size = os.path.getsize(output_file) / (1024**3)
    pct = output_size / input_size * 100 if input_size > 0 else 0
    method = "CTQ (row-wise + learned rounding)" if used_ctq else "naive cast (unscaled)"
    print(f"Done [{method}]. {input_size:.2f} GB -> {output_size:.2f} GB ({pct:.1f}%)")


if __name__ == "__main__":
    main()
