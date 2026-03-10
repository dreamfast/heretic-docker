#!/usr/bin/env python3
"""
Quantize a safetensors model to FP8 (float8_e4m3fn).

Converts large 2D float weight tensors to float8_e4m3fn.
Leaves biases, norms, embeddings, and small tensors in their original dtype
since ComfyUI performs element-wise ops (add) on those, and Blackwell GPUs
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
    # Only quantize 2D weight tensors (linear layers)
    if tensor.ndim != 2:
        return False
    # Skip small tensors
    if tensor.numel() < 256:
        return False
    # Skip embeddings, norms, biases by key name
    key_lower = key.lower()
    for skip in ["embed", "norm", "bias", "lm_head", "spiece_model"]:
        if skip in key_lower:
            return False
    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: quantize_fp8.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

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

    input_size = os.path.getsize(input_file) / (1024**3)
    output_size = os.path.getsize(output_file) / (1024**3)
    print(f"Done. {input_size:.2f} GB -> {output_size:.2f} GB ({output_size/input_size*100:.1f}%)")


if __name__ == "__main__":
    main()
