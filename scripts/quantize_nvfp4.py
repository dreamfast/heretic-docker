#!/usr/bin/env python3
"""
Quantize a safetensors model to NVFP4 (NVIDIA FP4 E2M1, block-scaled).

Uses convert_to_quant (CTQ) for NVFP4 quantization with SVD-guided learned
rounding (AdaRound). Produces ComfyUI-compatible output with comfy_quant
metadata (format=nvfp4, group_size=16).

NVFP4 (NVIDIA FP4):
  - FP4 E2M1 data (packed 2-per-byte as uint8) with FP8 E4M3 per-block scales
  - 16-element blocks (vs MXFP8's 32-element blocks)
  - Double quantization: per-tensor scale + per-block FP8 scale
  - Learned rounding optimizes each weight's rounding direction
  - Best ratio of quality to size among the FP4 formats
  - Inference requires SM100+ (Blackwell); comfy-kitchen provides the kernels

Requirements: convert-to-quant, torch, safetensors, comfy-kitchen
Optional: comfy-kitchen (CUDA/Triton kernels for NVFP4 quantization)

Usage:
    python3 quantize_nvfp4.py input.safetensors output.safetensors
"""

import os
import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: quantize_nvfp4.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    try:
        from convert_to_quant import quantize
    except ImportError:
        print("ERROR: convert-to-quant not installed.")
        print("       Install with: pip install convert-to-quant")
        sys.exit(1)

    print(f"NVFP4 (NVIDIA FP4 E2M1) quantization (learned rounding)")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"")

    quantize(
        input=input_file,
        output=output_file,
        nvfp4=True,
        comfy_quant=True,
        save_quant_metadata=True,
        low_memory=True,
        device="cuda",
        exclude_layers=r"(embed|norm|bias|lm_head|spiece_model|multi_modal_projector|patch_embed|patch_conv)",
        fallback_simple=True,
        verbose="VERBOSE",
    )

    input_size = os.path.getsize(input_file) / (1024**3)
    output_size = os.path.getsize(output_file) / (1024**3)
    ratio = output_size / input_size * 100 if input_size > 0 else 0
    print(f"\nDone. {input_size:.2f} GB -> {output_size:.2f} GB ({ratio:.1f}%)")


if __name__ == "__main__":
    main()
