#!/usr/bin/env python3
"""
Quantize a safetensors model to MXFP8 (Microscaling FP8, OCP MX standard).

Uses convert_to_quant (CTQ) for MXFP8 quantization with optional learned
rounding. Produces ComfyUI-compatible output with comfy_quant metadata.

MXFP8 (Microscaling FP8):
  - FP8 E4M3 data with E8M0 (power-of-2 exponent) per-block scales
  - 32-element blocks (vs NVFP4's 16-element blocks)
  - OCP MX standard format (vs NVIDIA-specific NVFP4)
  - Better dynamic range handling than per-tensor FP8
  - Requires SM100+ (Blackwell) for hardware-accelerated dequant
  - comfy-kitchen provides CUDA kernels; PyTorch fallback otherwise

Requirements: convert-to-quant, torch, safetensors
Optional: comfy-kitchen (CUDA/Triton kernels for faster quantization)

Usage:
    python3 quantize_mxfp8.py input.safetensors output.safetensors
"""

import os
import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: quantize_mxfp8.py <input_file> <output_file>")
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

    print(f"MXFP8 (Microscaling FP8) quantization")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"")

    quantize(
        input=input_file,
        output=output_file,
        mxfp8=True,
        comfy_quant=True,
        save_quant_metadata=True,
        low_memory=True,
        simple=True,
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
