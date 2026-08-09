#!/usr/bin/env python3
"""
Quantize a safetensors model to INT4 W4A4 ConvRot.

Uses convert_to_quant (CTQ) for INT4 quantization with group-wise Hadamard
rotation (ConvRot) + row-wise symmetric INT4 quantization and SVD-guided
learned rounding (AdaRound) that minimizes output error. Produces
ComfyUI-compatible output with comfy_quant metadata (format=convrot_w4a4).

INT4 W4A4 ConvRot:
  - Signed INT4 [-7, 7] with per-row scaling, packed 2-per-byte (int8 storage)
  - Mandatory group-wise Hadamard rotation (spreads outliers for quality)
  - Learned rounding (AdaRound) optimizes each weight's rounding direction
  - ~50% smaller than INT8; accelerated INT4 MMA in ComfyUI 0.30.0+ via comfy-kitchen
  - dynamic_convrot auto-resolves a compatible group size per layer

Requirements: convert-to-quant, torch, safetensors
Optional: comfy-kitchen (CUDA/Triton kernels for accelerated INT4 inference)

Usage:
    python3 quantize_int4.py input.safetensors output.safetensors
"""

import os
import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: quantize_int4.py <input_file> <output_file>")
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

    print(f"INT4 W4A4 ConvRot quantization (learned rounding)")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"")
    print(f"NOTE: INT4 requires ConvRot; --dynamic-convrot auto-resolves a compatible")
    print(f"      group size per layer. Learned rounding (AdaRound) runs per-tensor SVD +")
    print(f"      gradient descent to maximize quality (~30dB vs ~20dB for plain RTN).")
    print(f"")

    quantize(
        input=input_file,
        output=output_file,
        int4=True,
        dynamic_convrot=True,
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
