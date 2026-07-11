#!/usr/bin/env python3
"""
Quantize a safetensors model to INT8 (block-wise, with ConvRot learned rounding).

Uses convert_to_quant (CTQ) for high-quality INT8 quantization with SVD-based
learned rounding (AdaRound/ConvRot) that minimizes output error. Produces
ComfyUI-compatible output with comfy_quant metadata.

INT8 block-wise with ConvRot:
  - Symmetric INT8 [-127, 127] with per-block scaling
  - Learned rounding optimizes each weight's rounding direction
  - Works on any GPU (no Blackwell requirement)
  - Near-lossless quality vs naive round-to-nearest

Requirements: convert-to-quant, torch, safetensors
Optional: triton (for INT8 inference kernels), comfy-kitchen

Usage:
    python3 quantize_int8.py input.safetensors output.safetensors
"""

import os
import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: quantize_int8.py <input_file> <output_file>")
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

    print(f"INT8 block-wise quantization with ConvRot learned rounding")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"")
    print(f"NOTE: ConvRot optimization runs per-tensor SVD + gradient descent.")
    print(f"      This is slower than naive quantization but produces near-lossless INT8.")
    print(f"")

    quantize(
        input=input_file,
        output=output_file,
        int8=True,
        scaling_mode="tensor",
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
