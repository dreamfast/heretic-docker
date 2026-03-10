#!/usr/bin/env python3
"""
Convert abliterated model to ComfyUI format WITH vision capabilities preserved.

- Keeps vision_tower.* and multi_modal_projector.* keys
- Removes language_model.* prefix from language model keys
- Embeds tokenizer.model as spiece_model tensor

Usage:
    python3 convert_comfyui_vision.py /output/hf-model /output/comfyui/model.safetensors
"""

import glob
import os
import sys

import torch
from safetensors.torch import load_file, save_file


def main():
    if len(sys.argv) < 3:
        print("Usage: convert_comfyui_vision.py <model_dir> <output_file>")
        sys.exit(1)

    model_dir = sys.argv[1]
    output_file = sys.argv[2]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load all shards
    shard_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shard_files:
        print(f"ERROR: No .safetensors files found in {model_dir}")
        sys.exit(1)

    print(f"Loading {len(shard_files)} shard(s)...")
    all_tensors = {}
    for f in shard_files:
        print(f"  {os.path.basename(f)}")
        all_tensors.update(load_file(f))

    # Rename language_model.* keys but KEEP vision_tower.* and multi_modal_projector.*
    renamed = {}
    vision_keys = 0
    projector_keys = 0
    language_keys = 0

    for k, v in all_tensors.items():
        if k.startswith("vision_tower."):
            # Strip "vision_tower." prefix - ComfyUI expects "vision_model.*" not "vision_tower.vision_model.*"
            new_key = k[len("vision_tower."):]
            renamed[new_key] = v
            vision_keys += 1
        elif k.startswith("multi_modal_projector."):
            renamed[k] = v
            projector_keys += 1
        elif k.startswith("language_model."):
            new_key = k[len("language_model."):]
            renamed[new_key] = v
            language_keys += 1
        else:
            renamed[k] = v

    print(f"\nKey summary:")
    print(f"  vision_tower.*:          {vision_keys} tensors (preserved)")
    print(f"  multi_modal_projector.*: {projector_keys} tensors (preserved)")
    print(f"  language_model.*:        {language_keys} tensors (prefix stripped)")
    print(f"  TOTAL:                   {len(renamed)} tensors")

    # Embed tokenizer
    tokenizer_path = os.path.join(model_dir, "tokenizer.model")
    if os.path.exists(tokenizer_path):
        print(f"Embedding tokenizer from {tokenizer_path}")
        with open(tokenizer_path, "rb") as f:
            tokenizer_bytes = f.read()
        renamed["spiece_model"] = torch.frombuffer(
            bytearray(tokenizer_bytes), dtype=torch.uint8
        )
    else:
        print("WARNING: tokenizer.model not found, skipping tokenizer embedding")

    print(f"\nSaving ComfyUI vision format to {output_file}...")
    save_file(renamed, output_file)

    size_gb = os.path.getsize(output_file) / (1024**3)
    print(f"Done. Output size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
