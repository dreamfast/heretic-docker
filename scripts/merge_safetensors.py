#!/usr/bin/env python3
"""
Merge HuggingFace model shards into a single .safetensors file.
Keeps ALL keys including vision_tower and multi_modal_projector.

Usage:
    python3 merge_safetensors.py /output/hf-model /output/merged/model-full.safetensors
"""

import glob
import os
import sys

from safetensors.torch import load_file, save_file


def main():
    if len(sys.argv) < 3:
        print("Usage: merge_safetensors.py <model_dir> <output_file>")
        sys.exit(1)

    model_dir = sys.argv[1]
    output_file = sys.argv[2]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    shard_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shard_files:
        print(f"ERROR: No .safetensors files found in {model_dir}")
        sys.exit(1)

    print(f"Found {len(shard_files)} shard(s) in {model_dir}")

    all_tensors = {}
    for f in shard_files:
        print(f"  Loading {os.path.basename(f)}...")
        tensors = load_file(f)
        all_tensors.update(tensors)

    # Report what we have
    vision_keys = [k for k in all_tensors if k.startswith("vision_tower.")]
    projector_keys = [k for k in all_tensors if k.startswith("multi_modal_projector.")]
    language_keys = [k for k in all_tensors if k.startswith("language_model.")]
    other_keys = [
        k
        for k in all_tensors
        if not k.startswith(("vision_tower.", "multi_modal_projector.", "language_model."))
    ]

    print(f"\nKey summary:")
    print(f"  vision_tower.*:          {len(vision_keys)} tensors")
    print(f"  multi_modal_projector.*: {len(projector_keys)} tensors")
    print(f"  language_model.*:        {len(language_keys)} tensors")
    print(f"  other:                   {len(other_keys)} tensors")
    print(f"  TOTAL:                   {len(all_tensors)} tensors")

    print(f"\nSaving merged model to {output_file}...")
    save_file(all_tensors, output_file)

    size_gb = os.path.getsize(output_file) / (1024**3)
    print(f"Done. Output size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
