#!/usr/bin/env python3
"""Compare tensor differences between a base model and one or more abliterated models.

Loads model weights from HuggingFace-format directories (sharded safetensors),
computes per-tensor L2 distance / cosine similarity / max abs diff vs base,
and prints a ranked summary. Optionally writes per-tensor CSV.

Usage:
    python3 tensor_diff.py <base_dir> <model_a_dir> [<model_b_dir> ...]
        [--csv out.csv] [--top-k 20]
"""

import argparse
import csv
from pathlib import Path

from safetensors import safe_open


def parse_args():
    p = argparse.ArgumentParser(description="Compare tensor differences vs base model")
    p.add_argument("base_dir", help="Base model directory (HuggingFace format)")
    p.add_argument("models", nargs="+", help="One or more model dirs to compare")
    p.add_argument("--csv", default=None, help="Write per-tensor CSV to this path")
    p.add_argument("--top-k", type=int, default=20, help="Show top-K most changed tensors")
    return p.parse_args()


def load_sharded(model_dir):
    """Load all tensors from a sharded HuggingFace safetensors model."""
    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors in {model_dir}")
    tensors = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def compare(base_tensors, model_tensors):
    """Return list of (key, l2, cos_sim, max_abs, rel_l2) tuples."""
    import torch
    import torch.nn.functional as F

    results = []
    common = sorted(set(base_tensors) & set(model_tensors))

    for key in common:
        b = base_tensors[key].float().flatten()
        m = model_tensors[key].float().flatten()

        diff = m - b
        l2 = diff.norm().item()
        max_abs = diff.abs().max().item()
        base_norm = b.norm().item()
        rel_l2 = l2 / base_norm if base_norm > 0 else 0.0

        cos = F.cosine_similarity(b.unsqueeze(0), m.unsqueeze(0)).item()

        results.append((key, l2, cos, max_abs, rel_l2))

    return results


def main():
    args = parse_args()

    print(f"Loading base: {args.base_dir}", flush=True)
    base = load_sharded(args.base_dir)
    print(f"  {len(base)} tensors\n", flush=True)

    all_results = {}

    for model_dir in args.models:
        label = Path(model_dir).name
        print(f"Comparing: {label}", flush=True)
        model = load_sharded(model_dir)

        only_base = set(base) - set(model)
        only_model = set(model) - set(base)
        if only_base or only_model:
            print(f"  WARNING: key mismatch ({len(only_base)} only in base, "
                  f"{len(only_model)} only in model)", flush=True)

        results = compare(base, model)
        results.sort(key=lambda x: x[1], reverse=True)
        all_results[label] = results

        total_l2 = sum(r[1] ** 2 for r in results) ** 0.5
        avg_cos = sum(r[2] for r in results) / len(results) if results else 0
        avg_rel = sum(r[4] for r in results) / len(results) if results else 0

        print(f"  Total L2:      {total_l2:.6f}", flush=True)
        print(f"  Avg cosine:    {avg_cos:.8f}", flush=True)
        print(f"  Avg relative:  {avg_rel:.8f}", flush=True)
        print(f"  Top-{args.top_k} most changed:\n", flush=True)
        print(f"    {'L2':>12s} {'COS':>10s} {'MAX_ABS':>12s} {'REL_L2':>10s}  KEY", flush=True)
        for key, l2, cos, max_abs, rel_l2 in results[:args.top_k]:
            short = key.replace("base_model.model.", "").replace("language_model.", "")
            print(f"    {l2:12.6f} {cos:10.8f} {max_abs:12.8f} {rel_l2:10.8f}  {short}", flush=True)
        print(flush=True)

        del model

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "key", "l2", "cosine", "max_abs", "rel_l2"])
            for label, results in all_results.items():
                for key, l2, cos, max_abs, rel_l2 in results:
                    writer.writerow([label, key, l2, cos, max_abs, rel_l2])
        print(f"CSV written to: {args.csv}", flush=True)

    if len(all_results) >= 2:
        labels = list(all_results.keys())
        print("=== INTER-MODEL COMPARISON ===", flush=True)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a = {r[0]: r[1] for r in all_results[labels[i]]}
                b = {r[0]: r[1] for r in all_results[labels[j]]}
                common_keys = sorted(set(a) & set(b))
                total_diff = sum(
                    (a[k] - b[k]) ** 2 for k in common_keys
                ) ** 0.5
                print(f"  {labels[i]} vs {labels[j]}:  L2={total_diff:.6f}", flush=True)


if __name__ == "__main__":
    main()
