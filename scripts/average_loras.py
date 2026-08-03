#!/usr/bin/env python3
"""Average multiple LoRA adapter weight tensors into a single adapter.

Supports three combination strategies:
  --method mean     Simple arithmetic mean of all adapter weights
  --method kl       Inverse-KL weighting (lower KL = higher weight)
  --method score    Inverse-score weighting (lower ref+kl*10 = higher weight)

Usage:
    python3 average_loras.py <adapter1> <adapter2> [<adapter3> ...] <output_dir>
        [--method mean|kl|score]
        [--metrics m1.json m2.json ...]

Each adapter path must contain adapter_config.json and adapter_model.safetensors.
The output_dir will contain the same files with averaged weights.
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def parse_args():
    p = argparse.ArgumentParser(description="Average multiple LoRA adapters")
    p.add_argument("adapters", nargs="+", help="Adapter dirs, last one is output")
    p.add_argument("--method", default="mean",
                   choices=["mean", "kl", "score"],
                   help="Combination strategy (default: mean)")
    p.add_argument("--metrics", nargs="*",
                   help="metrics.json files (one per input adapter, same order)")
    return p.parse_args()


def load_weights(adapter_dir):
    st_path = Path(adapter_dir) / "adapter_model.safetensors"
    return load_file(str(st_path))


def compute_weights(method, n, metrics_list):
    """Return normalized weights [w0, w1, ...] summing to 1."""
    if method == "mean":
        return [1.0 / n] * n

    raw = []
    for m in metrics_list:
        if method == "kl":
            val = m["kl_divergence"]
        else:
            val = m["refusal_count"] + m["kl_divergence"] * 10
        raw.append(1.0 / (val + 1e-8))

    total = sum(raw)
    return [r / total for r in raw]


def main():
    args = parse_args()

    *input_dirs, output_dir = args.adapters
    n = len(input_dirs)

    if n < 2:
        print("ERROR: need at least 2 input adapters", flush=True)
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = []
    if args.method != "mean":
        if not args.metrics or len(args.metrics) != n:
            print(f"ERROR: --metrics required for method '{args.method}' "
                  f"({n} files needed)", flush=True)
            return
        for mf in args.metrics:
            with open(mf) as f:
                metrics_list.append(json.load(f))

    weights = compute_weights(args.method, n, metrics_list)

    print(f"Combining {n} adapters ({args.method}):", flush=True)
    for i, (d, w) in enumerate(zip(input_dirs, weights)):
        print(f"  [{i}] w={w:.4f}  {Path(d).name}", flush=True)

    all_state_dicts = [load_weights(d) for d in input_dirs]

    keys = all_state_dicts[0].keys()
    for i, sd in enumerate(all_state_dicts[1:], 1):
        if set(sd.keys()) != set(keys):
            print(f"ERROR: key mismatch between adapter 0 and {i}", flush=True)
            return

    averaged = {}
    for key in keys:
        acc = None
        for i, sd in enumerate(all_state_dicts):
            t = sd[key] * weights[i]
            acc = t if acc is None else acc + t
        averaged[key] = acc.contiguous()

    out_st = output_dir / "adapter_model.safetensors"
    save_file(averaged, str(out_st), metadata={"format": "pt"})
    print(f"\nSaved averaged adapter: {out_st}", flush=True)

    shutil.copy2(
        Path(input_dirs[0]) / "adapter_config.json",
        output_dir / "adapter_config.json",
    )
    print(f"Copied adapter_config.json from {input_dirs[0]}", flush=True)

    combined_metrics = {
        "method": args.method,
        "num_adapters": n,
        "weights": dict(zip([Path(d).name for d in input_dirs], weights)),
        "source_metrics": metrics_list if metrics_list else None,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(combined_metrics, f, indent=2)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
