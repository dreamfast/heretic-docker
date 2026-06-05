#!/usr/bin/env python3
"""Compare key structures between working and abliterated safetensors models."""

import sys
from safetensors import safe_open


def load_keys(path):
    keys = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            keys[k] = f.get_tensor(k).shape
    return keys


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_models.py <working.safetensors> <abliterated.safetensors>")
        sys.exit(1)

    working_path = sys.argv[1]
    abliterated_path = sys.argv[2]

    print(f"Working:     {working_path}")
    print(f"Abliterated: {abliterated_path}")
    print()

    working = load_keys(working_path)
    abliterated = load_keys(abliterated_path)

    w_keys = set(working.keys())
    a_keys = set(abliterated.keys())

    only_working = sorted(w_keys - a_keys)
    only_abliterated = sorted(a_keys - w_keys)
    common = sorted(w_keys & a_keys)

    print(f"Working keys:     {len(w_keys)}")
    print(f"Abliterated keys:  {len(a_keys)}")
    print(f"Common keys:       {len(common)}")
    print(f"Only in working:   {len(only_working)}")
    print(f"Only in abliterated: {len(only_abliterated)}")
    print()

    # Show dimension mismatches in common keys
    mismatches = []
    for k in common:
        if working[k] != abliterated[k]:
            mismatches.append((k, working[k], abliterated[k]))

    if mismatches:
        print("=== DIMENSION MISMATCHES (common keys with different shapes) ===")
        for k, ws, als in mismatches:
            print(f"  {k}")
            print(f"    working:     {list(ws)}")
            print(f"    abliterated: {list(als)}")
        print()

    # Show keys only in working (grouped by prefix)
    if only_working:
        print("=== KEYS ONLY IN WORKING MODEL ===")
        prefixes = {}
        for k in only_working:
            prefix = k.split(".")[0]
            prefixes.setdefault(prefix, []).append(k)
        for prefix, keys in sorted(prefixes.items()):
            print(f"  [{prefix}] ({len(keys)} keys)")
            for k in keys[:5]:
                print(f"    {k} {list(working[k])}")
            if len(keys) > 5:
                print(f"    ... and {len(keys) - 5} more")
        print()

    # Show keys only in abliterated (grouped by prefix)
    if only_abliterated:
        print("=== KEYS ONLY IN ABLITERATED MODEL ===")
        prefixes = {}
        for k in only_abliterated:
            prefix = k.split(".")[0]
            prefixes.setdefault(prefix, []).append(k)
        for prefix, keys in sorted(prefixes.items()):
            print(f"  [{prefix}] ({len(keys)} keys)")
            for k in keys[:5]:
                print(f"    {k} {list(abliterated[k])}")
            if len(keys) > 5:
                print(f"    ... and {len(keys) - 5} more")
        print()

    # Key prefix summary
    print("=== KEY PREFIX SUMMARY ===")
    for label, keys_dict in [("Working", working), ("Abliterated", abliterated)]:
        prefixes = {}
        for k in keys_dict:
            parts = k.split(".")
            prefix = parts[0] if len(parts) > 1 else k
            prefixes.setdefault(prefix, set()).add(k)
        print(f"\n  {label}:")
        for p in sorted(prefixes):
            count = len(prefixes[p])
            sample_shapes = set()
            for k in sorted(prefixes[p])[:3]:
                sample_shapes.add(str(list(keys_dict[k])))
            print(f"    {p}.* ({count} keys, samples: {', '.join(list(sample_shapes)[:2])})")


if __name__ == "__main__":
    main()
